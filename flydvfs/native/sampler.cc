// In-process CUPTI PM sampler. Loaded with LD_PRELOAD into the application.
// Sampling starts on the first kernel launch, which is after that process has
// a CUDA context. One pass collects SM active cycles, SM elapsed cycles, and
// DRAM bytes. Samples and session events are JSON lines on FLYDVFS_SAMPLE_PATH.
//
// Windows use GPU time (CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_TIME_INTERVAL).
// FLYDVFS_TRIGGER=sysclk forces the SYSCLK trigger. FLYDVFS_TRIGGER=time fails
// if GPU time is rejected. The default, auto, falls back to SYSCLK.
// The hardware buffer stays 64 MiB and KEEP_OLDEST. KEEP_LATEST stopped
// sampling and reported no error. On GB202 that buffer goes bad after about
// 300–370 s, so the session is rebuilt on a time limit as well as a sample limit.

#include <cuda.h>
#include <cupti.h>
#include <cupti_callbacks.h>
#include <cupti_pmsampling.h>
#include <cupti_profiler_host.h>
#include <cupti_profiler_target.h>
#include <cupti_target.h>

#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Metrics {
    double sm_active = 0;
    double sm_elapsed = 0;
    double dram_bytes = 0;
    uint64_t start_ns = 0;
    uint64_t end_ns = 0;
};

enum class TriggerRequest { Auto, Time, Sysclk };

const char* kMetricNames[] = {
    "sm__cycles_active.sum",
    "sm__cycles_elapsed.sum",
    "dram__bytes.sum",
};

const int kEmptyRestartLimit = 5;
const double kUnknownGraceSeconds = 2.0;

std::once_flag g_once;
std::atomic<bool> g_stop{false};
std::atomic<bool> g_seen_launch{false};
std::atomic<int> g_depth{0};
std::thread g_decode;
int g_fifo = -1;

CUpti_Profiler_Host_Object* g_host = nullptr;
CUpti_PmSampling_Object* g_sampler = nullptr;
std::vector<uint8_t> g_counter_image;
std::vector<uint8_t> g_config;
std::string g_chip;
int g_device = 0;
bool g_use_time = true;
uint64_t g_interval = 250000000ULL;
uint64_t g_restart_samples = 1500;
double g_restart_seconds = 120;
bool g_drop_first = true;
bool g_fatal = false;
int g_empty_restarts = 0;
uint64_t g_session_samples = 0;
std::chrono::steady_clock::time_point g_session_start;

void fail(const std::string& what) {
    std::cerr << "[flydvfs] " << what << std::endl;
}

std::string cupti_error(CUptiResult status) {
    const char* text = nullptr;
    cuptiGetResultString(status, &text);
    return text ? text : std::to_string(static_cast<int>(status));
}

void check(CUptiResult status, const char* what) {
    if (status != CUPTI_SUCCESS) {
        throw std::runtime_error(std::string(what) + ": " + cupti_error(status));
    }
}

uint64_t env_u64(const char* name, uint64_t fallback) {
    const char* env = std::getenv(name);
    if (env == nullptr || *env == '\0') {
        return fallback;
    }
    char* end = nullptr;
    unsigned long long value = std::strtoull(env, &end, 10);
    if (end == env || value == 0) {
        return fallback;
    }
    return value;
}

double env_seconds(const char* name, double fallback) {
    const char* env = std::getenv(name);
    if (env == nullptr || *env == '\0') {
        return fallback;
    }
    char* end = nullptr;
    double value = std::strtod(env, &end);
    if (end == env || !(value > 0) || !std::isfinite(value)) {
        return fallback;
    }
    return value;
}

TriggerRequest trigger_request() {
    const char* env = std::getenv("FLYDVFS_TRIGGER");
    if (env == nullptr || *env == '\0' || std::strcmp(env, "auto") == 0) {
        return TriggerRequest::Auto;
    }
    if (std::strcmp(env, "time") == 0) {
        return TriggerRequest::Time;
    }
    if (std::strcmp(env, "sysclk") == 0) {
        return TriggerRequest::Sysclk;
    }
    throw std::runtime_error(std::string("FLYDVFS_TRIGGER must be auto, time, or sysclk, got ") + env);
}

int device_index() {
    const char* env = std::getenv("FLYDVFS_GPU");
    if (env == nullptr || *env == '\0') {
        return 0;
    }
    return std::atoi(env);
}

std::string json_escape(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (char c : text) {
        switch (c) {
        case '\\':
        case '"':
            out.push_back('\\');
            out.push_back(c);
            break;
        case '\n':
            out += "\\n";
            break;
        default:
            out.push_back(c);
            break;
        }
    }
    return out;
}

std::string json_number(double value) {
    if (!std::isfinite(value)) {
        return "null";
    }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.0f", value);
    return buf;
}

void write_line(const std::string& line) {
    if (g_fifo < 0) {
        return;
    }
    std::string data = line;
    data.push_back('\n');
    const char* cursor = data.data();
    size_t left = data.size();
    while (left > 0) {
        ssize_t wrote = ::write(g_fifo, cursor, left);
        if (wrote < 0) {
            if (errno == EINTR) {
                continue;
            }
            return;
        }
        cursor += wrote;
        left -= static_cast<size_t>(wrote);
    }
}

void write_sample(const Metrics& sample) {
    write_line(
        "{\"sm_active\":" + json_number(sample.sm_active) +
        ",\"sm_elapsed\":" + json_number(sample.sm_elapsed) +
        ",\"dram_bytes\":" + json_number(sample.dram_bytes) +
        ",\"start_ns\":" + std::to_string(sample.start_ns) +
        ",\"end_ns\":" + std::to_string(sample.end_ns) + "}");
}

std::string json_seconds(double value) {
    char buf[64];
    if (std::floor(value) == value && value < 1e15) {
        std::snprintf(buf, sizeof(buf), "%.0f", value);
    } else {
        std::snprintf(buf, sizeof(buf), "%.6f", value);
    }
    return buf;
}

void write_config() {
    write_line(
        "{\"event\":\"config\",\"chip\":\"" + json_escape(g_chip) +
        "\",\"trigger\":\"" + (g_use_time ? "gpu_time" : "sysclk") +
        "\",\"interval\":" + std::to_string(g_interval) +
        ",\"restart_samples\":" + std::to_string(g_restart_samples) +
        ",\"restart_seconds\":" + json_seconds(g_restart_seconds) + "}");
}

void write_restart(const std::string& reason, long long ms, uint64_t session_samples) {
    write_line(
        "{\"event\":\"restart\",\"reason\":\"" + json_escape(reason) +
        "\",\"ms\":" + std::to_string(ms) +
        ",\"session_samples\":" + std::to_string(session_samples) + "}");
}

void write_stopped(const std::string& reason) {
    write_line("{\"event\":\"stopped\",\"reason\":\"" + json_escape(reason) + "\"}");
}

void load_limits() {
    g_restart_samples = env_u64("FLYDVFS_RESTART_SAMPLES", 1500);
    g_restart_seconds = env_seconds("FLYDVFS_RESTART_SECONDS", 120);
}

CUptiResult apply_config() {
    CUpti_PmSampling_SetConfig_Params set_config{CUpti_PmSampling_SetConfig_Params_STRUCT_SIZE};
    set_config.pPmSamplingObject = g_sampler;
    set_config.configSize = g_config.size();
    set_config.pConfig = g_config.data();
    set_config.hardwareBufferSize = 64 * 1024 * 1024;
    set_config.samplingInterval = g_interval;
    set_config.triggerMode = g_use_time
        ? CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_TIME_INTERVAL
        : CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_SYSCLK_INTERVAL;
    set_config.hwBufferAppendMode = CUPTI_PM_SAMPLING_HARDWARE_BUFFER_APPEND_MODE_KEEP_OLDEST;
    return cuptiPmSamplingSetConfig(&set_config);
}

void enable_sampler() {
    CUpti_PmSampling_Enable_Params enable{CUpti_PmSampling_Enable_Params_STRUCT_SIZE};
    enable.deviceIndex = static_cast<size_t>(g_device);
    check(cuptiPmSamplingEnable(&enable), "cuptiPmSamplingEnable");
    g_sampler = enable.pPmSamplingObject;
}

void stop_and_disable() {
    if (g_sampler == nullptr) {
        return;
    }
    CUpti_PmSampling_Stop_Params stop{CUpti_PmSampling_Stop_Params_STRUCT_SIZE};
    stop.pPmSamplingObject = g_sampler;
    cuptiPmSamplingStop(&stop);
    CUpti_PmSampling_Disable_Params disable{CUpti_PmSampling_Disable_Params_STRUCT_SIZE};
    disable.pPmSamplingObject = g_sampler;
    cuptiPmSamplingDisable(&disable);
    g_sampler = nullptr;
}

void init_counter_image() {
    // The image belongs to the current PM sampling object. Reusing the previous
    // session's image makes CounterDataImageInitialize return INVALID_OPERATION.
    CUpti_PmSampling_GetCounterDataSize_Params data_size{CUpti_PmSampling_GetCounterDataSize_Params_STRUCT_SIZE};
    data_size.pPmSamplingObject = g_sampler;
    data_size.numMetrics = 3;
    data_size.pMetricNames = kMetricNames;
    data_size.maxSamples = 128;
    check(cuptiPmSamplingGetCounterDataSize(&data_size), "cuptiPmSamplingGetCounterDataSize");
    g_counter_image.assign(data_size.counterDataSize, 0);
    CUpti_PmSampling_CounterDataImage_Initialize_Params init_image{CUpti_PmSampling_CounterDataImage_Initialize_Params_STRUCT_SIZE};
    init_image.pPmSamplingObject = g_sampler;
    init_image.counterDataSize = g_counter_image.size();
    init_image.pCounterData = g_counter_image.data();
    check(cuptiPmSamplingCounterDataImageInitialize(&init_image), "cuptiPmSamplingCounterDataImageInitialize");
}

void start_pm() {
    CUpti_PmSampling_Start_Params start{CUpti_PmSampling_Start_Params_STRUCT_SIZE};
    start.pPmSamplingObject = g_sampler;
    check(cuptiPmSamplingStart(&start), "cuptiPmSamplingStart");
}

void begin_session() {
    g_drop_first = true;
    g_session_samples = 0;
    g_session_start = std::chrono::steady_clock::now();
}

void select_trigger() {
    TriggerRequest request = trigger_request();
    if (request == TriggerRequest::Sysclk) {
        g_use_time = false;
        g_interval = env_u64("FLYDVFS_INTERVAL_CYCLES", 100000000ULL);
        check(apply_config(), "cuptiPmSamplingSetConfig");
        return;
    }
    g_use_time = true;
    g_interval = env_u64("FLYDVFS_INTERVAL_NS", 250000000ULL);
    CUptiResult status = apply_config();
    if (status == CUPTI_SUCCESS) {
        return;
    }
    if (request == TriggerRequest::Time) {
        throw std::runtime_error("GPU time trigger rejected: " + cupti_error(status));
    }
    fail("GPU time trigger rejected (" + cupti_error(status) + "), using SYSCLK");
    stop_and_disable();
    enable_sampler();
    g_use_time = false;
    g_interval = env_u64("FLYDVFS_INTERVAL_CYCLES", 100000000ULL);
    check(apply_config(), "cuptiPmSamplingSetConfig");
}

double session_seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - g_session_start).count();
}

bool restart_session(const std::string& reason) {
    uint64_t produced = g_session_samples;
    if (produced == 0) {
        g_empty_restarts += 1;
        if (g_empty_restarts >= kEmptyRestartLimit) {
            stop_and_disable();
            write_stopped("no_windows");
            std::cerr << "[flydvfs] stopped no_windows" << std::endl;
            g_fatal = true;
            return false;
        }
    } else {
        g_empty_restarts = 0;
    }
    auto started = std::chrono::steady_clock::now();
    try {
        stop_and_disable();
        enable_sampler();
        check(apply_config(), "cuptiPmSamplingSetConfig");
        init_counter_image();
        start_pm();
        begin_session();
        long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - started)
                           .count();
        write_restart(reason, ms, produced);
        return true;
    } catch (const std::exception& exc) {
        fail(std::string("restart failed: ") + exc.what());
        write_stopped(exc.what());
        g_fatal = true;
        return false;
    }
}

bool budget_reason(std::string& reason) {
    if (session_seconds() >= g_restart_seconds) {
        reason = "restart_seconds";
        return true;
    }
    if (g_session_samples >= g_restart_samples) {
        reason = "restart_samples";
        return true;
    }
    return false;
}

void emit_sample(const Metrics& sample) {
    // The first completed window of a session starts before the session, so
    // its length is not the configured interval.
    if (g_drop_first) {
        g_drop_first = false;
        return;
    }
    write_sample(sample);
    g_session_samples += 1;
}

void emit_completed(size_t count) {
    const char* names[] = {kMetricNames[0], kMetricNames[1], kMetricNames[2]};
    for (size_t index = 0; index < count; ++index) {
        CUpti_PmSampling_CounterData_GetSampleInfo_Params sample_info{CUpti_PmSampling_CounterData_GetSampleInfo_Params_STRUCT_SIZE};
        sample_info.pPmSamplingObject = g_sampler;
        sample_info.pCounterDataImage = g_counter_image.data();
        sample_info.counterDataImageSize = g_counter_image.size();
        sample_info.sampleIndex = index;
        check(cuptiPmSamplingCounterDataGetSampleInfo(&sample_info), "cuptiPmSamplingCounterDataGetSampleInfo");

        double values[3] = {};
        CUpti_Profiler_Host_EvaluateToGpuValues_Params evaluate{CUpti_Profiler_Host_EvaluateToGpuValues_Params_STRUCT_SIZE};
        evaluate.pHostObject = g_host;
        evaluate.pCounterDataImage = g_counter_image.data();
        evaluate.counterDataImageSize = g_counter_image.size();
        evaluate.ppMetricNames = names;
        evaluate.numMetrics = 3;
        evaluate.rangeIndex = index;
        evaluate.pMetricValues = values;
        check(cuptiProfilerHostEvaluateToGpuValues(&evaluate), "cuptiProfilerHostEvaluateToGpuValues");

        Metrics sample;
        sample.sm_active = values[0];
        sample.sm_elapsed = values[1];
        sample.dram_bytes = values[2];
        sample.start_ns = sample_info.startTimestamp;
        sample.end_ns = sample_info.endTimestamp;
        emit_sample(sample);
    }
}

bool setup_sampling(int device) {
    g_device = device;
    load_limits();
    CUdevice cu_device;
    if (cuDeviceGet(&cu_device, device) != CUDA_SUCCESS) {
        fail("cuDeviceGet failed");
        return false;
    }
    CUpti_Profiler_DeviceSupported_Params supported{CUpti_Profiler_DeviceSupported_Params_STRUCT_SIZE};
    supported.cuDevice = cu_device;
    supported.api = CUPTI_PROFILER_PM_SAMPLING;
    check(cuptiProfilerDeviceSupported(&supported), "cuptiProfilerDeviceSupported");
    if (supported.isSupported != CUPTI_PROFILER_CONFIGURATION_SUPPORTED) {
        fail("PM sampling is not supported on this GPU");
        return false;
    }

    CUpti_Profiler_Initialize_Params profiler_init{CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
    check(cuptiProfilerInitialize(&profiler_init), "cuptiProfilerInitialize");

    CUpti_Device_GetChipName_Params chip{CUpti_Device_GetChipName_Params_STRUCT_SIZE};
    chip.deviceIndex = static_cast<size_t>(device);
    check(cuptiDeviceGetChipName(&chip), "cuptiDeviceGetChipName");
    g_chip = chip.pChipName;

    CUpti_PmSampling_GetCounterAvailability_Params availability{CUpti_PmSampling_GetCounterAvailability_Params_STRUCT_SIZE};
    availability.deviceIndex = static_cast<size_t>(device);
    check(cuptiPmSamplingGetCounterAvailability(&availability), "cuptiPmSamplingGetCounterAvailability");
    std::vector<uint8_t> availability_image(availability.counterAvailabilityImageSize);
    availability.pCounterAvailabilityImage = availability_image.data();
    check(cuptiPmSamplingGetCounterAvailability(&availability), "cuptiPmSamplingGetCounterAvailability");

    CUpti_Profiler_Host_Initialize_Params host_init{CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE};
    host_init.profilerType = CUPTI_PROFILER_TYPE_PM_SAMPLING;
    host_init.pChipName = g_chip.c_str();
    host_init.pCounterAvailabilityImage = availability_image.data();
    check(cuptiProfilerHostInitialize(&host_init), "cuptiProfilerHostInitialize");
    g_host = host_init.pHostObject;

    CUpti_Profiler_Host_ConfigAddMetrics_Params add{CUpti_Profiler_Host_ConfigAddMetrics_Params_STRUCT_SIZE};
    add.pHostObject = g_host;
    add.ppMetricNames = kMetricNames;
    add.numMetrics = 3;
    check(cuptiProfilerHostConfigAddMetrics(&add), "cuptiProfilerHostConfigAddMetrics");

    CUpti_Profiler_Host_GetConfigImageSize_Params config_size{CUpti_Profiler_Host_GetConfigImageSize_Params_STRUCT_SIZE};
    config_size.pHostObject = g_host;
    check(cuptiProfilerHostGetConfigImageSize(&config_size), "cuptiProfilerHostGetConfigImageSize");
    g_config.resize(config_size.configImageSize);
    CUpti_Profiler_Host_GetConfigImage_Params config_image{CUpti_Profiler_Host_GetConfigImage_Params_STRUCT_SIZE};
    config_image.pHostObject = g_host;
    config_image.pConfigImage = g_config.data();
    config_image.configImageSize = g_config.size();
    check(cuptiProfilerHostGetConfigImage(&config_image), "cuptiProfilerHostGetConfigImage");

    CUpti_Profiler_Host_GetNumOfPasses_Params passes{CUpti_Profiler_Host_GetNumOfPasses_Params_STRUCT_SIZE};
    passes.pConfigImage = g_config.data();
    passes.configImageSize = g_config.size();
    check(cuptiProfilerHostGetNumOfPasses(&passes), "cuptiProfilerHostGetNumOfPasses");
    if (passes.numOfPasses != 1) {
        throw std::runtime_error("metrics need more than one pass");
    }

    enable_sampler();
    select_trigger();
    init_counter_image();
    start_pm();
    begin_session();
    write_config();
    std::cerr << "[flydvfs] sampling pid " << getpid() << " chip " << g_chip
              << " trigger " << (g_use_time ? "gpu_time" : "sysclk")
              << " interval " << g_interval
              << " restart_samples " << g_restart_samples
              << " restart_seconds " << g_restart_seconds << std::endl;
    return true;
}

void decode_loop() {
    CUcontext context = nullptr;
    CUdevice device;
    if (cuDeviceGet(&device, g_device) == CUDA_SUCCESS) {
        cuDevicePrimaryCtxRetain(&context, device);
        cuCtxSetCurrent(context);
    }
    while (!g_stop.load() && !g_fatal) {
        std::string reason;
        if (budget_reason(reason)) {
            if (!restart_session(reason)) {
                break;
            }
            continue;
        }
        try {
            CUpti_PmSampling_DecodeData_Params decode{CUpti_PmSampling_DecodeData_Params_STRUCT_SIZE};
            decode.pPmSamplingObject = g_sampler;
            decode.pCounterDataImage = g_counter_image.data();
            decode.counterDataImageSize = g_counter_image.size();
            CUptiResult decoded = cuptiPmSamplingDecodeData(&decode);
            if (decoded == CUPTI_ERROR_UNKNOWN || decoded == CUPTI_ERROR_NOT_READY) {
                // CUPTI returns these before the first record of a session exists.
                if (g_session_samples == 0 && session_seconds() < kUnknownGraceSeconds) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                    continue;
                }
                if (!restart_session("decode:" + cupti_error(decoded))) {
                    break;
                }
                continue;
            }

            size_t completed = 0;
            if (decoded == CUPTI_SUCCESS || decoded == CUPTI_ERROR_OUT_OF_MEMORY) {
                CUpti_PmSampling_GetCounterDataInfo_Params info{CUpti_PmSampling_GetCounterDataInfo_Params_STRUCT_SIZE};
                info.pCounterDataImage = g_counter_image.data();
                info.counterDataImageSize = g_counter_image.size();
                if (cuptiPmSamplingGetCounterDataInfo(&info) == CUPTI_SUCCESS) {
                    completed = info.numCompletedSamples;
                    if (completed > 0) {
                        emit_completed(completed);
                        init_counter_image();
                    }
                }
            }

            bool overflow = decoded == CUPTI_ERROR_OUT_OF_MEMORY || decode.overflow != 0;
            bool counter_full = decoded == CUPTI_SUCCESS &&
                decode.decodeStopReason == CUPTI_PM_SAMPLING_DECODE_STOP_REASON_COUNTER_DATA_FULL &&
                completed == 0;
            if (overflow) {
                if (!restart_session("overflow")) {
                    break;
                }
                continue;
            }
            if (counter_full) {
                if (!restart_session("counter_data_full")) {
                    break;
                }
                continue;
            }
            if (decoded != CUPTI_SUCCESS) {
                if (!restart_session("decode:" + cupti_error(decoded))) {
                    break;
                }
                continue;
            }
        } catch (const std::exception& exc) {
            fail(exc.what());
            if (!restart_session(exc.what())) {
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

void shutdown_sampling() {
    g_stop.store(true);
    if (g_decode.joinable()) {
        g_decode.join();
    }
    stop_and_disable();
    if (g_host != nullptr) {
        CUpti_Profiler_Host_Deinitialize_Params deinit{CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE};
        deinit.pHostObject = g_host;
        cuptiProfilerHostDeinitialize(&deinit);
        g_host = nullptr;
    }
    if (g_fifo >= 0) {
        ::close(g_fifo);
        g_fifo = -1;
    }
}

bool this_process_runs_the_model() {
    std::ifstream cmdline("/proc/self/cmdline");
    std::string text((std::istreambuf_iterator<char>(cmdline)), std::istreambuf_iterator<char>());
    // The TokenPowerBench parent imports CUDA, then vLLM runs the model in a child.
    // Sampling has to live in the child. A single-process application has neither name.
    return text.find("tokenpowerbench") == std::string::npos;
}

void start_sampling() {
    const char* path = std::getenv("FLYDVFS_SAMPLE_PATH");
    if (path == nullptr || *path == '\0') {
        return;
    }
    if (!this_process_runs_the_model()) {
        std::cerr << "[flydvfs] not sampling the launcher pid " << getpid() << std::endl;
        return;
    }
    try {
        if (cuInit(0) != CUDA_SUCCESS) {
            fail("cuInit failed");
            return;
        }
        g_fifo = ::open(path, O_WRONLY | O_CLOEXEC);
        if (g_fifo < 0) {
            fail(std::string("open sample fifo: ") + std::strerror(errno));
            return;
        }
        if (!setup_sampling(device_index())) {
            return;
        }
        g_decode = std::thread(decode_loop);
        std::atexit(shutdown_sampling);
    } catch (const std::exception& exc) {
        fail(exc.what());
        stop_and_disable();
    }
}

void ensure_sampling() {
    if (g_depth.load() > 0) {
        return;
    }
    g_depth.fetch_add(1);
    try {
        std::call_once(g_once, start_sampling);
    } catch (const std::exception& exc) {
        fail(exc.what());
    }
    g_depth.fetch_sub(1);
}

}  // namespace

void CUPTIAPI on_launch(void*, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void* data) {
    if (domain != CUPTI_CB_DOMAIN_DRIVER_API) {
        return;
    }
    if (cbid != CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel && cbid != CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx) {
        return;
    }
    const auto* info = static_cast<const CUpti_CallbackData*>(data);
    if (info->callbackSite == CUPTI_API_ENTER) {
        g_seen_launch.store(true);
    }
}

extern "C" {

int InitializeInjection(void) {
    if (std::getenv("FLYDVFS_SAMPLE_PATH") == nullptr || !this_process_runs_the_model()) {
        return 1;
    }
    CUpti_SubscriberHandle subscriber;
    CUptiResult status = cuptiSubscribe(&subscriber, on_launch, nullptr);
    if (status != CUPTI_SUCCESS) {
        fail(std::string("cuptiSubscribe: ") + cupti_error(status));
        return 0;
    }
    status = cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel);
    if (status != CUPTI_SUCCESS) {
        fail(std::string("cuptiEnableCallback: ") + cupti_error(status));
        return 0;
    }
    cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx);
    std::thread([] {
        while (!g_seen_launch.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        CUdevice device;
        CUcontext context = nullptr;
        if (cuInit(0) == CUDA_SUCCESS && cuDeviceGet(&device, device_index()) == CUDA_SUCCESS) {
            cuDevicePrimaryCtxRetain(&context, device);
            cuCtxSetCurrent(context);
        }
        ensure_sampling();
    }).detach();
    return 1;
}

}  // extern "C"
