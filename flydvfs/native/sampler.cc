// In-process CUPTI PM sampler. Loaded with LD_PRELOAD into the application.
// Sampling starts on the first kernel launch, which is after that process has
// a CUDA context. One pass collects SM active cycles, SM elapsed cycles, and
// DRAM bytes. Each completed window is one JSON line on FLYDVFS_SAMPLE_PATH.

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
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <cstdint>
#include <cstring>
#include <iostream>
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

const char* kMetricNames[] = {
    "sm__cycles_active.sum",
    "sm__cycles_elapsed.sum",
    "dram__bytes.sum",
};

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

uint64_t interval_cycles() {
    const char* env = std::getenv("FLYDVFS_INTERVAL_CYCLES");
    if (env == nullptr || *env == '\0') {
        return 100000000ULL;
    }
    return std::strtoull(env, nullptr, 10);
}

int device_index() {
    const char* env = std::getenv("FLYDVFS_GPU");
    if (env == nullptr || *env == '\0') {
        return 0;
    }
    return std::atoi(env);
}

void write_sample(const Metrics& sample) {
    if (g_fifo < 0) {
        return;
    }
    char line[256];
    int n = std::snprintf(
        line, sizeof(line),
        "{\"sm_active\":%.0f,\"sm_elapsed\":%.0f,\"dram_bytes\":%.0f,\"start_ns\":%llu,\"end_ns\":%llu}\n",
        sample.sm_active, sample.sm_elapsed, sample.dram_bytes,
        static_cast<unsigned long long>(sample.start_ns),
        static_cast<unsigned long long>(sample.end_ns));
    if (n <= 0) {
        return;
    }
    const char* cursor = line;
    int left = n;
    while (left > 0) {
        ssize_t wrote = ::write(g_fifo, cursor, static_cast<size_t>(left));
        if (wrote < 0) {
            if (errno == EINTR) {
                continue;
            }
            return;
        }
        cursor += wrote;
        left -= static_cast<int>(wrote);
    }
}

bool setup_sampling(int device) {
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

    CUpti_PmSampling_Enable_Params enable{CUpti_PmSampling_Enable_Params_STRUCT_SIZE};
    enable.deviceIndex = static_cast<size_t>(device);
    check(cuptiPmSamplingEnable(&enable), "cuptiPmSamplingEnable");
    g_sampler = enable.pPmSamplingObject;

    CUpti_PmSampling_SetConfig_Params set_config{CUpti_PmSampling_SetConfig_Params_STRUCT_SIZE};
    set_config.pPmSamplingObject = g_sampler;
    set_config.configSize = g_config.size();
    set_config.pConfig = g_config.data();
    set_config.hardwareBufferSize = 64 * 1024 * 1024;
    set_config.samplingInterval = interval_cycles();
    set_config.triggerMode = CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_SYSCLK_INTERVAL;
    set_config.hwBufferAppendMode = CUPTI_PM_SAMPLING_HARDWARE_BUFFER_APPEND_MODE_KEEP_OLDEST;
    check(cuptiPmSamplingSetConfig(&set_config), "cuptiPmSamplingSetConfig");

    CUpti_PmSampling_GetCounterDataSize_Params data_size{CUpti_PmSampling_GetCounterDataSize_Params_STRUCT_SIZE};
    data_size.pPmSamplingObject = g_sampler;
    data_size.numMetrics = 3;
    data_size.pMetricNames = kMetricNames;
    data_size.maxSamples = 128;
    check(cuptiPmSamplingGetCounterDataSize(&data_size), "cuptiPmSamplingGetCounterDataSize");
    g_counter_image.resize(data_size.counterDataSize);
    CUpti_PmSampling_CounterDataImage_Initialize_Params init_image{CUpti_PmSampling_CounterDataImage_Initialize_Params_STRUCT_SIZE};
    init_image.pPmSamplingObject = g_sampler;
    init_image.counterDataSize = g_counter_image.size();
    init_image.pCounterData = g_counter_image.data();
    check(cuptiPmSamplingCounterDataImageInitialize(&init_image), "cuptiPmSamplingCounterDataImageInitialize");

    CUpti_PmSampling_Start_Params start{CUpti_PmSampling_Start_Params_STRUCT_SIZE};
    start.pPmSamplingObject = g_sampler;
    check(cuptiPmSamplingStart(&start), "cuptiPmSamplingStart");
    std::cerr << "[flydvfs] sampling pid " << getpid() << " chip " << g_chip
              << " interval_cycles " << interval_cycles() << std::endl;
    return true;
}

void reset_counter_image() {
    CUpti_PmSampling_CounterDataImage_Initialize_Params init_image{CUpti_PmSampling_CounterDataImage_Initialize_Params_STRUCT_SIZE};
    init_image.pPmSamplingObject = g_sampler;
    init_image.counterDataSize = g_counter_image.size();
    init_image.pCounterData = g_counter_image.data();
    check(cuptiPmSamplingCounterDataImageInitialize(&init_image), "reset counter image");
}

void decode_loop() {
    CUcontext context = nullptr;
    CUdevice device;
    if (cuDeviceGet(&device, device_index()) == CUDA_SUCCESS) {
        cuDevicePrimaryCtxRetain(&context, device);
        cuCtxSetCurrent(context);
    }
    const char* names[] = {kMetricNames[0], kMetricNames[1], kMetricNames[2]};
    bool reported_unknown = false;
    while (!g_stop.load()) {
        try {
            CUpti_PmSampling_DecodeData_Params decode{CUpti_PmSampling_DecodeData_Params_STRUCT_SIZE};
            decode.pPmSamplingObject = g_sampler;
            decode.pCounterDataImage = g_counter_image.data();
            decode.counterDataImageSize = g_counter_image.size();
            CUptiResult decoded = cuptiPmSamplingDecodeData(&decode);
            if (decoded == CUPTI_ERROR_UNKNOWN) {
                if (!reported_unknown) {
                    fail("decode not ready yet");
                    reported_unknown = true;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                continue;
            }
            check(decoded, "cuptiPmSamplingDecodeData");

            CUpti_PmSampling_GetCounterDataInfo_Params info{CUpti_PmSampling_GetCounterDataInfo_Params_STRUCT_SIZE};
            info.pCounterDataImage = g_counter_image.data();
            info.counterDataImageSize = g_counter_image.size();
            check(cuptiPmSamplingGetCounterDataInfo(&info), "cuptiPmSamplingGetCounterDataInfo");

            for (size_t index = 0; index < info.numCompletedSamples; ++index) {
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
                write_sample(sample);
            }
            if (info.numCompletedSamples > 0) {
                reset_counter_image();
            }
        } catch (const std::exception& exc) {
            fail(exc.what());
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

void shutdown_sampling() {
    g_stop.store(true);
    if (g_decode.joinable()) {
        g_decode.join();
    }
    if (g_sampler != nullptr) {
        CUpti_PmSampling_Stop_Params stop{CUpti_PmSampling_Stop_Params_STRUCT_SIZE};
        stop.pPmSamplingObject = g_sampler;
        cuptiPmSamplingStop(&stop);
        CUpti_PmSampling_Disable_Params disable{CUpti_PmSampling_Disable_Params_STRUCT_SIZE};
        disable.pPmSamplingObject = g_sampler;
        cuptiPmSamplingDisable(&disable);
        g_sampler = nullptr;
    }
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
