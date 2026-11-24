/*
 * SPDX-License-Identifier: MIT
 *
 * AMD FidelityFX Super Resolution backend for the Detroit temporal integration.
 * See fsr_backend.hpp for the contract.
 *
 * Runtime: amd_fidelityfx_loader_vk.dll + amd_fidelityfx_upscaler_vk.dll
 * (AMD FidelityFX SDK signed binaries, MIT) next to the addon. The loader
 * picks the newest upscaler the GPU supports (FSR 3.1 analytic, or the ML
 * upscaler where AMD enables it).
 */

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif

#include <Windows.h>

#include <vulkan/vulkan.h>

#include <ffx_api.h>
#include <ffx_api_types.h>
#include <ffx_upscale.h>
#include <vk/ffx_api_vk.h>

#include <include/reshade.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "fsr_backend.hpp"

namespace renodx::games::detroitbecomehuman::dlss::fsr {
namespace {

using renodx::utils::dlss::vulkan::EvaluateInfo;
using renodx::utils::dlss::vulkan::FeatureConfig;
using renodx::utils::dlss::vulkan::ImageResource;
using renodx::utils::dlss::vulkan::ModeQuery;
using renodx::utils::dlss::vulkan::ModeSettings;

constexpr float kDefaultFrameTimeMs = 1000.f / 60.f;
constexpr float kMinimumFrameTimeMs = 1.f;
constexpr float kMaximumFrameTimeMs = 100.f;
// Detroit does not expose its projection to the capture path. These match the
// defaults OptiScaler uses for DLSS-input games; FSR only uses them for its
// depth-based disocclusion heuristics.
constexpr float kCameraNear = 0.1f;
constexpr float kCameraFar = 10000.f;
constexpr float kCameraFovVertical = 1.0471976f;  // 60 degrees in radians.

constexpr std::array<const wchar_t*, 2u> kLoaderNames = {
    L"amd_fidelityfx_loader_vk.dll",  // AMD FSR SDK 2.x
    L"amd_fidelityfx_vk.dll",         // FidelityFX SDK 1.1.x
};

constexpr std::uint32_t kParameterMagic = 0x46535250u;  // "FSRP"
constexpr std::uint32_t kFeatureMagic = 0x46535246u;    // "FSRF"

struct ParameterBlock final {
  std::uint32_t magic = kParameterMagic;
};

struct Feature final {
  std::uint32_t magic = kFeatureMagic;
  ffxContext context = nullptr;
  FeatureConfig config = {};
  std::uint32_t ffx_flags = 0u;
  bool needs_reset = true;
  bool first_dispatch_logged = false;
  std::uint32_t tuning_generation = 0u;
};

std::atomic<int> g_jitter_mode{3};
std::atomic<bool> g_jitter_cancellation{true};
std::atomic<std::uint32_t> g_tuning_generation{0u};

struct Backend final {
  std::mutex mutex;
  HMODULE module = nullptr;
  PfnFfxCreateContext create_context = nullptr;
  PfnFfxDestroyContext destroy_context = nullptr;
  PfnFfxDispatch dispatch = nullptr;
  PfnFfxQuery query = nullptr;
  VkDevice device = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  PFN_vkGetDeviceProcAddr get_device_proc_addr = nullptr;
  PFN_vkCmdPipelineBarrier cmd_pipeline_barrier = nullptr;
  std::wstring module_path;
  std::string default_version_name;
  LARGE_INTEGER frequency = {};
  LARGE_INTEGER last_dispatch = {};
  bool has_last_dispatch = false;
  bool load_attempted = false;
  std::uint32_t dispatch_failures_logged = 0u;
  bool faulted = false;
  // Tuning knobs read from ReShade.ini [RENODX_DETROIT_FSR] so sign mistakes
  // can be fixed without rebuilding.
  float jitter_sign_x = 1.f;
  float jitter_sign_y = 1.f;
  float motion_sign_x = 1.f;
  float motion_sign_y = 1.f;
  std::string status = "FSR: waiting for the Vulkan device";
};

// The FFX runtime runs inside the game process; if it faults, turn that into
// a clean "FSR unavailable" instead of taking Detroit down with it.
constexpr ffxReturnCode_t kSehFault = 0xFFFFFFFFu;

#ifdef _MSC_VER
ffxReturnCode_t GuardedCreate(
    PfnFfxCreateContext fn, ffxContext* context, ffxCreateContextDescHeader* desc, DWORD* fault) {
  __try {
    return fn(context, desc, nullptr);
  } __except (EXCEPTION_EXECUTE_HANDLER) {
    *fault = GetExceptionCode();
    return kSehFault;
  }
}
ffxReturnCode_t GuardedQuery(
    PfnFfxQuery fn, ffxContext* context, ffxQueryDescHeader* desc, DWORD* fault) {
  __try {
    return fn(context, desc);
  } __except (EXCEPTION_EXECUTE_HANDLER) {
    *fault = GetExceptionCode();
    return kSehFault;
  }
}
ffxReturnCode_t GuardedDispatch(
    PfnFfxDispatch fn, ffxContext* context, const ffxDispatchDescHeader* desc, DWORD* fault) {
  __try {
    return fn(context, desc);
  } __except (EXCEPTION_EXECUTE_HANDLER) {
    *fault = GetExceptionCode();
    return kSehFault;
  }
}
ffxReturnCode_t GuardedDestroy(PfnFfxDestroyContext fn, ffxContext* context, DWORD* fault) {
  __try {
    return fn(context, nullptr);
  } __except (EXCEPTION_EXECUTE_HANDLER) {
    *fault = GetExceptionCode();
    return kSehFault;
  }
}
#else
ffxReturnCode_t GuardedCreate(
    PfnFfxCreateContext fn, ffxContext* context, ffxCreateContextDescHeader* desc, DWORD*) {
  return fn(context, desc, nullptr);
}
ffxReturnCode_t GuardedQuery(PfnFfxQuery fn, ffxContext* context, ffxQueryDescHeader* desc, DWORD*) {
  return fn(context, desc);
}
ffxReturnCode_t GuardedDispatch(
    PfnFfxDispatch fn, ffxContext* context, const ffxDispatchDescHeader* desc, DWORD*) {
  return fn(context, desc);
}
ffxReturnCode_t GuardedDestroy(PfnFfxDestroyContext fn, ffxContext* context, DWORD*) {
  return fn(context, nullptr);
}
#endif

Backend& GetBackend() {
  static Backend backend;
  return backend;
}

void Log(reshade::log::level level, std::string_view message) noexcept {
  try {
    const std::string text = std::string("Detroit FSR: ") + std::string(message);
    reshade::log::message(level, text.c_str());
  } catch (...) {
    // Diagnostics must never change control flow.
  }
}

std::string Narrow(std::wstring_view text) {
  if (text.empty()) return {};
  const int size = WideCharToMultiByte(
      CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0, nullptr, nullptr);
  if (size <= 0) return {};
  std::string result(static_cast<std::size_t>(size), '\0');
  (void)WideCharToMultiByte(
      CP_UTF8, 0, text.data(), static_cast<int>(text.size()), result.data(), size, nullptr, nullptr);
  return result;
}

void OnFfxMessage(uint32_t type, const wchar_t* message) {
  if (message == nullptr) return;
  Log(type == FFX_API_MESSAGE_TYPE_ERROR ? reshade::log::level::error
                                          : reshade::log::level::warning,
      std::string("runtime: ") + Narrow(message));
}

std::filesystem::path GetThisModuleDirectory() {
  HMODULE self = nullptr;
  if (GetModuleHandleExW(
          GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
          reinterpret_cast<LPCWSTR>(&GetBackend),
          &self)
      == FALSE) {
    return {};
  }
  std::vector<wchar_t> buffer(1024u);
  for (;;) {
    const DWORD length =
        GetModuleFileNameW(self, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (length == 0u) return {};
    if (length < buffer.size() - 1u) {
      return std::filesystem::path(std::wstring(buffer.data(), length)).parent_path();
    }
    buffer.resize(buffer.size() * 2u);
  }
}

std::filesystem::path GetExecutableDirectory() {
  std::vector<wchar_t> buffer(1024u);
  for (;;) {
    const DWORD length =
        GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (length == 0u) return {};
    if (length < buffer.size() - 1u) {
      return std::filesystem::path(std::wstring(buffer.data(), length)).parent_path();
    }
    buffer.resize(buffer.size() * 2u);
  }
}

bool LoadRuntimeLocked(Backend* backend) {
  if (backend->module != nullptr) return true;
  if (backend->load_attempted) return false;
  backend->load_attempted = true;

  std::vector<std::filesystem::path> directories;
  if (auto directory = GetThisModuleDirectory(); !directory.empty()) {
    directories.push_back(std::move(directory));
  }
  if (auto directory = GetExecutableDirectory(); !directory.empty()
      && std::find(directories.begin(), directories.end(), directory) == directories.end()) {
    directories.push_back(std::move(directory));
  }

  for (const auto& directory : directories) {
    for (const auto* name : kLoaderNames) {
      const auto path = directory / name;
      std::error_code error;
      if (!std::filesystem::is_regular_file(path, error)) continue;
      // LOAD_WITH_ALTERED_SEARCH_PATH lets the loader find
      // amd_fidelityfx_upscaler_vk.dll in its own folder.
      HMODULE module = LoadLibraryExW(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
      if (module == nullptr) {
        Log(reshade::log::level::error,
            std::format("could not load {} (Windows error {})", Narrow(path.wstring()), GetLastError()));
        continue;
      }
      backend->create_context =
          reinterpret_cast<PfnFfxCreateContext>(GetProcAddress(module, "ffxCreateContext"));
      backend->destroy_context =
          reinterpret_cast<PfnFfxDestroyContext>(GetProcAddress(module, "ffxDestroyContext"));
      backend->dispatch =
          reinterpret_cast<PfnFfxDispatch>(GetProcAddress(module, "ffxDispatch"));
      backend->query = reinterpret_cast<PfnFfxQuery>(GetProcAddress(module, "ffxQuery"));
      if (backend->create_context == nullptr || backend->destroy_context == nullptr
          || backend->dispatch == nullptr || backend->query == nullptr) {
        Log(reshade::log::level::error,
            std::format("{} does not export the FidelityFX API", Narrow(path.wstring())));
        FreeLibrary(module);
        backend->create_context = nullptr;
        backend->destroy_context = nullptr;
        backend->dispatch = nullptr;
        backend->query = nullptr;
        continue;
      }
      backend->module = module;
      backend->module_path = path.wstring();
      Log(reshade::log::level::info, std::format("loaded {}", Narrow(path.wstring())));
      return true;
    }
  }

  backend->status = "FSR runtime missing: put amd_fidelityfx_loader_vk.dll and "
                    "amd_fidelityfx_upscaler_vk.dll next to DetroitBecomeHuman.exe";
  Log(reshade::log::level::error, backend->status);
  return false;
}

void LogAvailableVersionsLocked(Backend* backend) {
  VkDeviceContext device_context = {
      backend->device, backend->physical_device, backend->get_device_proc_addr};

  uint64_t count = 0u;
  ffxQueryDescGetVersions versions = {};
  versions.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
  versions.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
  versions.device = &device_context;
  versions.outputCount = &count;
  DWORD fault = 0u;
  if (GuardedQuery(backend->query, nullptr, &versions.header, &fault) != FFX_API_RETURN_OK
      || count == 0u) {
    if (fault != 0u) Log(reshade::log::level::error, std::format("version query faulted 0x{:08X}", fault));
    Log(reshade::log::level::warning, "could not enumerate FSR upscaler versions");
    return;
  }

  std::vector<uint64_t> ids(static_cast<std::size_t>(count), 0u);
  std::vector<const char*> names(static_cast<std::size_t>(count), nullptr);
  versions.versionIds = ids.data();
  versions.versionNames = names.data();
  if (GuardedQuery(backend->query, nullptr, &versions.header, &fault) != FFX_API_RETURN_OK) return;

  std::string list;
  for (uint64_t index = 0u; index < count; ++index) {
    if (!list.empty()) list += ", ";
    list += names[index] != nullptr ? names[index] : "?";
  }
  if (names[0] != nullptr) backend->default_version_name = names[0];
  Log(reshade::log::level::info, std::format("available upscaler versions: {}", list));
}

// The Detroit layer sits inside the Vulkan layer chain, where VkPhysicalDevice
// handles are the loader's unwrapped objects. The FFX runtime calls the
// vulkan-1.dll trampolines directly, and the loader aborts the process when it
// sees an unwrapped handle. Find the application-level handle for the same GPU.
VkPhysicalDevice ResolveApplicationPhysicalDevice(
    VkInstance instance,
    VkPhysicalDevice layer_physical_device,
    PFN_vkGetInstanceProcAddr layer_get_instance_proc_addr) {
  HMODULE loader = GetModuleHandleW(L"vulkan-1.dll");
  if (loader == nullptr || instance == VK_NULL_HANDLE) return VK_NULL_HANDLE;
  const auto enumerate = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(
      GetProcAddress(loader, "vkEnumeratePhysicalDevices"));
  const auto app_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
      GetProcAddress(loader, "vkGetPhysicalDeviceProperties"));
  if (enumerate == nullptr || app_properties == nullptr) return VK_NULL_HANDLE;

  VkPhysicalDeviceProperties wanted = {};
  bool have_wanted = false;
  if (layer_get_instance_proc_addr != nullptr) {
    const auto layer_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
        layer_get_instance_proc_addr(instance, "vkGetPhysicalDeviceProperties"));
    if (layer_properties != nullptr) {
      layer_properties(layer_physical_device, &wanted);
      have_wanted = true;
    }
  }

  uint32_t count = 0u;
  if (enumerate(instance, &count, nullptr) != VK_SUCCESS || count == 0u) return VK_NULL_HANDLE;
  std::vector<VkPhysicalDevice> devices(count, VK_NULL_HANDLE);
  if (enumerate(instance, &count, devices.data()) < VK_SUCCESS) return VK_NULL_HANDLE;
  devices.resize(count);

  VkPhysicalDevice match = VK_NULL_HANDLE;
  for (auto candidate : devices) {
    if (candidate == layer_physical_device) return candidate;
    VkPhysicalDeviceProperties properties = {};
    app_properties(candidate, &properties);
    Log(reshade::log::level::info,
        std::format(
            "GPU candidate {} vendor=0x{:X} device=0x{:X}",
            properties.deviceName, properties.vendorID, properties.deviceID));
    if (have_wanted && properties.vendorID == wanted.vendorID
        && properties.deviceID == wanted.deviceID
        && std::memcmp(properties.pipelineCacheUUID, wanted.pipelineCacheUUID, VK_UUID_SIZE) == 0
        && match == VK_NULL_HANDLE) {
      match = candidate;
    }
  }
  if (match == VK_NULL_HANDLE && devices.size() == 1u) match = devices[0];
  return match;
}

std::uint32_t ToFfxCreateFlags(std::uint32_t ngx_flags) {
  std::uint32_t flags = 0u;
  if ((ngx_flags & NVSDK_NGX_DLSS_Feature_Flags_IsHDR) != 0u) {
    flags |= FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE;
  }
  if ((ngx_flags & NVSDK_NGX_DLSS_Feature_Flags_MVLowRes) == 0u) {
    flags |= FFX_UPSCALE_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS;
  }
  if ((ngx_flags & NVSDK_NGX_DLSS_Feature_Flags_MVJittered) != 0u
      && g_jitter_cancellation.load(std::memory_order_relaxed)) {
    flags |= FFX_UPSCALE_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION;
  }
  if ((ngx_flags & NVSDK_NGX_DLSS_Feature_Flags_DepthInverted) != 0u) {
    flags |= FFX_UPSCALE_ENABLE_DEPTH_INVERTED;
  }
  if ((ngx_flags & NVSDK_NGX_DLSS_Feature_Flags_AutoExposure) != 0u) {
    flags |= FFX_UPSCALE_ENABLE_AUTO_EXPOSURE;
  }
  return flags;
}

// Per-dimension upscale ratio for each NGX quality value (matches FSR's modes;
// DLAA is native resolution).
float RatioForMode(NVSDK_NGX_PerfQuality_Value mode) {
  switch (mode) {
    case NVSDK_NGX_PerfQuality_Value_MaxPerf:          return 2.0f;
    case NVSDK_NGX_PerfQuality_Value_Balanced:         return 1.7f;
    case NVSDK_NGX_PerfQuality_Value_MaxQuality:       return 1.5f;
    case NVSDK_NGX_PerfQuality_Value_UltraPerformance: return 3.0f;
    case NVSDK_NGX_PerfQuality_Value_UltraQuality:     return 1.3f;
    case NVSDK_NGX_PerfQuality_Value_DLAA:
    default:                                           return 1.0f;
  }
}

bool IsDepthStencilFormat(VkFormat format) {
  return format == VK_FORMAT_D16_UNORM_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT
         || format == VK_FORMAT_D32_SFLOAT_S8_UINT;
}

bool IsReadableByFfx(VkImageLayout layout) {
  return layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
         || layout == VK_IMAGE_LAYOUT_GENERAL;
}

struct LayoutFix final {
  VkImage image = VK_NULL_HANDLE;
  VkImageSubresourceRange range = {};
  VkImageLayout original = VK_IMAGE_LAYOUT_UNDEFINED;
  VkImageLayout temporary = VK_IMAGE_LAYOUT_UNDEFINED;
};

// Returns the FFX resource state for the image's current layout. Layouts FFX
// cannot express (depth read-only, attachment layouts, ...) are moved to a
// layout it understands for the duration of the dispatch and restored after.
std::uint32_t ResolveState(
    const ImageResource& image, bool output, std::vector<LayoutFix>* fixes) {
  const auto& view = image.ngx.Resource.ImageViewInfo;
  const VkImageLayout layout = image.state.layout;
  if (output) {
    if (layout != VK_IMAGE_LAYOUT_GENERAL) {
      fixes->push_back({view.Image, view.SubresourceRange, layout, VK_IMAGE_LAYOUT_GENERAL});
    }
    return FFX_API_RESOURCE_STATE_UNORDERED_ACCESS;
  }
  if (layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
    return FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ;
  }
  if (layout == VK_IMAGE_LAYOUT_GENERAL) {
    return FFX_API_RESOURCE_STATE_UNORDERED_ACCESS;
  }
  VkImageSubresourceRange range = view.SubresourceRange;
  if (IsDepthStencilFormat(view.Format)) {
    // Without separateDepthStencilLayouts both aspects must transition together.
    range.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
  }
  fixes->push_back({view.Image, range, layout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
  return FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ;
}

FfxApiResource MakeResource(const ImageResource& image, std::uint32_t state, bool output) {
  const auto& view = image.ngx.Resource.ImageViewInfo;
  FfxApiResourceDescription description = {};
  description.type = FFX_API_RESOURCE_TYPE_TEXTURE2D;
  description.format = ffxApiGetSurfaceFormatVK(view.Format);
  description.width = image.width != 0u ? image.width : view.Width;
  description.height = image.height != 0u ? image.height : view.Height;
  description.depth = 1u;
  description.mipCount = 1u;
  description.flags = FFX_API_RESOURCE_FLAGS_NONE;
  description.usage = FFX_API_RESOURCE_USAGE_READ_ONLY;
  if (ffxApiIsDepthFormat(view.Format)) description.usage |= FFX_API_RESOURCE_USAGE_DEPTHTARGET;
  if (ffxApiIsStencilFormat(view.Format)) description.usage |= FFX_API_RESOURCE_USAGE_STENCILTARGET;
  if (output || (image.state.usage & VK_IMAGE_USAGE_STORAGE_BIT) != 0u) {
    description.usage |= FFX_API_RESOURCE_USAGE_UAV;
  }
  return ffxApiGetResourceVK(reinterpret_cast<void*>(view.Image), description, state);
}

void RecordLayoutFixes(
    const Backend& backend,
    VkCommandBuffer command_buffer,
    const std::vector<LayoutFix>& fixes,
    bool restore) {
  if (fixes.empty()) return;
  std::vector<VkImageMemoryBarrier> barriers;
  barriers.reserve(fixes.size());
  for (const auto& fix : fixes) {
    barriers.push_back({
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        .oldLayout = restore ? fix.temporary : fix.original,
        .newLayout = restore ? fix.original : fix.temporary,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = fix.image,
        .subresourceRange = fix.range,
    });
  }
  backend.cmd_pipeline_barrier(
      command_buffer,
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      0u,
      0u,
      nullptr,
      0u,
      nullptr,
      static_cast<std::uint32_t>(barriers.size()),
      barriers.data());
}

float NextFrameTimeLocked(Backend* backend) {
  LARGE_INTEGER now = {};
  if (backend->frequency.QuadPart == 0 || QueryPerformanceCounter(&now) == FALSE) {
    return kDefaultFrameTimeMs;
  }
  float milliseconds = kDefaultFrameTimeMs;
  if (backend->has_last_dispatch) {
    milliseconds = static_cast<float>(
        static_cast<double>(now.QuadPart - backend->last_dispatch.QuadPart) * 1000.0
        / static_cast<double>(backend->frequency.QuadPart));
    if (!std::isfinite(milliseconds) || milliseconds < kMinimumFrameTimeMs
        || milliseconds > kMaximumFrameTimeMs) {
      milliseconds = kDefaultFrameTimeMs;
    }
  }
  backend->last_dispatch = now;
  backend->has_last_dispatch = true;
  return milliseconds;
}

Feature* ToFeature(NVSDK_NGX_Handle* handle) {
  auto* feature = reinterpret_cast<Feature*>(handle);
  return feature != nullptr && feature->magic == kFeatureMagic ? feature : nullptr;
}

}  // namespace

NVSDK_NGX_Result InitializeProject(
    void*,
    const char*,
    NVSDK_NGX_EngineType,
    const char*,
    const wchar_t*,
    VkInstance instance,
    VkPhysicalDevice physical_device,
    VkDevice device,
    PFN_vkGetInstanceProcAddr get_instance_proc_addr,
    PFN_vkGetDeviceProcAddr get_device_proc_addr,
    const NVSDK_NGX_FeatureCommonInfo*,
    NVSDK_NGX_Version) {
  auto& backend = GetBackend();
  const std::lock_guard lock(backend.mutex);
  if (device == VK_NULL_HANDLE || physical_device == VK_NULL_HANDLE
      || get_device_proc_addr == nullptr) {
    return NVSDK_NGX_Result_FAIL_InvalidParameter;
  }
  Log(reshade::log::level::info, "initializing");
  if (!LoadRuntimeLocked(&backend)) return NVSDK_NGX_Result_FAIL_FeatureNotSupported;

  const VkPhysicalDevice app_physical_device =
      ResolveApplicationPhysicalDevice(instance, physical_device, get_instance_proc_addr);
  if (app_physical_device == VK_NULL_HANDLE) {
    backend.status = "FSR: could not identify the GPU for the FSR runtime";
    Log(reshade::log::level::error, backend.status);
    return NVSDK_NGX_Result_FAIL_PlatformError;
  }
  Log(reshade::log::level::info,
      std::format(
          "physical device layer=0x{:X} application=0x{:X}",
          reinterpret_cast<std::uintptr_t>(physical_device),
          reinterpret_cast<std::uintptr_t>(app_physical_device)));
  backend.device = device;
  backend.physical_device = app_physical_device;
  backend.get_device_proc_addr = get_device_proc_addr;
  backend.cmd_pipeline_barrier = reinterpret_cast<PFN_vkCmdPipelineBarrier>(
      get_device_proc_addr(device, "vkCmdPipelineBarrier"));
  if (backend.cmd_pipeline_barrier == nullptr) {
    backend.status = "FSR: vkCmdPipelineBarrier unavailable";
    Log(reshade::log::level::error, backend.status);
    return NVSDK_NGX_Result_FAIL_PlatformError;
  }
  (void)QueryPerformanceFrequency(&backend.frequency);
  {
    const auto read_sign = [](const char* key, float* value) {
      float configured = 1.f;
      if (reshade::get_config_value(nullptr, "RENODX_DETROIT_FSR", key, configured)) {
        *value = configured < 0.f ? -1.f : 1.f;
      } else {
        reshade::set_config_value(nullptr, "RENODX_DETROIT_FSR", key, 1);
      }
    };
    read_sign("JitterSignX", &backend.jitter_sign_x);
    read_sign("JitterSignY", &backend.jitter_sign_y);
    read_sign("MotionVectorSignX", &backend.motion_sign_x);
    read_sign("MotionVectorSignY", &backend.motion_sign_y);
    Log(reshade::log::level::info,
        std::format(
            "signs jitter=({},{}) motion=({},{})",
            backend.jitter_sign_x,
            backend.jitter_sign_y,
            backend.motion_sign_x,
            backend.motion_sign_y));
  }
  backend.has_last_dispatch = false;
  {
    bool list_versions = false;
    (void)reshade::get_config_value(nullptr, "RENODX_DETROIT_FSR", "ListVersions", list_versions);
    if (list_versions) {
      Log(reshade::log::level::info, "querying upscaler versions");
      LogAvailableVersionsLocked(&backend);
    }
  }
  backend.status = backend.default_version_name.empty()
                       ? std::string("FSR ready")
                       : std::format("FSR ready ({})", backend.default_version_name);
  Log(reshade::log::level::info, backend.status);
  return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result GetCapabilityParameters(void*, NVSDK_NGX_Parameter** parameters) {
  if (parameters == nullptr) return NVSDK_NGX_Result_FAIL_InvalidParameter;
  *parameters = reinterpret_cast<NVSDK_NGX_Parameter*>(new ParameterBlock());
  return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result GetParameterI(void*, NVSDK_NGX_Parameter*, const char* name, int* value) {
  if (name == nullptr || value == nullptr) return NVSDK_NGX_Result_FAIL_InvalidParameter;
  auto& backend = GetBackend();
  const std::lock_guard lock(backend.mutex);
  *value = 0;
  if (std::strcmp(name, NVSDK_NGX_Parameter_SuperSampling_Available) == 0) {
    *value = backend.module != nullptr ? 1 : 0;
  }
  return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result QueryMode(
    void*,
    NVSDK_NGX_Parameter*,
    const ModeQuery* query,
    ModeSettings* settings) {
  if (query == nullptr || settings == nullptr || query->output_width == 0u
      || query->output_height == 0u) {
    return NVSDK_NGX_Result_FAIL_InvalidParameter;
  }
  const float ratio = RatioForMode(query->mode);
  const auto scaled = [ratio](std::uint32_t size) {
    return std::max(1u, static_cast<std::uint32_t>(std::lround(static_cast<double>(size) / ratio)));
  };
  settings->optimal_width = scaled(query->output_width);
  settings->optimal_height = scaled(query->output_height);
  settings->minimum_width = std::max(1u, query->output_width / 3u);
  settings->minimum_height = std::max(1u, query->output_height / 3u);
  settings->maximum_width = query->output_width;
  settings->maximum_height = query->output_height;
  settings->sharpness = 0.f;
  return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result AllocateParameters(void*, NVSDK_NGX_Parameter** parameters) {
  if (parameters == nullptr) return NVSDK_NGX_Result_FAIL_InvalidParameter;
  *parameters = reinterpret_cast<NVSDK_NGX_Parameter*>(new ParameterBlock());
  return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result DestroyParameters(void*, NVSDK_NGX_Parameter* parameters) {
  auto* block = reinterpret_cast<ParameterBlock*>(parameters);
  if (block != nullptr && block->magic == kParameterMagic) {
    block->magic = 0u;
    delete block;
  }
  return NVSDK_NGX_Result_Success;
}

// Picks the newest FSR 3.x upscaler version. The FidelityFX loader otherwise
// selects the newest provider (e.g. 4.x supplied by the AMD driver), which can
// be far slower on GPUs without FSR4 hardware. Set PreferNewestVersion=1 in
// [RENODX_DETROIT_FSR] to keep the loader default.
bool PickPreferredVersionLocked(Backend* backend, uint64_t* id, std::string* name) {
  bool prefer_newest = false;
  (void)reshade::get_config_value(nullptr, "RENODX_DETROIT_FSR", "PreferNewestVersion", prefer_newest);
  if (prefer_newest) return false;

  VkDeviceContext device_context = {
      backend->device, backend->physical_device, backend->get_device_proc_addr};
  uint64_t count = 0u;
  ffxQueryDescGetVersions versions = {};
  versions.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
  versions.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
  versions.device = &device_context;
  versions.outputCount = &count;
  DWORD fault = 0u;
  if (GuardedQuery(backend->query, nullptr, &versions.header, &fault) != FFX_API_RETURN_OK
      || count == 0u) {
    return false;
  }
  std::vector<uint64_t> ids(static_cast<std::size_t>(count), 0u);
  std::vector<const char*> names(static_cast<std::size_t>(count), nullptr);
  versions.versionIds = ids.data();
  versions.versionNames = names.data();
  if (GuardedQuery(backend->query, nullptr, &versions.header, &fault) != FFX_API_RETURN_OK) return false;

  // FSRVersion in [renodx-preset1]: 0 = Auto (newest 3.x), 1 = FSR 4.x.
  float version_choice = 0.f;
  (void)reshade::get_config_value(nullptr, "renodx-preset1", "FSRVersion", version_choice);
  const char* wanted_prefix = version_choice > 0.5f ? "4." : "3.";

  std::string list;
  int best = -1;
  int best_auto = -1;
  for (uint64_t i = 0u; i < count; ++i) {
    const std::string n = names[i] != nullptr ? names[i] : "?";
    if (!list.empty()) list += ", ";
    list += n;
    if (n.rfind(wanted_prefix, 0) == 0 && (best < 0 || n > std::string(names[best]))) best = static_cast<int>(i);
    if (n.rfind("3.", 0) == 0 && (best_auto < 0 || n > std::string(names[best_auto]))) best_auto = static_cast<int>(i);
  }
  Log(reshade::log::level::info, std::format("available upscaler versions: {}", list));
  if (best < 0 && best_auto >= 0) {
    Log(reshade::log::level::warning,
        std::format("the selected FSR version ({}x) is not available on this GPU; using the newest 3.x instead.", wanted_prefix));
    best = best_auto;
  }
  if (best < 0) return false;
  *id = ids[static_cast<std::size_t>(best)];
  *name = names[static_cast<std::size_t>(best)];
  return true;
}

NVSDK_NGX_Result CreateFeature(
    void*,
    VkDevice,
    VkCommandBuffer,
    NVSDK_NGX_Handle** handle,
    NVSDK_NGX_Parameter*,
    const FeatureConfig* config) {
  if (handle == nullptr || config == nullptr || config->render_width == 0u
      || config->render_height == 0u || config->output_width == 0u
      || config->output_height == 0u) {
    return NVSDK_NGX_Result_FAIL_InvalidParameter;
  }
  *handle = nullptr;
  auto& backend = GetBackend();
  const std::lock_guard lock(backend.mutex);
  if (backend.module == nullptr || backend.device == VK_NULL_HANDLE || backend.faulted) {
    return NVSDK_NGX_Result_FAIL_NotInitialized;
  }

  auto feature = std::make_unique<Feature>();
  feature->config = *config;
  feature->ffx_flags = ToFfxCreateFlags(config->create_flags);

  ffxCreateBackendVKDesc backend_description = {};
  backend_description.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
  backend_description.deviceContext.vkDevice = backend.device;
  backend_description.deviceContext.vkPhysicalDevice = backend.physical_device;
  backend_description.deviceContext.vkDeviceProcAddr = backend.get_device_proc_addr;

  ffxCreateContextDescUpscale description = {};
  description.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
  description.header.pNext = &backend_description.header;
  ffxOverrideVersion version_override = {};
  uint64_t preferred_id = 0u;
  std::string preferred_name;
  if (PickPreferredVersionLocked(&backend, &preferred_id, &preferred_name)) {
    version_override.header.type = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
    version_override.header.pNext = &backend_description.header;
    version_override.versionId = preferred_id;
    description.header.pNext = &version_override.header;
    Log(reshade::log::level::info, std::format("pinning FSR upscaler version {}", preferred_name));
  }
  description.flags = feature->ffx_flags;
  description.maxRenderSize = {config->render_width, config->render_height};
  description.maxUpscaleSize = {config->output_width, config->output_height};
  description.fpMessage = &OnFfxMessage;

  Log(reshade::log::level::info,
      std::format(
          "creating context render={}x{} output={}x{} flags=0x{:X}",
          config->render_width, config->render_height,
          config->output_width, config->output_height, feature->ffx_flags));
  DWORD fault = 0u;
  const ffxReturnCode_t result =
      GuardedCreate(backend.create_context, &feature->context, &description.header, &fault);
  if (fault != 0u) {
    backend.faulted = true;
    backend.status = std::format("FSR crashed while creating its context (0x{:08X}); using native TAA", fault);
    Log(reshade::log::level::error, backend.status);
    return NVSDK_NGX_Result_FAIL_UnableToInitializeFeature;
  }
  if (result != FFX_API_RETURN_OK || feature->context == nullptr) {
    backend.status = std::format("FSR context creation failed (code {})", result);
    Log(reshade::log::level::error, backend.status);
    return NVSDK_NGX_Result_FAIL_UnableToInitializeFeature;
  }

  ffxQueryGetProviderVersion provider = {};
  provider.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
  std::string provider_name = "FSR";
  if (GuardedQuery(backend.query, &feature->context, &provider.header, &fault) == FFX_API_RETURN_OK
      && provider.versionName != nullptr) {
    provider_name = provider.versionName;
  }
  backend.status = std::format("FSR active ({})", provider_name);
  Log(reshade::log::level::info,
      std::format(
          "created upscaler {} render={}x{} output={}x{} ffx_flags=0x{:X}",
          provider_name,
          config->render_width,
          config->render_height,
          config->output_width,
          config->output_height,
          feature->ffx_flags));

  *handle = reinterpret_cast<NVSDK_NGX_Handle*>(feature.release());
  return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result EvaluateFeature(
    void*,
    VkCommandBuffer command_buffer,
    NVSDK_NGX_Handle* handle,
    NVSDK_NGX_Parameter*,
    const EvaluateInfo* info) {
  Feature* feature = ToFeature(handle);
  if (feature == nullptr || info == nullptr || command_buffer == VK_NULL_HANDLE
      || info->color == nullptr || info->depth == nullptr
      || info->motion_vectors == nullptr || info->output == nullptr) {
    return NVSDK_NGX_Result_FAIL_InvalidParameter;
  }
  auto& backend = GetBackend();
  const std::lock_guard lock(backend.mutex);
  if (backend.dispatch == nullptr || feature->context == nullptr || backend.faulted) {
    return NVSDK_NGX_Result_FAIL_NotInitialized;
  }

  std::vector<LayoutFix> fixes;
  const auto color_state = ResolveState(*info->color, false, &fixes);
  const auto depth_state = ResolveState(*info->depth, false, &fixes);
  const auto motion_state = ResolveState(*info->motion_vectors, false, &fixes);
  const auto output_state = ResolveState(*info->output, true, &fixes);
  std::uint32_t exposure_state = 0u;
  if (info->exposure != nullptr) exposure_state = ResolveState(*info->exposure, false, &fixes);

  ffxDispatchDescUpscale dispatch = {};
  dispatch.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
  dispatch.commandList = command_buffer;
  dispatch.color = MakeResource(*info->color, color_state, false);
  dispatch.depth = MakeResource(*info->depth, depth_state, false);
  dispatch.motionVectors = MakeResource(*info->motion_vectors, motion_state, false);
  if (info->exposure != nullptr) {
    dispatch.exposure = MakeResource(*info->exposure, exposure_state, false);
  }
  dispatch.output = MakeResource(*info->output, output_state, true);
  const int jitter_mode = g_jitter_mode.load(std::memory_order_relaxed);
  const float flip_x = (jitter_mode == 1 || jitter_mode == 3) ? -1.f : 1.f;
  const float flip_y = (jitter_mode == 2 || jitter_mode == 3) ? -1.f : 1.f;
  const std::uint32_t generation = g_tuning_generation.load(std::memory_order_relaxed);
  if (generation != feature->tuning_generation) {
    feature->tuning_generation = generation;
    feature->needs_reset = true;
    Log(reshade::log::level::info, std::format("jitter mode {}", jitter_mode));
  }
  dispatch.jitterOffset = {
      info->jitter_x * backend.jitter_sign_x * flip_x,
      info->jitter_y * backend.jitter_sign_y * flip_y};
  dispatch.motionVectorScale = {
      info->motion_vector_scale_x * backend.motion_sign_x,
      info->motion_vector_scale_y * backend.motion_sign_y};
  dispatch.renderSize = {info->render_width, info->render_height};
  dispatch.upscaleSize = {feature->config.output_width, feature->config.output_height};
  dispatch.enableSharpening = false;
  dispatch.sharpness = 0.f;
  dispatch.frameTimeDelta = NextFrameTimeLocked(&backend);
  dispatch.preExposure = info->pre_exposure > 0.f && std::isfinite(info->pre_exposure)
                             ? info->pre_exposure
                             : 1.f;
  dispatch.reset = info->reset || feature->needs_reset;
  dispatch.cameraNear = kCameraNear;
  dispatch.cameraFar = kCameraFar;
  dispatch.cameraFovAngleVertical = kCameraFovVertical;
  dispatch.viewSpaceToMetersFactor = 1.f;
  dispatch.flags = 0u;

  if (!feature->first_dispatch_logged) {
    feature->first_dispatch_logged = true;
    const auto layout_name = [](VkImageLayout layout) { return static_cast<int>(layout); };
    Log(reshade::log::level::info,
        std::format(
            "first dispatch render={}x{} jitter=({:.3f},{:.3f}) mv_scale=({:.1f},{:.1f}) "
            "layouts color={} depth={} mv={} output={} depth_format={} layout_fixes={}",
            info->render_width,
            info->render_height,
            info->jitter_x,
            info->jitter_y,
            info->motion_vector_scale_x,
            info->motion_vector_scale_y,
            layout_name(info->color->state.layout),
            layout_name(info->depth->state.layout),
            layout_name(info->motion_vectors->state.layout),
            layout_name(info->output->state.layout),
            static_cast<int>(info->depth->ngx.Resource.ImageViewInfo.Format),
            fixes.size()));
  }

  RecordLayoutFixes(backend, command_buffer, fixes, false);
  DWORD fault = 0u;
  const ffxReturnCode_t result =
      GuardedDispatch(backend.dispatch, &feature->context, &dispatch.header, &fault);
  if (fault != 0u) {
    backend.faulted = true;
    backend.status = std::format("FSR crashed while upscaling (0x{:08X}); using native TAA", fault);
    Log(reshade::log::level::error, backend.status);
  }
  RecordLayoutFixes(backend, command_buffer, fixes, true);

  if (result != FFX_API_RETURN_OK) {
    if (backend.dispatch_failures_logged < 5u) {
      ++backend.dispatch_failures_logged;
      Log(reshade::log::level::error, std::format("dispatch failed (code {})", result));
    }
    feature->needs_reset = true;
    return NVSDK_NGX_Result_Fail;
  }
  feature->needs_reset = false;
  return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result ReleaseFeature(void*, NVSDK_NGX_Handle* handle) {
  Feature* feature = ToFeature(handle);
  if (feature == nullptr) return NVSDK_NGX_Result_FAIL_InvalidParameter;
  auto& backend = GetBackend();
  {
    const std::lock_guard lock(backend.mutex);
    if (feature->context != nullptr && backend.destroy_context != nullptr) {
      DWORD fault = 0u;
      (void)GuardedDestroy(backend.destroy_context, &feature->context, &fault);
    }
    feature->context = nullptr;
  }
  feature->magic = 0u;
  delete feature;
  return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result Shutdown(void*, VkDevice) {
  auto& backend = GetBackend();
  const std::lock_guard lock(backend.mutex);
  // Keep the FFX module loaded: unloading while the driver may still reference
  // its pipelines is not worth the risk for a once-per-process shutdown.
  backend.device = VK_NULL_HANDLE;
  backend.physical_device = VK_NULL_HANDLE;
  backend.get_device_proc_addr = nullptr;
  backend.cmd_pipeline_barrier = nullptr;
  backend.has_last_dispatch = false;
  backend.status = "FSR: device shut down";
  return NVSDK_NGX_Result_Success;
}

namespace {
std::atomic<bool> g_backend_active{true};
}  // namespace

void SetActive(bool active) { g_backend_active.store(active, std::memory_order_release); }
bool IsActive() { return g_backend_active.load(std::memory_order_acquire); }

void SetTuning(int jitter_mode, bool jitter_cancellation) {
  g_jitter_mode.store(std::clamp(jitter_mode, 0, 3), std::memory_order_relaxed);
  g_jitter_cancellation.store(jitter_cancellation, std::memory_order_relaxed);
  g_tuning_generation.fetch_add(1u, std::memory_order_relaxed);
}

std::string GetStatusText() {
  auto& backend = GetBackend();
  const std::lock_guard lock(backend.mutex);
  return backend.status;
}

}  // namespace renodx::games::detroitbecomehuman::dlss::fsr
