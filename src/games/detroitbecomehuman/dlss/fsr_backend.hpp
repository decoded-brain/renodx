/*
 * SPDX-License-Identifier: MIT
 *
 * AMD FidelityFX Super Resolution backend for the Detroit temporal integration.
 *
 * The Detroit layer drives its temporal upscaler through
 * renodx::utils::dlss::vulkan::NgxDispatchTable. These functions implement
 * that table with the AMD FidelityFX API (ffx_api, Vulkan backend), so the
 * existing TAA capture/adapter path feeds FSR instead of NVIDIA NGX/DLSS.
 *
 * NGX handle/parameter pointers handed back to the caller are opaque objects
 * owned by this backend; the caller never dereferences them.
 */

#pragma once

#include <vulkan/vulkan.h>

#include <nvsdk_ngx_helpers_vk.h>

#include <string>

#include "utils/dlss/ngx_vulkan.hpp"

namespace renodx::games::detroitbecomehuman::dlss::fsr {

NVSDK_NGX_Result InitializeProject(
    void* context,
    const char* project_id,
    NVSDK_NGX_EngineType engine_type,
    const char* engine_version,
    const wchar_t* application_data_path,
    VkInstance instance,
    VkPhysicalDevice physical_device,
    VkDevice device,
    PFN_vkGetInstanceProcAddr get_instance_proc_addr,
    PFN_vkGetDeviceProcAddr get_device_proc_addr,
    const NVSDK_NGX_FeatureCommonInfo* feature_info,
    NVSDK_NGX_Version sdk_version);

NVSDK_NGX_Result GetCapabilityParameters(
    void* context, NVSDK_NGX_Parameter** parameters);

NVSDK_NGX_Result GetParameterI(
    void* context, NVSDK_NGX_Parameter* parameters, const char* name, int* value);

NVSDK_NGX_Result QueryMode(
    void* context,
    NVSDK_NGX_Parameter* parameters,
    const renodx::utils::dlss::vulkan::ModeQuery* query,
    renodx::utils::dlss::vulkan::ModeSettings* settings);

NVSDK_NGX_Result AllocateParameters(
    void* context, NVSDK_NGX_Parameter** parameters);

NVSDK_NGX_Result DestroyParameters(
    void* context, NVSDK_NGX_Parameter* parameters);

NVSDK_NGX_Result CreateFeature(
    void* context,
    VkDevice device,
    VkCommandBuffer command_buffer,
    NVSDK_NGX_Handle** feature,
    NVSDK_NGX_Parameter* parameters,
    const renodx::utils::dlss::vulkan::FeatureConfig* config);

NVSDK_NGX_Result EvaluateFeature(
    void* context,
    VkCommandBuffer command_buffer,
    NVSDK_NGX_Handle* feature,
    NVSDK_NGX_Parameter* parameters,
    const renodx::utils::dlss::vulkan::EvaluateInfo* info);

NVSDK_NGX_Result ReleaseFeature(void* context, NVSDK_NGX_Handle* feature);

NVSDK_NGX_Result Shutdown(void* context, VkDevice device);

// Human-readable backend state for the RenoDX settings panel, e.g.
// "FSR ready (FSR 3.1.5)" or "FSR runtime DLL missing".
std::string GetStatusText();

// Which backend this session uses (decided once at startup by the add-on).
void SetActive(bool active);
bool IsActive();

}  // namespace renodx::games::detroitbecomehuman::dlss::fsr
