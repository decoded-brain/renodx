/*
 * SPDX-License-Identifier: MIT
 */

#define ImTextureID ImU64
#define DEBUG_LEVEL_0

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <deps/imgui/imgui.h>
#include <include/reshade.hpp>

#include <embed/shaders.h>

#include "../../utils/bitwise.hpp"
#include "../../utils/resource.hpp"
#include "../../utils/settings.hpp"
#include "../../utils/shader.hpp"
#include "../../utils/swapchain.hpp"
#include "./descriptor_tracker.hpp"
#include "./shared.h"

namespace {

constexpr uint32_t TLOU_TONE_MAP_SHADER_HASH = 0x5921B6BB;
constexpr uint32_t TLOU_GUI_SHADER_HASH = 0xECBF2D51;
constexpr uint32_t TLOU_LUT_SIZE = 32u;
constexpr uint32_t TLOU_LUT_GROUP_SIZE = 4u;
constexpr float TLOU_NATIVE_ENCODER_REFERENCE_NITS = 300.f;

constexpr uint64_t C432_OFFSET = 432u;
constexpr uint64_t C684_OFFSET = 684u;
constexpr uint64_t C688_OFFSET = 688u;
constexpr uint64_t C692_OFFSET = 692u;
constexpr uint64_t C700_OFFSET = 700u;
constexpr uint64_t TONE_MAP_CONSTANT_BYTES = C700_OFFSET + sizeof(float);
constexpr uint64_t UI_MAGIC_OFFSET = 16u;
constexpr uint64_t UI_SCALE_OFFSET = 20u;
constexpr uint64_t UI_CONSTANT_BYTES = UI_SCALE_OFFSET + sizeof(float);
constexpr uint32_t UI_MAGIC = 0x52445855u;

ShaderInjectData shader_injection = {};
std::atomic<reshade::api::swapchain*> tracked_swapchain = nullptr;
std::atomic_bool logged_lut_dispatch = false;
std::atomic_bool logged_ui_scale = false;
std::atomic_uint32_t logged_gate_failures = 0u;
bool detected_peak = false;

void LogGateFailureOnce(uint32_t bit, const std::string& message) {
  const uint32_t previous = logged_gate_failures.fetch_or(bit);
  if ((previous & bit) == 0u) {
    reshade::log::message(reshade::log::level::warning, message.c_str());
  }
}

struct NativeToneMapConstants {
  float lut_enabled = 0.f;
  float lut_scale = 0.f;
  float lut_offset = 0.f;
  float post_lut_amount = 0.f;
  float lut_compression = 0.f;
  int output_mode = 0;
};

struct PreviousComputeState {
  reshade::api::pipeline_stage pipeline_stages =
      static_cast<reshade::api::pipeline_stage>(0u);
  reshade::api::pipeline pipeline = {0u};
  reshade::api::pipeline_layout layout = {0u};
  std::vector<reshade::api::descriptor_table> descriptor_tables = {};
};

struct LutDirtyKey {
  int tone_map_type = 0;
  float peak_white_nits = 0.f;
  float diffuse_white_nits = 0.f;
  float cone_response = 0.f;
  float exposure_match = 0.f;
  float vanilla_slope_amount = 0.f;
  float lut_scale = 0.f;
  float lut_offset = 0.f;
  float lut_compression = 0.f;
  float post_lut_amount = 0.f;
  uint64_t t7_view = 0u;
  uint64_t t7_resource = 0u;
  uint64_t t7_generation = 0u;
  uint64_t t8_view = 0u;
  uint64_t t8_resource = 0u;
  uint64_t t8_generation = 0u;

  bool operator==(const LutDirtyKey&) const = default;
};

struct __declspec(uuid("d9260245-7538-43d4-87ac-f6f3438f258c")) DeviceData {
  reshade::api::pipeline_layout lut_layout = {0u};
  reshade::api::sampler lut_sampler = {0u};
  std::array<reshade::api::pipeline, 8> lut_pipelines = {};
  reshade::api::resource lut_resource = {0u};
  reshade::api::resource_view lut_srv = {0u};
  reshade::api::resource_view lut_uav = {0u};
  reshade::api::resource_usage lut_state = reshade::api::resource_usage::unordered_access;
  std::optional<LutDirtyKey> lut_key = std::nullopt;
};

int ToneMapType() {
  return std::clamp(static_cast<int>(std::lround(shader_injection.tone_map_type)), 0, 7);
}

bool IsCustomToneMapperEnabled() {
  return ToneMapType() != 0;
}

bool IsPsychoVEnabled() {
  const int tone_map_type = ToneMapType();
  return tone_map_type >= 1 && tone_map_type <= 5;
}

bool IsNativeHDREnabled() {
  auto* swapchain = tracked_swapchain.load();
  return swapchain != nullptr
         && swapchain->get_color_space() == reshade::api::color_space::hdr10_st2084;
}

bool IsFinalHDRGuiPass(reshade::api::command_list* cmd_list) {
  if (!IsNativeHDREnabled()) return false;

  const auto& render_targets = renodx::utils::swapchain::GetRenderTargets(cmd_list);
  if (render_targets.size() != 1u) return false;

  auto* device = cmd_list->get_device();
  const auto resource = renodx::utils::resource::GetResourceFromView(
      device,
      render_targets[0]);
  if (resource.handle == 0u) return false;

  const auto desc = device->get_resource_desc(resource);
  return desc.type == reshade::api::resource_type::texture_2d
         && desc.texture.format == reshade::api::format::r11g11b10_float;
}

std::span<const uint8_t> GetLutShader(int tone_map_type) {
  switch (tone_map_type) {
    case 1:
      return __lut_psychov17;
    case 2:
      return __lut_psychov22;
    case 3:
      return __lut_psychov24;
    case 4:
      return __lut_psychov25;
    case 5:
      return __lut_psychov30;
    case 6:
      return __lut_renodrt;
    case 7:
      return __lut_neutwo;
    default:
      return {};
  }
}

PreviousComputeState CaptureComputeState(reshade::api::command_list* cmd_list) {
  PreviousComputeState result = {};
  const auto* current_state = tlou::descriptor_tracker::Get(cmd_list);
  if (current_state == nullptr) return result;
  result.pipeline_stages = current_state->compute_pipeline_stages;
  result.pipeline = current_state->compute_pipeline;
  result.layout = current_state->compute_layout;
  result.descriptor_tables = current_state->compute_tables;
  return result;
}

void RestoreComputeState(
    reshade::api::command_list* cmd_list,
    const PreviousComputeState& state) {
  if (state.pipeline.handle != 0u) {
    cmd_list->bind_pipeline(state.pipeline_stages, state.pipeline);
  }
  if (state.layout.handle == 0u) return;
  for (uint32_t i = 0u; i < state.descriptor_tables.size(); ++i) {
    if (state.descriptor_tables[i].handle == 0u) continue;
    cmd_list->bind_descriptor_tables(
        reshade::api::shader_stage::all_compute,
        state.layout,
        i,
        1u,
        &state.descriptor_tables[i]);
  }
}

bool UpdateBoundView(
    reshade::api::device* device,
    const tlou::descriptor_tracker::CommandListData& command_data,
    const tlou::descriptor_tracker::ViewBinding& binding,
    reshade::api::resource_view view) {
  if (device == nullptr
      || !binding.valid
      || view.handle == 0u
      || binding.type
             != reshade::api::descriptor_type::shader_resource_view
      || binding.layout_param >= command_data.compute_tables.size()) {
    return false;
  }
  const auto table = command_data.compute_tables[binding.layout_param];
  if (table.handle == 0u) return false;
  const auto update = reshade::api::descriptor_table_update{
      .table = table,
      .binding = binding.binding,
      .array_offset = binding.array_offset,
      .count = 1u,
      .type = binding.type,
      .descriptors = &view,
  };
  device->update_descriptor_tables(1u, &update);
  return true;
}

bool MapNativeToneMapConstants(
    reshade::api::device* device,
    const reshade::api::buffer_range& range,
    NativeToneMapConstants& constants,
    void** mapped_out) {
  *mapped_out = nullptr;
  if (range.buffer.handle == 0u
      || (range.size != UINT64_MAX && range.size < TONE_MAP_CONSTANT_BYTES)) {
    return false;
  }

  if (!device->map_buffer_region(
          range.buffer,
          range.offset,
          TONE_MAP_CONSTANT_BYTES,
          reshade::api::map_access::read_write,
          mapped_out)
      || *mapped_out == nullptr) {
    return false;
  }

  const auto* bytes = static_cast<const std::byte*>(*mapped_out);
  std::memcpy(&constants.lut_enabled, bytes + C432_OFFSET, sizeof(float));
  std::memcpy(&constants.lut_scale, bytes + C432_OFFSET + sizeof(float), sizeof(float));
  std::memcpy(&constants.lut_offset, bytes + C432_OFFSET + sizeof(float) * 2u, sizeof(float));
  std::memcpy(&constants.post_lut_amount, bytes + C684_OFFSET, sizeof(float));
  std::memcpy(&constants.lut_compression, bytes + C688_OFFSET, sizeof(float));
  std::memcpy(&constants.output_mode, bytes + C692_OFFSET, sizeof(int));

  const bool valid = std::isfinite(constants.lut_enabled)
                     && std::isfinite(constants.lut_scale)
                     && std::isfinite(constants.lut_offset)
                     && std::isfinite(constants.post_lut_amount)
                     && std::isfinite(constants.lut_compression)
                     && constants.lut_scale > 0.f
                     && constants.lut_compression > 0.f;
  if (!valid) {
    device->unmap_buffer_region(range.buffer);
    *mapped_out = nullptr;
  }
  return valid;
}

void PatchMappedNativeToneMapConstants(void* mapped) {
  constexpr float zero = 0.f;
  auto* bytes = static_cast<std::byte*>(mapped);
  std::memcpy(bytes + C684_OFFSET, &zero, sizeof(float));
  std::memcpy(bytes + C700_OFFSET, &zero, sizeof(float));
}

bool EnsureLutLayout(reshade::api::device* device, DeviceData* data) {
  if (data->lut_layout.handle != 0u && data->lut_sampler.handle != 0u) return true;

  if (data->lut_sampler.handle == 0u
      && !device->create_sampler({}, &data->lut_sampler)) {
    return false;
  }

  const std::array<reshade::api::pipeline_layout_param, 4> params = {
      reshade::api::descriptor_range{
          .binding = 0u,
          .dx_register_index = 0u,
          .dx_register_space = 0u,
          .count = 1u,
          .visibility = reshade::api::shader_stage::compute,
          .array_size = 1u,
          .type = reshade::api::descriptor_type::sampler},
      reshade::api::descriptor_range{
          .binding = 0u,
          .dx_register_index = 0u,
          .dx_register_space = 0u,
          .count = 2u,
          .visibility = reshade::api::shader_stage::compute,
          .array_size = 1u,
          .type = reshade::api::descriptor_type::texture_shader_resource_view},
      reshade::api::descriptor_range{
          .binding = 0u,
          .dx_register_index = 0u,
          .dx_register_space = 0u,
          .count = 1u,
          .visibility = reshade::api::shader_stage::compute,
          .array_size = 1u,
          .type = reshade::api::descriptor_type::texture_unordered_access_view},
      reshade::api::constant_range{
          .binding = 0u,
          .dx_register_index = 13u,
          .dx_register_space = 50u,
          .count = sizeof(ShaderInjectData) / sizeof(uint32_t),
          .visibility = reshade::api::shader_stage::compute},
  };
  return device->create_pipeline_layout(
      static_cast<uint32_t>(params.size()),
      params.data(),
      &data->lut_layout);
}

bool EnsureLutPipeline(
    reshade::api::device* device,
    DeviceData* data,
    int tone_map_type) {
  if (tone_map_type < 1 || tone_map_type > 7 || !EnsureLutLayout(device, data)) {
    return false;
  }
  if (data->lut_pipelines[tone_map_type].handle != 0u) return true;

  const auto code = GetLutShader(tone_map_type);
  if (code.empty()) return false;
  reshade::api::shader_desc shader_desc = {
      .code = code.data(),
      .code_size = code.size(),
  };
  reshade::api::pipeline_subobject subobject = {
      .type = reshade::api::pipeline_subobject_type::compute_shader,
      .count = 1u,
      .data = &shader_desc,
  };
  return device->create_pipeline(
      data->lut_layout,
      1u,
      &subobject,
      &data->lut_pipelines[tone_map_type]);
}

bool EnsureLutResource(reshade::api::device* device, DeviceData* data) {
  if (data->lut_resource.handle != 0u
      && data->lut_srv.handle != 0u
      && data->lut_uav.handle != 0u) {
    return true;
  }

  const auto usage = reshade::api::resource_usage::shader_resource
                     | reshade::api::resource_usage::unordered_access;
  const auto desc = reshade::api::resource_desc(
      reshade::api::resource_type::texture_3d,
      TLOU_LUT_SIZE,
      TLOU_LUT_SIZE,
      TLOU_LUT_SIZE,
      1u,
      reshade::api::format::r16g16b16a16_float,
      1u,
      reshade::api::memory_heap::gpu_only,
      usage);
  if (!device->create_resource(
          desc,
          nullptr,
          reshade::api::resource_usage::unordered_access,
          &data->lut_resource)) {
    return false;
  }

  auto view_desc = reshade::api::resource_view_desc{};
  view_desc.type = reshade::api::resource_view_type::texture_3d;
  view_desc.format = reshade::api::format::r16g16b16a16_float;
  view_desc.texture.first_level = 0u;
  view_desc.texture.level_count = 1u;
  view_desc.texture.first_layer = 0u;
  view_desc.texture.layer_count = UINT32_MAX;
  if (!device->create_resource_view(
          data->lut_resource,
          reshade::api::resource_usage::shader_resource,
          view_desc,
          &data->lut_srv)
      || !device->create_resource_view(
          data->lut_resource,
          reshade::api::resource_usage::unordered_access,
          view_desc,
          &data->lut_uav)) {
    if (data->lut_srv.handle != 0u) device->destroy_resource_view(data->lut_srv);
    if (data->lut_uav.handle != 0u) device->destroy_resource_view(data->lut_uav);
    device->destroy_resource(data->lut_resource);
    data->lut_resource = {0u};
    data->lut_srv = {0u};
    data->lut_uav = {0u};
    return false;
  }

  data->lut_state = reshade::api::resource_usage::unordered_access;
  return true;
}

bool ValidateLutView(
    reshade::api::device* device,
    reshade::api::resource_view view) {
  if (view.handle == 0u) return false;
  const auto resource = renodx::utils::resource::GetResourceFromView(device, view);
  if (resource.handle == 0u) return false;
  const auto desc = device->get_resource_desc(resource);
  return desc.type == reshade::api::resource_type::texture_3d
         && desc.texture.width == TLOU_LUT_SIZE
         && desc.texture.height == TLOU_LUT_SIZE
         && desc.texture.depth_or_layers == TLOU_LUT_SIZE;
}

bool DispatchCustomLut(
    reshade::api::command_list* cmd_list,
    DeviceData* data,
    const tlou::descriptor_tracker::CommandListData& bindings,
    ShaderInjectData& lut_constants,
    int tone_map_type) {
  auto* device = cmd_list->get_device();
  if (!EnsureLutPipeline(device, data, tone_map_type)
      || !EnsureLutResource(device, data)) {
    return false;
  }

  const PreviousComputeState previous_state = CaptureComputeState(cmd_list);
  if (data->lut_state != reshade::api::resource_usage::unordered_access) {
    const auto before = data->lut_state;
    const auto after = reshade::api::resource_usage::unordered_access;
    cmd_list->barrier(1u, &data->lut_resource, &before, &after);
    data->lut_state = after;
  }

  cmd_list->bind_pipeline(
      reshade::api::pipeline_stage::all_compute,
      data->lut_pipelines[tone_map_type]);
  tlou::descriptor_tracker::ignore_updates = true;
  cmd_list->push_descriptors(
      reshade::api::shader_stage::all_compute,
      data->lut_layout,
      0u,
      reshade::api::descriptor_table_update{
          {},
          0u,
          0u,
          1u,
          reshade::api::descriptor_type::sampler,
          &data->lut_sampler});
  const std::array<reshade::api::resource_view, 2> source_views = {
      bindings.compute_t7.view,
      bindings.compute_t8.valid ? bindings.compute_t8.view : bindings.compute_t7.view,
  };
  cmd_list->push_descriptors(
      reshade::api::shader_stage::all_compute,
      data->lut_layout,
      1u,
      reshade::api::descriptor_table_update{
          {},
          0u,
          0u,
          static_cast<uint32_t>(source_views.size()),
          reshade::api::descriptor_type::texture_shader_resource_view,
          source_views.data()});
  cmd_list->push_descriptors(
      reshade::api::shader_stage::all_compute,
      data->lut_layout,
      2u,
      reshade::api::descriptor_table_update{
          {},
          0u,
          0u,
          1u,
          reshade::api::descriptor_type::texture_unordered_access_view,
          &data->lut_uav});
  cmd_list->push_constants(
      reshade::api::shader_stage::all_compute,
      data->lut_layout,
      3u,
      0u,
      sizeof(lut_constants) / sizeof(uint32_t),
      &lut_constants);
  cmd_list->dispatch(
      TLOU_LUT_SIZE / TLOU_LUT_GROUP_SIZE,
      TLOU_LUT_SIZE / TLOU_LUT_GROUP_SIZE,
      TLOU_LUT_SIZE / TLOU_LUT_GROUP_SIZE);

  const auto before = reshade::api::resource_usage::unordered_access;
  const auto after = reshade::api::resource_usage::shader_resource;
  cmd_list->barrier(1u, &data->lut_resource, &before, &after);
  data->lut_state = after;
  RestoreComputeState(cmd_list, previous_state);
  tlou::descriptor_tracker::ignore_updates = false;
  return true;
}

bool OnToneMapDispatch(reshade::api::command_list* cmd_list) {
  const int tone_map_type = ToneMapType();
  if (tone_map_type == 0) return true;

  auto* bindings = tlou::descriptor_tracker::Get(cmd_list);
  if (bindings == nullptr) {
    LogGateFailureOnce(1u << 1u, "TLOU Part I: LUT gate has no command-list bindings");
    return true;
  }
  tlou::descriptor_tracker::ResolvedLayoutRoute route = {};
  const bool resolved = tlou::descriptor_tracker::ResolveToneMapBindings(
      cmd_list,
      bindings,
      &route);
  if (!resolved) {
    std::stringstream message;
    message << "TLOU Part I: sparse descriptor resolve failed";
    message << " layout=0x" << std::hex << bindings->compute_layout.handle;
    message << " signature=0x" << route.signature;
    message << std::dec << " tables=" << bindings->compute_tables.size();
    message << " route=" << route.t7.valid;
    message << "/" << route.t8.valid;
    message << "/" << route.b0.valid;
    message << " t7=" << bindings->compute_t7.valid;
    message << " b0=" << bindings->compute_b0.valid;
    LogGateFailureOnce(1u << 2u, message.str());
    return true;
  }
  if (bindings->compute_layout.handle == 0u
      || bindings->compute_t7.layout.handle != bindings->compute_layout.handle
      || bindings->compute_b0.layout.handle != bindings->compute_layout.handle) {
    LogGateFailureOnce(1u << 3u, "TLOU Part I: resolved descriptor layout mismatch");
    return true;
  }
  if (!ValidateLutView(cmd_list->get_device(), bindings->compute_t7.view)) {
    LogGateFailureOnce(1u << 4u, "TLOU Part I: resolved t7 is not a 32x32x32 LUT");
    return true;
  }

  NativeToneMapConstants native = {};
  auto* device = cmd_list->get_device();
  void* mapped_constants = nullptr;
  if (!MapNativeToneMapConstants(
          device,
          bindings->compute_b0.range,
          native,
          &mapped_constants)) {
    LogGateFailureOnce(1u << 5u, "TLOU Part I: native b0 could not be mapped");
    return true;
  }
  if (native.output_mode != 3 || native.lut_enabled <= 0.5f) {
    device->unmap_buffer_region(bindings->compute_b0.range.buffer);
    const std::string message =
        "TLOU Part I: native HDR/LUT gate failed (mode="
        + std::to_string(native.output_mode)
        + ", lut=" + std::to_string(native.lut_enabled) + ")";
    LogGateFailureOnce(1u << 6u, message);
    return true;
  }
  if (native.post_lut_amount > 0.f
      && (!bindings->compute_t8.valid
          || !ValidateLutView(device, bindings->compute_t8.view))) {
    device->unmap_buffer_region(bindings->compute_b0.range.buffer);
    LogGateFailureOnce(1u << 7u, "TLOU Part I: active post-LUT t8 was not resolved");
    return true;
  }

  auto* data = device->get_private_data<DeviceData>();
  if (data == nullptr) data = device->create_private_data<DeviceData>();
  if (data == nullptr
      || !EnsureLutPipeline(device, data, tone_map_type)
      || !EnsureLutResource(device, data)) {
    device->unmap_buffer_region(bindings->compute_b0.range.buffer);
    return true;
  }

  ShaderInjectData lut_constants = shader_injection;
  lut_constants.tone_map_type = static_cast<float>(tone_map_type);
  lut_constants.lut_scale = native.lut_scale;
  lut_constants.lut_offset = native.lut_offset;
  lut_constants.lut_compression = native.lut_compression;
  lut_constants.post_lut_amount = native.post_lut_amount;

  const LutDirtyKey current_key = {
      .tone_map_type = tone_map_type,
      .peak_white_nits = shader_injection.peak_white_nits,
      .diffuse_white_nits = shader_injection.diffuse_white_nits,
      .cone_response = shader_injection.cone_response,
      .exposure_match = shader_injection.exposure_match,
      .vanilla_slope_amount = shader_injection.vanilla_slope_amount,
      .lut_scale = native.lut_scale,
      .lut_offset = native.lut_offset,
      .lut_compression = native.lut_compression,
      .post_lut_amount = native.post_lut_amount,
      .t7_view = bindings->compute_t7.view.handle,
      .t7_resource = bindings->compute_t7.resource.handle,
      .t7_generation = bindings->compute_t7.generation,
      .t8_view = bindings->compute_t8.view.handle,
      .t8_resource = bindings->compute_t8.resource.handle,
      .t8_generation = bindings->compute_t8.generation,
  };
  const bool rebuild_lut = !data->lut_key.has_value()
                           || data->lut_key.value() != current_key;
  if (rebuild_lut
      && !DispatchCustomLut(
          cmd_list,
          data,
          *bindings,
          lut_constants,
          tone_map_type)) {
    device->unmap_buffer_region(bindings->compute_b0.range.buffer);
    return true;
  }
  if (rebuild_lut) {
    data->lut_key = current_key;
  }

  tlou::descriptor_tracker::ignore_updates = true;
  const bool updated_view = UpdateBoundView(
      device,
      *bindings,
      bindings->compute_t7,
      data->lut_srv);
  tlou::descriptor_tracker::ignore_updates = false;
  if (!updated_view) {
    device->unmap_buffer_region(bindings->compute_b0.range.buffer);
    return true;
  }

  PatchMappedNativeToneMapConstants(mapped_constants);
  device->unmap_buffer_region(bindings->compute_b0.range.buffer);
  if (!logged_lut_dispatch.exchange(true)) {
    const std::string message =
        "TLOU Part I: active cached 32x32x32 tone-map LUT mode "
        + std::to_string(tone_map_type);
    reshade::log::message(reshade::log::level::info, message.c_str());
  }
  return true;
}

bool PatchGuiConstants(reshade::api::command_list* cmd_list) {
  auto* bindings = tlou::descriptor_tracker::Get(cmd_list);
  if (bindings == nullptr
      || !bindings->vertex_b0.valid
      || bindings->graphics_layout.handle == 0u
      || bindings->vertex_b0.layout.handle != bindings->graphics_layout.handle) {
    return false;
  }

  const auto& range = bindings->vertex_b0.range;
  if (range.buffer.handle == 0u
      || (range.size != UINT64_MAX && range.size < UI_CONSTANT_BYTES)) {
    return false;
  }

  void* mapped = nullptr;
  auto* device = cmd_list->get_device();
  if (!device->map_buffer_region(
          range.buffer,
          range.offset + UI_MAGIC_OFFSET,
          UI_CONSTANT_BYTES - UI_MAGIC_OFFSET,
          reshade::api::map_access::write_only,
          &mapped)
      || mapped == nullptr) {
    return false;
  }

  const bool scale_ui = IsCustomToneMapperEnabled() && IsFinalHDRGuiPass(cmd_list);
  float gamma_scale = 1.f;
  if (scale_ui) {
    const float linear_scale = std::max(
        shader_injection.graphics_white_nits
            / TLOU_NATIVE_ENCODER_REFERENCE_NITS,
        0.f);
    gamma_scale = std::pow(linear_scale, 1.f / 2.4f);
  }
  auto* bytes = static_cast<std::byte*>(mapped);
  std::memcpy(bytes, &UI_MAGIC, sizeof(UI_MAGIC));
  std::memcpy(
      bytes + UI_SCALE_OFFSET - UI_MAGIC_OFFSET,
      &gamma_scale,
      sizeof(gamma_scale));
  device->unmap_buffer_region(range.buffer);
  if (scale_ui && !logged_ui_scale.exchange(true)) {
    reshade::log::message(
        reshade::log::level::info,
        "TLOU Part I: active HDR UI brightness scale");
  }
  return true;
}

bool OnDispatch(
    reshade::api::command_list* cmd_list,
    uint32_t,
    uint32_t,
    uint32_t) {
  auto* shader_state = renodx::utils::shader::GetCurrentState(cmd_list);
  if (shader_state == nullptr) return false;
  auto* compute_state = renodx::utils::shader::GetCurrentComputeState(
      shader_state);
  if (renodx::utils::shader::GetCurrentComputeShaderHash(compute_state)
      != TLOU_TONE_MAP_SHADER_HASH) {
    return false;
  }
  renodx::utils::shader::PopulateStageState(compute_state);
  auto* bindings = tlou::descriptor_tracker::Get(cmd_list);
  if (bindings != nullptr
      && compute_state->pipeline_details != nullptr
      && compute_state->pipeline_details->layout.handle != 0u) {
    bindings->compute_layout = compute_state->pipeline_details->layout;
  }
  OnToneMapDispatch(cmd_list);
  return false;
}

void PatchCurrentGuiDraw(reshade::api::command_list* cmd_list) {
  auto* shader_state = renodx::utils::shader::GetCurrentState(cmd_list);
  if (shader_state != nullptr
      && renodx::utils::shader::GetCurrentVertexShaderHash(shader_state)
             == TLOU_GUI_SHADER_HASH) {
    PatchGuiConstants(cmd_list);
  }
}

bool OnDraw(
    reshade::api::command_list* cmd_list,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t) {
  PatchCurrentGuiDraw(cmd_list);
  return false;
}

bool OnDrawIndexed(
    reshade::api::command_list* cmd_list,
    uint32_t,
    uint32_t,
    uint32_t,
    int32_t,
    uint32_t) {
  PatchCurrentGuiDraw(cmd_list);
  return false;
}

auto* tone_map_peak_nits_setting = new renodx::utils::settings::Setting{
    .key = "ToneMapPeakNits",
    .binding = &shader_injection.peak_white_nits,
    .default_value = 1000.f,
    .can_reset = false,
    .label = "Peak Brightness",
    .section = "Tone Mapping",
    .tooltip = "Sets the display peak brightness in nits.",
    .min = 48.f,
    .max = 4000.f,
    .is_enabled = &IsCustomToneMapperEnabled,
};

renodx::utils::settings::Settings settings = {
    new renodx::utils::settings::Setting{
        .key = "ToneMapType",
        .binding = &shader_injection.tone_map_type,
        .value_type = renodx::utils::settings::SettingValueType::INTEGER,
        .default_value = TLOU_TONE_MAP_TYPE_PSYCHOV17,
        .label = "Tone Mapper",
        .section = "Tone Mapping",
        .tooltip = "Vanilla preserves the original game shader output.",
        .labels = {
            "Vanilla",
            "PsychoV-17",
            "PsychoV-22",
            "PsychoV-24",
            "PsychoV-25",
            "PsychoV-30",
            "RenoDRT",
            "NeuTwo",
        },
    },
    tone_map_peak_nits_setting,
    new renodx::utils::settings::Setting{
        .key = "ToneMapGameNits",
        .binding = &shader_injection.diffuse_white_nits,
        .default_value = 203.f,
        .label = "Game Brightness",
        .section = "Tone Mapping",
        .tooltip = "Sets scene/reference white in nits.",
        .min = 48.f,
        .max = 500.f,
        .is_enabled = &IsCustomToneMapperEnabled,
    },
    new renodx::utils::settings::Setting{
        .key = "ToneMapUINits",
        .binding = &shader_injection.graphics_white_nits,
        .default_value = 203.f,
        .label = "UI Brightness",
        .section = "Tone Mapping",
        .tooltip = "Sets UI, HUD, and subtitle white in nits.",
        .min = 48.f,
        .max = 500.f,
        .is_enabled = &IsCustomToneMapperEnabled,
    },
    new renodx::utils::settings::Setting{
        .key = "ColorGradeConeResponse",
        .binding = &shader_injection.cone_response,
        .default_value = 50.f,
        .label = "PsychoV Response",
        .section = "Tone Mapping",
        .tooltip = "Scales the PsychoV cone response.",
        .min = 0.f,
        .max = 100.f,
        .is_enabled = &IsPsychoVEnabled,
        .parse = [](float value) { return value * 0.02f; },
    },
    new renodx::utils::settings::Setting{
        .key = "ToneMapPsychoVExposureMatch",
        .binding = &shader_injection.exposure_match,
        .value_type = renodx::utils::settings::SettingValueType::INTEGER,
        .default_value = 1.f,
        .label = "Exposure Match",
        .section = "Tone Mapping",
        .tooltip = "Matches the native neutral 18% anchor.",
        .labels = {"Off", "On"},
        .is_enabled = &IsPsychoVEnabled,
    },
    new renodx::utils::settings::Setting{
        .key = "ToneMapPsychoVVanillaHDRSlope",
        .binding = &shader_injection.vanilla_slope_amount,
        .default_value = 100.f,
        .label = "Vanilla HDR Slope",
        .section = "Tone Mapping",
        .tooltip = "Blends identity response toward the native local log slope.",
        .min = 0.f,
        .max = 100.f,
        .is_enabled = &IsPsychoVEnabled,
        .parse = [](float value) { return value * 0.01f; },
    },
};

void OnPresetOff() {
  renodx::utils::settings::UpdateSettings({
      {"ToneMapType", TLOU_TONE_MAP_TYPE_VANILLA},
      {"ToneMapPeakNits", tone_map_peak_nits_setting->default_value},
      {"ToneMapGameNits", 203.f},
      {"ToneMapUINits", 203.f},
      {"ColorGradeConeResponse", 50.f},
      {"ToneMapPsychoVExposureMatch", 1.f},
      {"ToneMapPsychoVVanillaHDRSlope", 100.f},
  });
}

void OnInitDevice(reshade::api::device* device) {
  device->create_private_data<DeviceData>();
  tlou::descriptor_tracker::OnInitDevice(device);
}

void OnDestroyDevice(reshade::api::device* device) {
  auto* data = device->get_private_data<DeviceData>();
  if (data != nullptr) {
    for (const auto pipeline : data->lut_pipelines) {
      if (pipeline.handle != 0u) device->destroy_pipeline(pipeline);
    }
    if (data->lut_layout.handle != 0u) device->destroy_pipeline_layout(data->lut_layout);
    if (data->lut_sampler.handle != 0u) device->destroy_sampler(data->lut_sampler);
    if (data->lut_srv.handle != 0u) device->destroy_resource_view(data->lut_srv);
    if (data->lut_uav.handle != 0u) device->destroy_resource_view(data->lut_uav);
    if (data->lut_resource.handle != 0u) device->destroy_resource(data->lut_resource);
    device->destroy_private_data<DeviceData>();
  }
  tlou::descriptor_tracker::OnDestroyDevice(device);
}

void OnInitSwapchain(reshade::api::swapchain* swapchain, bool) {
  tracked_swapchain.store(swapchain);
  if (detected_peak) return;

  const auto peak = renodx::utils::swapchain::GetPeakNits(swapchain);
  const float detected_default = peak.has_value()
                                     ? std::clamp(std::round(peak.value()), 48.f, 4000.f)
                                     : 1000.f;
  tone_map_peak_nits_setting->default_value = detected_default;
  tone_map_peak_nits_setting->can_reset = true;

  const std::string preset_section = renodx::utils::settings::global_name
                                     + "-preset"
                                     + std::to_string(renodx::utils::settings::preset_index);
  float saved_peak = 0.f;
  if (!reshade::get_config_value(
          nullptr,
          preset_section.c_str(),
          tone_map_peak_nits_setting->key.c_str(),
          saved_peak)) {
    tone_map_peak_nits_setting->Set(detected_default)->Write();
  }
  detected_peak = true;
}

void OnDestroySwapchain(reshade::api::swapchain* swapchain, bool) {
  auto* expected = swapchain;
  tracked_swapchain.compare_exchange_strong(expected, nullptr);
}

}  // namespace

extern "C" __declspec(dllexport) constexpr const char* NAME = "RenoDX";
extern "C" __declspec(dllexport) constexpr const char* DESCRIPTION =
    "RenoDX HDR tone mappers for The Last of Us Part I";

BOOL APIENTRY DllMain(HMODULE h_module, DWORD reason, LPVOID) {
  switch (reason) {
    case DLL_PROCESS_ATTACH:
      if (!reshade::register_addon(h_module)) return FALSE;
      renodx::utils::shader::QueueCompileTimeReplacement(
          TLOU_GUI_SHADER_HASH,
          __0xECBF2D51);
      renodx::utils::shader::QueueRuntimeReplacement(
          TLOU_GUI_SHADER_HASH,
          __0xECBF2D51);
      reshade::register_event<reshade::addon_event::init_device>(OnInitDevice);
      reshade::register_event<reshade::addon_event::destroy_device>(OnDestroyDevice);
      reshade::register_event<reshade::addon_event::init_swapchain>(OnInitSwapchain);
      reshade::register_event<reshade::addon_event::destroy_swapchain>(OnDestroySwapchain);
      reshade::register_event<reshade::addon_event::init_command_list>(
          tlou::descriptor_tracker::OnInitCommandList);
      reshade::register_event<reshade::addon_event::destroy_command_list>(
          tlou::descriptor_tracker::OnDestroyCommandList);
      reshade::register_event<reshade::addon_event::reset_command_list>(
          tlou::descriptor_tracker::OnResetCommandList);
      reshade::register_event<reshade::addon_event::bind_pipeline>(
          tlou::descriptor_tracker::OnBindPipeline);
      reshade::register_event<reshade::addon_event::bind_descriptor_tables>(
          tlou::descriptor_tracker::OnBindDescriptorTables);
      reshade::register_event<reshade::addon_event::push_descriptors>(
          tlou::descriptor_tracker::OnPushDescriptors);
      reshade::register_event<reshade::addon_event::init_resource_view>(
          tlou::descriptor_tracker::OnInitResourceView);
      reshade::register_event<reshade::addon_event::destroy_resource_view>(
          tlou::descriptor_tracker::OnDestroyResourceView);
      reshade::register_event<reshade::addon_event::update_descriptor_tables>(
          tlou::descriptor_tracker::OnUpdateDescriptorTables);
      reshade::register_event<reshade::addon_event::copy_descriptor_tables>(
          tlou::descriptor_tracker::OnCopyDescriptorTables);
      reshade::register_event<reshade::addon_event::destroy_pipeline_layout>(
          tlou::descriptor_tracker::OnDestroyPipelineLayout);
      reshade::register_event<reshade::addon_event::update_texture_region>(
          tlou::descriptor_tracker::OnUpdateTextureRegion);
      reshade::register_event<reshade::addon_event::copy_resource>(
          tlou::descriptor_tracker::OnCopyResource);
      reshade::register_event<reshade::addon_event::copy_buffer_to_texture>(
          tlou::descriptor_tracker::OnCopyBufferToTexture);
      reshade::register_event<reshade::addon_event::copy_texture_region>(
          tlou::descriptor_tracker::OnCopyTextureRegion);
      reshade::register_event<reshade::addon_event::dispatch>(OnDispatch);
      reshade::register_event<reshade::addon_event::draw>(OnDraw);
      reshade::register_event<reshade::addon_event::draw_indexed>(OnDrawIndexed);
      break;

    case DLL_PROCESS_DETACH:
      reshade::unregister_event<reshade::addon_event::draw_indexed>(OnDrawIndexed);
      reshade::unregister_event<reshade::addon_event::draw>(OnDraw);
      reshade::unregister_event<reshade::addon_event::dispatch>(OnDispatch);
      reshade::unregister_event<reshade::addon_event::copy_texture_region>(
          tlou::descriptor_tracker::OnCopyTextureRegion);
      reshade::unregister_event<reshade::addon_event::copy_buffer_to_texture>(
          tlou::descriptor_tracker::OnCopyBufferToTexture);
      reshade::unregister_event<reshade::addon_event::copy_resource>(
          tlou::descriptor_tracker::OnCopyResource);
      reshade::unregister_event<reshade::addon_event::update_texture_region>(
          tlou::descriptor_tracker::OnUpdateTextureRegion);
      reshade::unregister_event<reshade::addon_event::destroy_pipeline_layout>(
          tlou::descriptor_tracker::OnDestroyPipelineLayout);
      reshade::unregister_event<reshade::addon_event::copy_descriptor_tables>(
          tlou::descriptor_tracker::OnCopyDescriptorTables);
      reshade::unregister_event<reshade::addon_event::update_descriptor_tables>(
          tlou::descriptor_tracker::OnUpdateDescriptorTables);
      reshade::unregister_event<reshade::addon_event::destroy_resource_view>(
          tlou::descriptor_tracker::OnDestroyResourceView);
      reshade::unregister_event<reshade::addon_event::init_resource_view>(
          tlou::descriptor_tracker::OnInitResourceView);
      reshade::unregister_event<reshade::addon_event::push_descriptors>(
          tlou::descriptor_tracker::OnPushDescriptors);
      reshade::unregister_event<reshade::addon_event::bind_descriptor_tables>(
          tlou::descriptor_tracker::OnBindDescriptorTables);
      reshade::unregister_event<reshade::addon_event::bind_pipeline>(
          tlou::descriptor_tracker::OnBindPipeline);
      reshade::unregister_event<reshade::addon_event::reset_command_list>(
          tlou::descriptor_tracker::OnResetCommandList);
      reshade::unregister_event<reshade::addon_event::destroy_command_list>(
          tlou::descriptor_tracker::OnDestroyCommandList);
      reshade::unregister_event<reshade::addon_event::init_command_list>(
          tlou::descriptor_tracker::OnInitCommandList);
      reshade::unregister_event<reshade::addon_event::destroy_swapchain>(OnDestroySwapchain);
      reshade::unregister_event<reshade::addon_event::init_swapchain>(OnInitSwapchain);
      reshade::unregister_event<reshade::addon_event::destroy_device>(OnDestroyDevice);
      reshade::unregister_event<reshade::addon_event::init_device>(OnInitDevice);
      renodx::utils::shader::UnqueueRuntimeReplacement(
          TLOU_GUI_SHADER_HASH,
          __0xECBF2D51);
      renodx::utils::shader::UnqueueCompileTimeReplacement(TLOU_GUI_SHADER_HASH);
      reshade::unregister_addon(h_module);
      break;
  }

  renodx::utils::settings::Use(reason, &settings, &OnPresetOff);
  renodx::utils::swapchain::Use(reason);
  renodx::utils::shader::Use(reason);
  return TRUE;
}
