#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include <include/reshade.hpp>

#include "../../utils/bitwise.hpp"
#include "../../utils/data.hpp"
#include "../../utils/pipeline_layout.hpp"

namespace tlou::descriptor_tracker {

constexpr uint32_t LUT_SIZE = 32u;
constexpr uint64_t MIN_TONE_MAP_CBV_SIZE = 832u;

using RegisterSlot = std::pair<uint32_t, uint32_t>;
inline thread_local bool ignore_updates = false;

struct DescriptorKey {
  uint64_t heap = 0u;
  uint32_t index = 0u;

  bool operator==(const DescriptorKey&) const = default;
};

struct DescriptorKeyHash {
  size_t operator()(const DescriptorKey& key) const noexcept {
    return std::hash<uint64_t>{}(key.heap)
           ^ (std::hash<uint32_t>{}(key.index) << 1u);
  }
};

struct SparseDescriptor {
  reshade::api::descriptor_type type = reshade::api::descriptor_type::sampler;
  reshade::api::resource_view view = {0u};
  reshade::api::resource resource = {0u};
  reshade::api::buffer_range buffer = {};
};

struct CandidateView {
  reshade::api::resource_view view = {0u};
  reshade::api::resource resource = {0u};
};

struct RouteSlot {
  bool valid = false;
  uint32_t layout_param = 0u;
  uint32_t binding = 0u;
  reshade::api::descriptor_type type =
      reshade::api::descriptor_type::shader_resource_view;
};

struct ResolvedLayoutRoute {
  RouteSlot t7 = {};
  RouteSlot t8 = {};
  RouteSlot b0 = {};
  uint64_t signature = 1469598103934665603ull;
};

struct ViewBinding {
  bool valid = false;
  reshade::api::shader_stage stages =
      static_cast<reshade::api::shader_stage>(0u);
  reshade::api::pipeline_layout layout = {0u};
  uint32_t layout_param = 0u;
  uint32_t binding = 0u;
  uint32_t array_offset = 0u;
  reshade::api::descriptor_type type =
      reshade::api::descriptor_type::shader_resource_view;
  reshade::api::resource_view view = {0u};
  reshade::api::resource resource = {0u};
  uint64_t generation = 0u;
  reshade::api::sampler sampler = {0u};
};

struct BufferBinding {
  bool valid = false;
  reshade::api::shader_stage stages =
      static_cast<reshade::api::shader_stage>(0u);
  reshade::api::pipeline_layout layout = {0u};
  uint32_t layout_param = 0u;
  uint32_t binding = 0u;
  uint32_t array_offset = 0u;
  reshade::api::buffer_range range = {};
};

struct __declspec(uuid("75c21043-7b05-4366-bf63-e45cb83b2474")) CommandListData {
  ViewBinding compute_t7 = {};
  ViewBinding compute_t8 = {};
  BufferBinding compute_b0 = {};
  BufferBinding vertex_b0 = {};
  reshade::api::pipeline_stage compute_pipeline_stages =
      static_cast<reshade::api::pipeline_stage>(0u);
  reshade::api::pipeline compute_pipeline = {0u};
  reshade::api::pipeline_layout compute_layout = {0u};
  std::vector<reshade::api::descriptor_table> compute_tables = {};
  reshade::api::pipeline_layout graphics_layout = {0u};
};

struct __declspec(uuid("2b82a3dd-f21f-4538-bcdb-b06dc63723e7")) DeviceData {
  std::shared_mutex mutex;
  std::unordered_map<uint64_t, CandidateView> candidate_views;
  std::unordered_map<uint64_t, uint64_t> resource_generations;
  std::unordered_map<DescriptorKey, SparseDescriptor, DescriptorKeyHash>
      sparse_descriptors;
  std::unordered_map<uint64_t, ResolvedLayoutRoute> routes;
};

inline CommandListData* Get(reshade::api::command_list* cmd_list) {
  if (cmd_list == nullptr) return nullptr;
  auto* data = cmd_list->get_private_data<CommandListData>();
  return data != nullptr ? data : cmd_list->create_private_data<CommandListData>();
}

inline bool GetDescriptorKey(
    reshade::api::device* device,
    reshade::api::descriptor_table table,
    uint32_t binding,
    uint32_t array_offset,
    DescriptorKey& key) {
  if (device == nullptr || table.handle == 0u) return false;
  reshade::api::descriptor_heap heap = {0u};
  uint32_t index = 0u;
  device->get_descriptor_heap_offset(
      table,
      binding,
      array_offset,
      &heap,
      &index);
  if (heap.handle == 0u) return false;
  key = {.heap = heap.handle, .index = index};
  return true;
}

inline bool IsCandidateLut(
    reshade::api::device* device,
    reshade::api::resource resource,
    reshade::api::resource_usage usage) {
  if (device == nullptr
      || resource.handle == 0u
      || !renodx::utils::bitwise::HasFlag(
          usage,
          reshade::api::resource_usage::shader_resource)) {
    return false;
  }
  const auto desc = device->get_resource_desc(resource);
  return desc.type == reshade::api::resource_type::texture_3d
         && desc.texture.width == LUT_SIZE
         && desc.texture.height == LUT_SIZE
         && desc.texture.depth_or_layers == LUT_SIZE
         && desc.texture.format == reshade::api::format::r8g8b8a8_unorm;
}

inline uint64_t GetGeneration(DeviceData* data, reshade::api::resource resource) {
  if (data == nullptr || resource.handle == 0u) return 0u;
  const auto it = data->resource_generations.find(resource.handle);
  return it == data->resource_generations.end() ? 0u : it->second;
}

inline void MarkResourceWritten(reshade::api::device* device, reshade::api::resource resource) {
  if (device == nullptr || resource.handle == 0u) return;
  auto* data = device->get_private_data<DeviceData>();
  if (data == nullptr) return;
  const std::unique_lock lock(data->mutex);
  const auto it = data->resource_generations.find(resource.handle);
  if (it != data->resource_generations.end()) ++it->second;
}

inline bool ResolveRegister(
    reshade::api::pipeline_layout layout,
    uint32_t layout_param,
    const reshade::api::descriptor_table_update& update,
    uint32_t descriptor_index,
    RegisterSlot& slot) {
  bool resolved = false;
  renodx::utils::pipeline_layout::GetPipelineLayoutData(
      layout,
      [&](const renodx::utils::pipeline_layout::PipelineLayoutData* layout_data) {
        if (layout_param >= layout_data->params.size()) return;
        const auto& param = layout_data->params[layout_param];
        const uint32_t binding = update.binding + descriptor_index;
        if (param.type == reshade::api::pipeline_layout_param_type::push_descriptors) {
          slot = {
              param.push_descriptors.dx_register_index + binding,
              param.push_descriptors.dx_register_space};
          resolved = true;
          return;
        }
        if (layout_param >= layout_data->ranges.size()) return;
        for (const auto& range : layout_data->ranges[layout_param]) {
          const bool in_range = binding >= range.binding
                                && (range.count == UINT32_MAX
                                    || binding < range.binding + range.count);
          if (!in_range) continue;
          slot = {
              range.dx_register_index + (binding - range.binding),
              range.dx_register_space};
          resolved = true;
          return;
        }
      });
  return resolved;
}

inline void OnInitDevice(reshade::api::device* device) {
  device->create_private_data<DeviceData>();
}

inline void OnDestroyDevice(reshade::api::device* device) {
  device->destroy_private_data<DeviceData>();
}

inline void OnInitCommandList(reshade::api::command_list* cmd_list) {
  cmd_list->create_private_data<CommandListData>();
}

inline void OnDestroyCommandList(reshade::api::command_list* cmd_list) {
  cmd_list->destroy_private_data<CommandListData>();
}

inline void OnResetCommandList(reshade::api::command_list* cmd_list) {
  if (auto* data = Get(cmd_list); data != nullptr) *data = {};
}

inline void OnBindPipeline(
    reshade::api::command_list* cmd_list,
    reshade::api::pipeline_stage stages,
    reshade::api::pipeline pipeline) {
  auto* data = Get(cmd_list);
  if (data == nullptr) return;
  if ((static_cast<uint32_t>(stages)
       & static_cast<uint32_t>(reshade::api::pipeline_stage::all_compute))
      == 0u) {
    return;
  }
  data->compute_pipeline_stages = stages;
  data->compute_pipeline = pipeline;
}

inline void OnBindDescriptorTables(
    reshade::api::command_list* cmd_list,
    reshade::api::shader_stage stages,
    reshade::api::pipeline_layout layout,
    uint32_t first,
    uint32_t count,
    const reshade::api::descriptor_table* tables) {
  auto* data = Get(cmd_list);
  if (data == nullptr) return;
  const bool compute = renodx::utils::bitwise::HasFlag(
      stages,
      reshade::api::shader_stage::all_compute);
  const bool graphics = renodx::utils::bitwise::HasFlag(
      stages,
      reshade::api::shader_stage::all_graphics);
  if (graphics) data->graphics_layout = layout;
  if (!compute) return;

  if (data->compute_layout != layout) {
    data->compute_tables.clear();
    data->compute_t7 = {};
    data->compute_t8 = {};
    data->compute_b0 = {};
  }
  data->compute_layout = layout;
  if (layout.handle == 0u) return;
  if (data->compute_tables.size() < first + count) {
    data->compute_tables.resize(first + count);
  }
  for (uint32_t i = 0u; i < count; ++i) {
    data->compute_tables[first + i] = tables[i];
  }
}

inline void OnPushDescriptors(
    reshade::api::command_list* cmd_list,
    reshade::api::shader_stage stages,
    reshade::api::pipeline_layout layout,
    uint32_t layout_param,
    const reshade::api::descriptor_table_update& update) {
  if (ignore_updates) return;
  auto* data = Get(cmd_list);
  if (data == nullptr) return;
  const bool compute = renodx::utils::bitwise::HasFlag(
      stages,
      reshade::api::shader_stage::compute);
  const bool vertex = renodx::utils::bitwise::HasFlag(
      stages,
      reshade::api::shader_stage::vertex);
  const bool constant_buffer =
      update.type == reshade::api::descriptor_type::constant_buffer;
  const bool resource_view =
      update.type == reshade::api::descriptor_type::shader_resource_view
      || update.type == reshade::api::descriptor_type::sampler_with_resource_view;
  if ((!constant_buffer && !resource_view)
      || (constant_buffer && !compute && !vertex)
      || (resource_view && !compute)) {
    return;
  }

  for (uint32_t i = 0u; i < update.count; ++i) {
    RegisterSlot slot = {};
    if (!ResolveRegister(layout, layout_param, update, i, slot)
        || slot.second != 0u) {
      continue;
    }
    if (constant_buffer && slot.first == 0u) {
      const auto range =
          static_cast<const reshade::api::buffer_range*>(update.descriptors)[i];
      const BufferBinding binding = {
          .valid = range.buffer.handle != 0u,
          .stages = stages,
          .layout = layout,
          .layout_param = layout_param,
          .binding = update.binding + i,
          .array_offset = update.array_offset,
          .range = range,
      };
      if (compute) data->compute_b0 = binding;
      if (vertex) data->vertex_b0 = binding;
      continue;
    }
    if (!resource_view || (slot.first != 7u && slot.first != 8u)) continue;

    reshade::api::resource_view view = {0u};
    reshade::api::sampler sampler = {0u};
    if (update.type == reshade::api::descriptor_type::sampler_with_resource_view) {
      const auto pair =
          static_cast<const reshade::api::sampler_with_resource_view*>(
              update.descriptors)[i];
      view = pair.view;
      sampler = pair.sampler;
    } else {
      view = static_cast<const reshade::api::resource_view*>(update.descriptors)[i];
    }

    ViewBinding binding = {
        .valid = view.handle != 0u,
        .stages = stages,
        .layout = layout,
        .layout_param = layout_param,
        .binding = update.binding + i,
        .array_offset = update.array_offset,
        .type = update.type,
        .view = view,
        .sampler = sampler,
    };
    if (slot.first == 7u) {
      data->compute_t7 = binding;
    } else {
      data->compute_t8 = binding;
    }
  }
}

inline void OnInitResourceView(
    reshade::api::device* device,
    reshade::api::resource resource,
    reshade::api::resource_usage usage,
    const reshade::api::resource_view_desc&,
    reshade::api::resource_view view) {
  if (view.handle == 0u || !IsCandidateLut(device, resource, usage)) return;
  auto* data = device->get_private_data<DeviceData>();
  if (data == nullptr) return;
  const std::unique_lock lock(data->mutex);
  data->candidate_views[view.handle] = {.view = view, .resource = resource};
  data->resource_generations.try_emplace(resource.handle, 1u);

  DescriptorKey key = {};
  if (GetDescriptorKey(device, {view.handle}, 0u, 0u, key)) {
    data->sparse_descriptors[key] = {
        .type = reshade::api::descriptor_type::shader_resource_view,
        .view = view,
        .resource = resource,
    };
  }
}

inline void OnDestroyResourceView(
    reshade::api::device* device,
    reshade::api::resource_view view) {
  auto* data = device->get_private_data<DeviceData>();
  if (data == nullptr || view.handle == 0u) return;
  const std::unique_lock lock(data->mutex);
  const auto candidate = data->candidate_views.find(view.handle);
  if (candidate == data->candidate_views.end()) return;
  const uint64_t resource_handle = candidate->second.resource.handle;
  data->candidate_views.erase(candidate);
  std::erase_if(data->sparse_descriptors, [&](const auto& pair) {
    return pair.second.view.handle == view.handle;
  });
  const bool resource_still_used = std::ranges::any_of(
      data->candidate_views,
      [&](const auto& pair) {
        return pair.second.resource.handle == resource_handle;
      });
  if (!resource_still_used) data->resource_generations.erase(resource_handle);
}

inline bool OnUpdateDescriptorTables(
    reshade::api::device* device,
    uint32_t count,
    const reshade::api::descriptor_table_update* updates) {
  if (ignore_updates || count == 0u) return false;
  auto* data = device->get_private_data<DeviceData>();
  if (data == nullptr) return false;
  const std::unique_lock lock(data->mutex);

  for (uint32_t i = 0u; i < count; ++i) {
    const auto& update = updates[i];
    DescriptorKey first = {};
    if (!GetDescriptorKey(
            device,
            update.table,
            update.binding,
            update.array_offset,
            first)) {
      continue;
    }
    for (uint32_t k = 0u; k < update.count; ++k) {
      const DescriptorKey key = {.heap = first.heap, .index = first.index + k};
      data->sparse_descriptors.erase(key);

      if (update.type == reshade::api::descriptor_type::constant_buffer) {
        const auto range =
            static_cast<const reshade::api::buffer_range*>(update.descriptors)[k];
        if (range.buffer.handle != 0u
            && (range.size == UINT64_MAX
                || range.size >= MIN_TONE_MAP_CBV_SIZE)) {
          data->sparse_descriptors[key] = {
              .type = update.type,
              .buffer = range,
          };
        }
        continue;
      }

      reshade::api::resource_view view = {0u};
      if (update.type == reshade::api::descriptor_type::sampler_with_resource_view) {
        view = static_cast<const reshade::api::sampler_with_resource_view*>(
                   update.descriptors)[k]
                   .view;
      } else if (
          update.type == reshade::api::descriptor_type::shader_resource_view) {
        view = static_cast<const reshade::api::resource_view*>(
            update.descriptors)[k];
      } else {
        continue;
      }
      const auto candidate = data->candidate_views.find(view.handle);
      if (candidate == data->candidate_views.end()) continue;
      data->sparse_descriptors[key] = {
          .type = update.type,
          .view = view,
          .resource = candidate->second.resource,
      };
    }
  }
  return false;
}

inline bool OnCopyDescriptorTables(
    reshade::api::device* device,
    uint32_t count,
    const reshade::api::descriptor_table_copy* copies) {
  if (ignore_updates || count == 0u) return false;
  auto* data = device->get_private_data<DeviceData>();
  if (data == nullptr) return false;
  const std::unique_lock lock(data->mutex);

  for (uint32_t i = 0u; i < count; ++i) {
    const auto& copy = copies[i];
    DescriptorKey source = {};
    DescriptorKey destination = {};
    if (!GetDescriptorKey(
            device,
            copy.source_table,
            copy.source_binding,
            copy.source_array_offset,
            source)
        || !GetDescriptorKey(
            device,
            copy.dest_table,
            copy.dest_binding,
            copy.dest_array_offset,
            destination)) {
      continue;
    }
    for (uint32_t k = 0u; k < copy.count; ++k) {
      const DescriptorKey src = {.heap = source.heap, .index = source.index + k};
      const DescriptorKey dst = {
          .heap = destination.heap,
          .index = destination.index + k};
      data->sparse_descriptors.erase(dst);
      const auto entry = data->sparse_descriptors.find(src);
      if (entry == data->sparse_descriptors.end()) continue;
      data->sparse_descriptors[dst] = entry->second;
    }
  }
  return false;
}

inline uint64_t HashRouteValue(uint64_t hash, uint64_t value) {
  return (hash ^ value) * 1099511628211ull;
}

inline ResolvedLayoutRoute BuildRoute(reshade::api::pipeline_layout layout) {
  ResolvedLayoutRoute route = {};
  renodx::utils::pipeline_layout::GetPipelineLayoutData(
      layout,
      [&](const renodx::utils::pipeline_layout::PipelineLayoutData* layout_data) {
        for (uint32_t param_index = 0u;
             param_index < layout_data->params.size();
             ++param_index) {
          const auto& param = layout_data->params[param_index];
          uint32_t range_count = 0u;
          const reshade::api::descriptor_range* ranges = nullptr;
          const bool table_like =
              param.type
                  == reshade::api::pipeline_layout_param_type::descriptor_table
              || param.type
                     == reshade::api::pipeline_layout_param_type::descriptor_table_with_static_samplers
              || param.type
                     == reshade::api::pipeline_layout_param_type::push_descriptors_with_ranges
              || param.type
                     == reshade::api::pipeline_layout_param_type::push_descriptors_with_static_samplers;
          if (table_like
              && param_index < layout_data->ranges.size()
              && !layout_data->ranges[param_index].empty()) {
            range_count = static_cast<uint32_t>(
                layout_data->ranges[param_index].size());
            ranges = layout_data->ranges[param_index].data();
          } else if (
              param.type
              == reshade::api::pipeline_layout_param_type::descriptor_table) {
            range_count = param.descriptor_table.count;
            ranges = param.descriptor_table.ranges;
          } else if (
              param.type
              == reshade::api::pipeline_layout_param_type::descriptor_table_with_static_samplers) {
            range_count = param.descriptor_table_with_static_samplers.count;
            ranges = param.descriptor_table_with_static_samplers.ranges;
          } else {
            continue;
          }

          for (uint32_t i = 0u; i < range_count; ++i) {
            const auto& range = ranges[i];
            route.signature = HashRouteValue(
                route.signature,
                (static_cast<uint64_t>(param_index) << 48u)
                    ^ (static_cast<uint64_t>(range.type) << 40u)
                    ^ (static_cast<uint64_t>(range.dx_register_space) << 32u)
                    ^ (static_cast<uint64_t>(range.dx_register_index) << 16u)
                    ^ range.binding);
            if (range.dx_register_space != 0u
                || !renodx::utils::bitwise::HasFlag(
                    range.visibility,
                    reshade::api::shader_stage::compute)) {
              continue;
            }
            const auto covers = [&](uint32_t shader_register) {
              if (shader_register < range.dx_register_index) return false;
              const uint32_t offset =
                  shader_register - range.dx_register_index;
              return range.count == UINT32_MAX || offset < range.count;
            };
            const auto make_slot = [&](uint32_t shader_register) {
              return RouteSlot{
                  .valid = true,
                  .layout_param = param_index,
                  .binding = range.binding
                             + shader_register
                             - range.dx_register_index,
                  .type = range.type,
              };
            };
            if (range.type
                == reshade::api::descriptor_type::shader_resource_view) {
              if (covers(7u)) route.t7 = make_slot(7u);
              if (covers(8u)) route.t8 = make_slot(8u);
            } else if (
                range.type == reshade::api::descriptor_type::constant_buffer
                && covers(0u)) {
              route.b0 = make_slot(0u);
            }
          }
        }
      });
  return route;
}

inline bool ResolveToneMapBindings(
    reshade::api::command_list* cmd_list,
    CommandListData* command_data,
    ResolvedLayoutRoute* route_out = nullptr) {
  if (cmd_list == nullptr
      || command_data == nullptr
      || command_data->compute_layout.handle == 0u) {
    return false;
  }
  auto* device = cmd_list->get_device();
  auto* device_data = device->get_private_data<DeviceData>();
  if (device_data == nullptr) return false;

  ResolvedLayoutRoute route = {};
  {
    const std::shared_lock lock(device_data->mutex);
    const auto cached = device_data->routes.find(
        command_data->compute_layout.handle);
    if (cached != device_data->routes.end()) route = cached->second;
  }
  if (!route.t7.valid && !route.b0.valid) {
    route = BuildRoute(command_data->compute_layout);
    const std::unique_lock lock(device_data->mutex);
    device_data->routes[command_data->compute_layout.handle] = route;
  }
  if (route_out != nullptr) *route_out = route;

  const auto get_entry = [&](const RouteSlot& slot) -> SparseDescriptor {
    if (!slot.valid || slot.layout_param >= command_data->compute_tables.size()) {
      return {};
    }
    const auto table = command_data->compute_tables[slot.layout_param];
    DescriptorKey key = {};
    if (!GetDescriptorKey(device, table, slot.binding, 0u, key)) return {};
    const std::shared_lock lock(device_data->mutex);
    const auto entry = device_data->sparse_descriptors.find(key);
    return entry == device_data->sparse_descriptors.end()
               ? SparseDescriptor{}
               : entry->second;
  };

  if (route.t7.valid) {
    const auto entry = get_entry(route.t7);
    if (entry.view.handle != 0u) {
      const std::shared_lock lock(device_data->mutex);
      command_data->compute_t7 = {
          .valid = true,
          .stages = reshade::api::shader_stage::all_compute,
          .layout = command_data->compute_layout,
          .layout_param = route.t7.layout_param,
          .binding = route.t7.binding,
          .type = route.t7.type,
          .view = entry.view,
          .resource = entry.resource,
          .generation = GetGeneration(device_data, entry.resource),
      };
    }
  }
  if (route.t8.valid) {
    const auto entry = get_entry(route.t8);
    if (entry.view.handle != 0u) {
      const std::shared_lock lock(device_data->mutex);
      command_data->compute_t8 = {
          .valid = true,
          .stages = reshade::api::shader_stage::all_compute,
          .layout = command_data->compute_layout,
          .layout_param = route.t8.layout_param,
          .binding = route.t8.binding,
          .type = route.t8.type,
          .view = entry.view,
          .resource = entry.resource,
          .generation = GetGeneration(device_data, entry.resource),
      };
    }
  }
  if (route.b0.valid) {
    const auto entry = get_entry(route.b0);
    if (entry.buffer.buffer.handle != 0u) {
      command_data->compute_b0 = {
          .valid = true,
          .stages = reshade::api::shader_stage::all_compute,
          .layout = command_data->compute_layout,
          .layout_param = route.b0.layout_param,
          .binding = route.b0.binding,
          .range = entry.buffer,
      };
    }
  }
  return command_data->compute_t7.valid
         && command_data->compute_b0.valid;
}

inline void OnDestroyPipelineLayout(
    reshade::api::device* device,
    reshade::api::pipeline_layout layout) {
  auto* data = device->get_private_data<DeviceData>();
  if (data == nullptr || layout.handle == 0u) return;
  const std::unique_lock lock(data->mutex);
  data->routes.erase(layout.handle);
}

inline bool OnUpdateTextureRegion(
    reshade::api::device* device,
    const reshade::api::subresource_data&,
    reshade::api::resource dest,
    uint32_t,
    const reshade::api::subresource_box*) {
  MarkResourceWritten(device, dest);
  return false;
}

inline bool OnCopyResource(
    reshade::api::command_list* cmd_list,
    reshade::api::resource,
    reshade::api::resource dest) {
  MarkResourceWritten(cmd_list->get_device(), dest);
  return false;
}

inline bool OnCopyBufferToTexture(
    reshade::api::command_list* cmd_list,
    reshade::api::resource,
    uint64_t,
    uint32_t,
    uint32_t,
    reshade::api::resource dest,
    uint32_t,
    const reshade::api::subresource_box*) {
  MarkResourceWritten(cmd_list->get_device(), dest);
  return false;
}

inline bool OnCopyTextureRegion(
    reshade::api::command_list* cmd_list,
    reshade::api::resource,
    uint32_t,
    const reshade::api::subresource_box*,
    reshade::api::resource dest,
    uint32_t,
    const reshade::api::subresource_box*,
    reshade::api::filter_mode) {
  MarkResourceWritten(cmd_list->get_device(), dest);
  return false;
}

}  // namespace tlou::descriptor_tracker
