#ifndef SRC_GAMES_THELASTOFUSPART1_SHARED_H_
#define SRC_GAMES_THELASTOFUSPART1_SHARED_H_

#ifdef __cplusplus
struct alignas(16) ShaderInjectData {
#else
struct ShaderInjectData {
#endif
  float peak_white_nits;
  float diffuse_white_nits;
  float graphics_white_nits;
  float tone_map_type;
  float cone_response;
  float exposure_match;
  float vanilla_slope_amount;
  float lut_scale;
  float lut_offset;
  float lut_compression;
  float post_lut_amount;
  float native_curve_enabled;
  float native_curve_a;
  float native_curve_b;
  float native_curve_c;
  float native_curve_d;
  float native_curve_e;
};

#ifdef __cplusplus
static_assert(sizeof(ShaderInjectData) == 80);
static_assert(alignof(ShaderInjectData) == 16);
#endif

#define TLOU_TONE_MAP_TYPE_VANILLA   0.f
#define TLOU_TONE_MAP_TYPE_PSYCHOV17 1.f
#define TLOU_TONE_MAP_TYPE_PSYCHOV22 2.f
#define TLOU_TONE_MAP_TYPE_PSYCHOV24 3.f
#define TLOU_TONE_MAP_TYPE_PSYCHOV25 4.f
#define TLOU_TONE_MAP_TYPE_PSYCHOV30 5.f
#define TLOU_TONE_MAP_TYPE_RENODRT   6.f
#define TLOU_TONE_MAP_TYPE_NEUTWO    7.f

#ifndef __cplusplus
cbuffer shader_injection : register(b13, space50) {
  ShaderInjectData shader_injection : packoffset(c0);
}

#define RENODX_PEAK_WHITE_NITS     shader_injection.peak_white_nits
#define RENODX_DIFFUSE_WHITE_NITS  shader_injection.diffuse_white_nits
#define RENODX_GRAPHICS_WHITE_NITS shader_injection.graphics_white_nits
#define RENODX_TONE_MAP_TYPE       shader_injection.tone_map_type
#define TLOU_CONE_RESPONSE         shader_injection.cone_response
#define TLOU_EXPOSURE_MATCH        shader_injection.exposure_match
#define TLOU_VANILLA_SLOPE_AMOUNT  shader_injection.vanilla_slope_amount
#define TLOU_LUT_SCALE             shader_injection.lut_scale
#define TLOU_LUT_OFFSET            shader_injection.lut_offset
#define TLOU_LUT_COMPRESSION       shader_injection.lut_compression
#define TLOU_POST_LUT_AMOUNT       shader_injection.post_lut_amount
#define TLOU_NATIVE_CURVE_ENABLED  shader_injection.native_curve_enabled
#define TLOU_NATIVE_CURVE_A        shader_injection.native_curve_a
#define TLOU_NATIVE_CURVE_B        shader_injection.native_curve_b
#define TLOU_NATIVE_CURVE_C        shader_injection.native_curve_c
#define TLOU_NATIVE_CURVE_D        shader_injection.native_curve_d
#define TLOU_NATIVE_CURVE_E        shader_injection.native_curve_e

#include "../../shaders/renodx.hlsl"
#endif

#endif  // SRC_GAMES_THELASTOFUSPART1_SHARED_H_
