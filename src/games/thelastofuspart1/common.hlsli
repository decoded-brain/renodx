#ifndef SRC_GAMES_THELASTOFUSPART1_COMMON_HLSLI_
#define SRC_GAMES_THELASTOFUSPART1_COMMON_HLSLI_

#include "./shared.h"

#ifndef TLOU_COMPILED_TONE_MAP_TYPE
#define TLOU_COMPILED_TONE_MAP_TYPE 1
#endif

#if TLOU_COMPILED_TONE_MAP_TYPE == 3
#include "./psychov_test24.hlsli"
#elif TLOU_COMPILED_TONE_MAP_TYPE == 4
#include "./psychov_test25.hlsli"
#elif TLOU_COMPILED_TONE_MAP_TYPE == 5
#include "./psychov_test30.hlsli"
#endif

static const float TLOU_FIXED_REFERENCE_NITS = 203.f;
static const float TLOU_NATIVE_ENCODER_REFERENCE_NITS = 300.f;
static const float TLOU_REFERENCE_GRAY = 0.18f;

struct TLOUToneMapBridgeState {
  float enabled;
  float3 lut_target_native_bt709;
  float3 adaptive_state_lms;
  float max_channel_scale;
  float gamut_compression_scale;
};

float TLOUNativeCurve(float x) {
  x = max(x, 0.f);
  if (TLOU_NATIVE_CURVE_ENABLED < 0.5f) return x;
  float denominator = x * x + TLOU_NATIVE_CURVE_A * x + TLOU_NATIVE_CURVE_B;
  return (TLOU_NATIVE_CURVE_D * x + TLOU_NATIVE_CURVE_E) / denominator + TLOU_NATIVE_CURVE_C;
}

float TLOUNativeCurveDerivative(float x) {
  if (TLOU_NATIVE_CURVE_ENABLED < 0.5f) return 1.f;
  float denominator = x * x + TLOU_NATIVE_CURVE_A * x + TLOU_NATIVE_CURVE_B;
  float numerator = TLOU_NATIVE_CURVE_D * x + TLOU_NATIVE_CURVE_E;
  return (TLOU_NATIVE_CURVE_D * denominator
          - numerator * (2.f * x + TLOU_NATIVE_CURVE_A))
         / (denominator * denominator);
}

float TLOULutInputMax() {
  float compression = max(TLOU_LUT_COMPRESSION, 1e-6f);
  return pow(2.f, 4.f / 3.f) * pow(compression, 1.f / 3.f);
}

float TLOULutOutputCeiling() {
  return clamp(TLOUNativeCurve(TLOULutInputMax()), 1e-4f, 1.f - 1e-4f);
}

float TLOUNativeCurveInverse(float y) {
  if (TLOU_NATIVE_CURVE_ENABLED < 0.5f) return max(y, 0.f);
  float minimum = max(TLOUNativeCurve(0.f), 0.f);
  y = clamp(y, minimum, TLOULutOutputCeiling() - 1e-6f);
  float k = y - TLOU_NATIVE_CURVE_C;
  float quadratic_b = k * TLOU_NATIVE_CURVE_A - TLOU_NATIVE_CURVE_D;
  float quadratic_c = k * TLOU_NATIVE_CURVE_B - TLOU_NATIVE_CURVE_E;
  float discriminant = max(0.f, quadratic_b * quadratic_b - 4.f * k * quadratic_c);
  float denominator = 2.f * k;
  float root_a = renodx::math::DivideSafe(-quadratic_b + sqrt(discriminant), denominator, 0.f);
  float root_b = renodx::math::DivideSafe(-quadratic_b - sqrt(discriminant), denominator, 0.f);
  if (root_a >= 0.f && root_b >= 0.f) return min(root_a, root_b);
  return max(root_a, root_b);
}

float3 TLOUSanitize(float3 color) {
  return all(isfinite(color)) ? max(color, 0.f) : 0.f.xxx;
}

float3 TLOULimitPeak(float3 color, float peak_value) {
  color = TLOUSanitize(color);
  peak_value = max(peak_value, 1e-6f);
  float max_channel = renodx::math::Max(color);
  if (max_channel > peak_value) color *= peak_value / max_channel;
  return color;
}

TLOUToneMapBridgeState TLOUCreateToneMapBridgeState() {
  TLOUToneMapBridgeState state;
  state.enabled = 0.f;
  state.lut_target_native_bt709 = 0.f.xxx;
  float native_gray = TLOU_REFERENCE_GRAY
                      * RENODX_DIFFUSE_WHITE_NITS
                      / TLOU_NATIVE_ENCODER_REFERENCE_NITS;
  state.adaptive_state_lms = renodx::color::lms::from::BT709(native_gray.xxx);
  state.max_channel_scale = 1.f;
  state.gamut_compression_scale = 1.f;
  return state;
}

bool TLOUIsPsychoV() {
#if TLOU_COMPILED_TONE_MAP_TYPE >= 1 && TLOU_COMPILED_TONE_MAP_TYPE <= 5
  return true;
#else
  return false;
#endif
}

#if TLOU_COMPILED_TONE_MAP_TYPE == 6
float3 TLOUMapRenoDRT(
    float3 exposed_scene_bt709,
    float peak_value,
    float reference_input,
    float reference_output) {
  renodx::tonemap::renodrt::Config config = renodx::tonemap::renodrt::config::Create();
  config.nits_peak = peak_value * 100.f;
  config.mid_gray_value = reference_input;
  config.mid_gray_nits = reference_output * 100.f;
  config.exposure = 1.f;
  config.highlights = 1.f;
  config.shadows = 1.f;
  config.contrast = 1.f;
  config.saturation = 1.f;
  config.dechroma = 0.f;
  config.flare = 0.f;
  config.hue_correction_strength = 0.f;
  config.tone_map_method = renodx::tonemap::renodrt::config::tone_map_method::DANIELE;
  config.working_color_space = 0.f;
  config.clamp_color_space = -1.f;
  config.clamp_peak = -1.f;
  config.white_clip = 100.f;
  config.scaling_method = renodx::tonemap::renodrt::config::scaling_method::MAX_CHANNEL;
  return renodx::tonemap::renodrt::BT709(max(exposed_scene_bt709, 0.f), config);
}
#endif

#if TLOU_COMPILED_TONE_MAP_TYPE == 7
float3 TLOUMapNeutwo(
    float3 exposed_scene_bt709,
    float peak_value,
    float reference_input,
    float reference_output) {
  float3 color = max(exposed_scene_bt709, 0.f);
  float max_channel = renodx::math::Max(color);
  if (max_channel <= 0.f) return 0.f.xxx;
  float mapped_max = renodx::tonemap::Neutwo(
      max_channel,
      peak_value,
      100.f,
      reference_input,
      reference_output);
  return color * mapped_max / max_channel;
}
#endif

float3 TLOUMatchVanillaDiffuse(
    float3 exposed_scene_bt709,
    float3 tone_mapped_bt709,
    float reference_input,
    float reference_output) {
  float scene_max = renodx::math::Max(max(exposed_scene_bt709, 0.f));
  float mapped_max = renodx::math::Max(max(tone_mapped_bt709, 0.f));
  if (scene_max <= 0.f || mapped_max <= 0.f) return max(tone_mapped_bt709, 0.f);

  float native_reference = max(TLOUNativeCurve(reference_input), 1e-6f);
  float native_max = TLOUNativeCurve(scene_max)
                     * reference_output
                     / native_reference;
  float mapper_weight = smoothstep(reference_input * 2.f, reference_input * 4.f, scene_max);
  float target_max = lerp(native_max, mapped_max, mapper_weight);
  return max(tone_mapped_bt709, 0.f) * target_max / mapped_max;
}

float3 TLOUPrepareToneMapLut(float3 exposed_scene_bt709, out TLOUToneMapBridgeState state) {
  state = TLOUCreateToneMapBridgeState();
  state.enabled = 1.f;

  float reference_input = TLOU_REFERENCE_GRAY;
  float reference_output = TLOU_REFERENCE_GRAY;
  if (!TLOUIsPsychoV() || TLOU_EXPOSURE_MATCH > 0.5f) {
    float native_reference_output = TLOU_REFERENCE_GRAY
                                    * TLOU_FIXED_REFERENCE_NITS
                                    / TLOU_NATIVE_ENCODER_REFERENCE_NITS;
    reference_input = TLOUNativeCurveInverse(native_reference_output);
    reference_output = TLOUNativeCurve(reference_input)
                       * TLOU_NATIVE_ENCODER_REFERENCE_NITS
                       / TLOU_FIXED_REFERENCE_NITS;
  }

  float native_log_slope = reference_input * TLOUNativeCurveDerivative(reference_input)
                           / max(TLOUNativeCurve(reference_input), 1e-6f);
  float cone_response = TLOU_CONE_RESPONSE
                        * lerp(1.f, native_log_slope, saturate(TLOU_VANILLA_SLOPE_AMOUNT));
  float peak_value = max(RENODX_PEAK_WHITE_NITS / max(RENODX_DIFFUSE_WHITE_NITS, 1e-6f), 1e-6f);

  float3 tone_mapped;
#if TLOU_COMPILED_TONE_MAP_TYPE == 2
  tone_mapped = renodx::tonemap::psychov::psychotm_test22(
        exposed_scene_bt709,
        peak_value,
        1.f, 1.f, 1.f, 1.f,
        1.f, 1.f, 100.f, 1.f, 1.f, 0,
        cone_response,
        reference_input.xxx,
        reference_output.xxx,
        1.f, 1, 1.f, 1.f);
#elif TLOU_COMPILED_TONE_MAP_TYPE == 3
  tone_mapped = renodx::tonemap::psychov::psychotm_test24(
        exposed_scene_bt709,
        peak_value,
        1.f, 1.f, 1.f, 1.f,
        1.f, 1.f, 100.f, 1.f, 1.f, 0,
        cone_response,
        reference_input.xxx,
        reference_output.xxx,
        1.f, 1, 1.f, 1.f, 1.f, 0.f);
#elif TLOU_COMPILED_TONE_MAP_TYPE == 4
  tone_mapped = renodx::tonemap::psychov::psychotm_test25(
        exposed_scene_bt709,
        peak_value,
        1.f, 1.f, 1.f, 1.f,
        1.f, 1.f, 100.f, 1.f, 1.f, 0,
        cone_response,
        reference_input.xxx,
        reference_output.xxx,
        1.f, 1, 1.f);
#elif TLOU_COMPILED_TONE_MAP_TYPE == 5
  tone_mapped = renodx::tonemap::psychov::psychotm_test30(
        exposed_scene_bt709,
        peak_value,
        1.f, 1.f, 1.f, 1.f,
        1.f, 1.f, 100.f, 1.f, 1.f, 0,
        cone_response,
        reference_input.xxx,
        reference_output.xxx,
        1.f, 1, 1.f, 1.f);
#elif TLOU_COMPILED_TONE_MAP_TYPE == 6
  tone_mapped = TLOUMapRenoDRT(
        exposed_scene_bt709,
        peak_value,
        reference_input,
        reference_output);
#elif TLOU_COMPILED_TONE_MAP_TYPE == 7
  tone_mapped = TLOUMapNeutwo(
        exposed_scene_bt709,
        peak_value,
        reference_input,
        reference_output);
#else
  tone_mapped = renodx::tonemap::psychov::psychotm_test17(
        exposed_scene_bt709,
        peak_value,
        1.f, 1.f, 1.f, 1.f,
        1.f, 1.f, 100.f, 1.f, 1.f, 0,
        cone_response,
        reference_input.xxx,
        reference_output.xxx,
        1.f, 1, 1.f);
#endif

  tone_mapped = TLOUMatchVanillaDiffuse(
      exposed_scene_bt709,
      tone_mapped,
      reference_input,
      reference_output);
  tone_mapped = TLOULimitPeak(tone_mapped, peak_value);
  float native_scale = RENODX_DIFFUSE_WHITE_NITS / TLOU_NATIVE_ENCODER_REFERENCE_NITS;
  float3 tone_mapped_native = tone_mapped * native_scale;
  state.gamut_compression_scale = renodx::color::gamut::ComputeGamutCompressionScaleBT709AdaptiveD65(
      tone_mapped_native,
      state.adaptive_state_lms,
      1.f);
  float3 compressed = renodx::color::gamut::GamutCompressBT709AdaptiveD65(
      tone_mapped_native,
      state.adaptive_state_lms,
      state.gamut_compression_scale);
  float max_channel = renodx::math::Max(compressed);
  float lut_output_ceiling = TLOULutOutputCeiling();
  state.max_channel_scale = max_channel > lut_output_ceiling
                                ? renodx::tonemap::neutwo::ComputeMaxChannelScale(
                                      compressed,
                                      lut_output_ceiling)
                                : 1.f;
  state.lut_target_native_bt709 = compressed * state.max_channel_scale;

  // t7 bakes the native rational curve together with grading. Remove only that
  // curve, in the native 300-nit domain, before sampling the original LUT.
  return float3(
      TLOUNativeCurveInverse(state.lut_target_native_bt709.r),
      TLOUNativeCurveInverse(state.lut_target_native_bt709.g),
      TLOUNativeCurveInverse(state.lut_target_native_bt709.b));
}

float3 TLOUFinalizeToneMap(float3 graded_srgb, TLOUToneMapBridgeState state) {
  if (state.enabled < 0.5f) return graded_srgb;

  float3 graded_sdr = renodx::color::srgb::DecodeSafe(graded_srgb);
  float3 graded_compressed = renodx::math::DivideSafe(
      graded_sdr,
      state.max_channel_scale.xxx,
      graded_sdr);
  float3 reconstructed = renodx::color::gamut::GamutDecompressBT709AdaptiveD65(
      graded_compressed,
      state.adaptive_state_lms,
      state.gamut_compression_scale);
  float native_peak = RENODX_PEAK_WHITE_NITS / TLOU_NATIVE_ENCODER_REFERENCE_NITS;
  reconstructed = TLOULimitPeak(reconstructed, native_peak);

  // Preserve the game's sRGB-coded intermediate. The final native shader owns
  // its runtime gamma exponent and PQ/BT.2020 encoding.
  return renodx::color::srgb::EncodeSafe(TLOUSanitize(reconstructed));
}

#endif  // SRC_GAMES_THELASTOFUSPART1_COMMON_HLSLI_
