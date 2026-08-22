#include "./common.hlsli"

Texture3D<float4> t0 : register(t0);
Texture3D<float4> t1 : register(t1);
RWTexture3D<float4> u0 : register(u0);
SamplerState s0 : register(s0);

static const uint TLOU_LUT_SIZE = 32u;

float3 TLOUDecodeLutInput(uint3 id) {
  float3 texel_center = (float3(id) + 0.5f) / float(TLOU_LUT_SIZE);
  float scale = max(TLOU_LUT_SCALE, 1e-6f);
  float3 compressed = saturate((texel_center - TLOU_LUT_OFFSET) / scale);
  float compression = max(TLOU_LUT_COMPRESSION, 1e-6f);
  float threshold = 0.5f / compression;
  float3 low_srgb = compressed * 2.f;
  float3 high_srgb = pow(max(compressed * 2.f, 0.f), 4.f / 3.f)
                     * pow(compression, 1.f / 3.f);
  float3 srgb = float3(
      compressed.r <= threshold ? low_srgb.r : high_srgb.r,
      compressed.g <= threshold ? low_srgb.g : high_srgb.g,
      compressed.b <= threshold ? low_srgb.b : high_srgb.b);
  return renodx::color::srgb::DecodeSafe(srgb);
}

float3 TLOUEncodeLutCoordinate(float3 color) {
  float compression = max(TLOU_LUT_COMPRESSION, 1e-6f);
  float3 srgb = renodx::color::srgb::EncodeSafe(max(color, 0.f));
  float3 scaled = srgb * compression;
  float3 compressed = (saturate(scaled)
                       + max(pow(max(scaled, 0.f), 0.75f) - 1.f, 0.f))
                      * (0.5f / compression);
  return saturate(compressed) * TLOU_LUT_SCALE + TLOU_LUT_OFFSET;
}

[numthreads(4, 4, 4)]
void main(uint3 id : SV_DispatchThreadID) {
  if (any(id >= TLOU_LUT_SIZE)) return;

  float3 exposed_scene = TLOUDecodeLutInput(id);
  TLOUToneMapBridgeState bridge;
  float3 native_lut_input = TLOUPrepareToneMapLut(exposed_scene, bridge);
  float3 graded_srgb = t0.SampleLevel(
      s0,
      TLOUEncodeLutCoordinate(native_lut_input),
      0.f).rgb;

  if (TLOU_POST_LUT_AMOUNT > 0.f) {
    float3 post_lut_coordinate = saturate(graded_srgb)
                                 * TLOU_LUT_SCALE
                                 + TLOU_LUT_OFFSET;
    float3 post_graded_srgb = t1.SampleLevel(s0, post_lut_coordinate, 0.f).rgb;
    graded_srgb = lerp(
        graded_srgb,
        post_graded_srgb,
        saturate(TLOU_POST_LUT_AMOUNT));
  }

  u0[id] = float4(TLOUFinalizeToneMap(graded_srgb, bridge), 1.f);
}
