struct PrimitiveVertex {
  float4 position;
  float2 texcoord;
  float2 texcoord_1;
  uint color;
  uint color_1;
};

StructuredBuffer<PrimitiveVertex> t0_space2 : register(t0, space2);

cbuffer cb0 : register(b0) {
  struct Vs2dGlobalParams {
    float4 Vs2dGlobalParams_000;
    float4 RenoDXParams_016;
  } g_gui2PassThroughConstants_000 : packoffset(c000.x);
};

struct OutputSignature {
  noperspective float4 SV_Position : SV_Position;
  linear float2 TEXCOORD : TEXCOORD;
  linear float2 TEXCOORD_1 : TEXCOORD1;
  linear float4 COLOR : COLOR;
  linear float4 COLOR_1 : COLOR1;
};

static const uint RENODX_UI_MAGIC = 0x52445855u;

OutputSignature main(
    uint SV_VertexID : SV_VertexID) {
  float4 SV_Position;
  float2 TEXCOORD;
  float2 TEXCOORD_1;
  float4 COLOR;
  float4 COLOR_1;
  PrimitiveVertex _4 = t0_space2.Load(SV_VertexID);
  PrimitiveVertex _8 = t0_space2.Load(SV_VertexID);
  PrimitiveVertex _11 = t0_space2.Load(SV_VertexID);
  PrimitiveVertex _14 = t0_space2.Load(SV_VertexID);
  PrimitiveVertex _16 = t0_space2.Load(SV_VertexID);
  int _18 = _14.color & 255;
  float _19 = float((uint)_18);
  int _20 = (uint)((int)(_14.color)) >> 8;
  int _21 = _20 & 255;
  float _22 = float((uint)_21);
  int _23 = (uint)((int)(_14.color)) >> 16;
  int _24 = _23 & 255;
  float _25 = float((uint)_24);
  int _26 = (uint)((int)(_14.color)) >> 24;
  float _27 = float((uint)_26);
  float _28 = _19 * 0.003921568859368563f;
  float _29 = _22 * 0.003921568859368563f;
  float _30 = _25 * 0.003921568859368563f;
  float _31 = _27 * 0.003921568859368563f;
  int _32 = _16.color_1 & 255;
  float _33 = float((uint)_32);
  int _34 = (uint)((int)(_16.color_1)) >> 8;
  int _35 = _34 & 255;
  float _36 = float((uint)_35);
  int _37 = (uint)((int)(_16.color_1)) >> 16;
  int _38 = _37 & 255;
  float _39 = float((uint)_38);
  int _40 = (uint)((int)(_16.color_1)) >> 24;
  float _41 = float((uint)_40);
  float _42 = _33 * 0.003921568859368563f;
  float _43 = _36 * 0.003921568859368563f;
  float _44 = _39 * 0.003921568859368563f;
  float _45 = _41 * 0.003921568859368563f;
  float _51 = g_gui2PassThroughConstants_000.Vs2dGlobalParams_000.x * _4.position.x;
  float _52 = g_gui2PassThroughConstants_000.Vs2dGlobalParams_000.y * _4.position.y;
  float _53 = _51 + g_gui2PassThroughConstants_000.Vs2dGlobalParams_000.z;
  float _54 = _52 + g_gui2PassThroughConstants_000.Vs2dGlobalParams_000.w;
  SV_Position.x = _53;
  SV_Position.y = _54;
  SV_Position.z = _4.position.z;
  SV_Position.w = 1.0f;
  TEXCOORD.x = _8.texcoord.x;
  TEXCOORD.y = _8.texcoord.y;
  TEXCOORD_1.x = _11.texcoord_1.x;
  TEXCOORD_1.y = _11.texcoord_1.y;
  COLOR.x = _28;
  COLOR.y = _29;
  COLOR.z = _30;
  COLOR.w = _31;
  COLOR_1.x = _42;
  COLOR_1.y = _43;
  COLOR_1.z = _44;
  COLOR_1.w = _45;
  float renodx_ui_scale =
      asuint(g_gui2PassThroughConstants_000.RenoDXParams_016.x) == RENODX_UI_MAGIC
          ? max(g_gui2PassThroughConstants_000.RenoDXParams_016.y, 0.f)
          : 1.f;
  COLOR.rgb *= renodx_ui_scale;
  COLOR_1.rgb *= renodx_ui_scale;
  OutputSignature output_signature = {SV_Position, TEXCOORD, TEXCOORD_1, COLOR, COLOR_1};
  return output_signature;
}
