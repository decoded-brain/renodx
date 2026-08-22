#!/usr/bin/env python3
"""Focused numerical contract for the TLOU Part I HDR tone-map glue."""

from __future__ import annotations

from decimal import Decimal, getcontext
from math import isfinite
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
COMMON_PATH = ROOT / "src/games/thelastofuspart1/common.hlsli"
ADDON_PATH = ROOT / "src/games/thelastofuspart1/addon.cpp"
SHARED_PATH = ROOT / "src/games/thelastofuspart1/shared.h"
LUT_PATH = ROOT / "src/games/thelastofuspart1/lut_impl.hlsli"
GUI_PATH = ROOT / "src/games/thelastofuspart1/gui_0xECBF2D51.vs_6_0.hlsl"
TRACKER_PATH = ROOT / "src/games/thelastofuspart1/descriptor_tracker.hpp"
PSYCHOV_PATHS = {
    24: ROOT / "src/games/thelastofuspart1/psychov_test24.hlsli",
    25: ROOT / "src/games/thelastofuspart1/psychov_test25.hlsli",
    30: ROOT / "src/games/thelastofuspart1/psychov_test30.hlsli",
}
VARIANT_PATHS = {
    1: ROOT / "src/games/thelastofuspart1/lut_psychov17.cs_6_0.hlsl",
    2: ROOT / "src/games/thelastofuspart1/lut_psychov22.cs_6_0.hlsl",
    3: ROOT / "src/games/thelastofuspart1/lut_psychov24.cs_6_0.hlsl",
    4: ROOT / "src/games/thelastofuspart1/lut_psychov25.cs_6_0.hlsl",
    5: ROOT / "src/games/thelastofuspart1/lut_psychov30.cs_6_0.hlsl",
    6: ROOT / "src/games/thelastofuspart1/lut_renodrt.cs_6_0.hlsl",
    7: ROOT / "src/games/thelastofuspart1/lut_neutwo.cs_6_0.hlsl",
}

COMMON = COMMON_PATH.read_text(encoding="utf-8")
ADDON = ADDON_PATH.read_text(encoding="utf-8")
SHARED = SHARED_PATH.read_text(encoding="utf-8")
LUT = LUT_PATH.read_text(encoding="utf-8")
GUI = GUI_PATH.read_text(encoding="utf-8")
TRACKER = TRACKER_PATH.read_text(encoding="utf-8")


def constant(name: str) -> float:
    match = re.search(rf"static const float {name} = ([+-]?[0-9.]+)f;", COMMON)
    assert match is not None, f"missing {name}"
    return float(match.group(1))


A = constant("TLOU_NATIVE_CURVE_A")
B = constant("TLOU_NATIVE_CURVE_B")
C = constant("TLOU_NATIVE_CURVE_C")
D = constant("TLOU_NATIVE_CURVE_D")
E = constant("TLOU_NATIVE_CURVE_E")
FIXED_REFERENCE_NITS = constant("TLOU_FIXED_REFERENCE_NITS")
ENCODER_REFERENCE_NITS = constant("TLOU_NATIVE_ENCODER_REFERENCE_NITS")
REFERENCE_GRAY = constant("TLOU_REFERENCE_GRAY")

# HDR mode 3 and SDR mode 0 captures used the same neutral coefficients.
CAPTURED_CURVES = {
    "hdr_mode_3": (A, B, C, D, E),
    "sdr_mode_0": (A, B, C, D, E),
}


def native_curve(x: float, params: tuple[float, ...] = CAPTURED_CURVES["hdr_mode_3"]) -> float:
    a, b, c, d, e = params
    return (d * x + e) / (x * x + a * x + b) + c


def native_derivative(x: float, params: tuple[float, ...] = CAPTURED_CURVES["hdr_mode_3"]) -> float:
    a, b, _, d, e = params
    denominator = x * x + a * x + b
    numerator = d * x + e
    return (d * denominator - numerator * (2 * x + a)) / (denominator * denominator)


def native_inverse(y: float, params: tuple[float, ...] = CAPTURED_CURVES["hdr_mode_3"]) -> float:
    a, b, c, d, e = params
    y = min(max(y, 0.0), c - 1e-6)
    k = y - c
    quadratic_b = k * a - d
    quadratic_c = k * b - e
    discriminant = max(0.0, quadratic_b * quadratic_b - 4.0 * k * quadratic_c)
    root = discriminant**0.5
    roots = ((-quadratic_b + root) / (2.0 * k), (-quadratic_b - root) / (2.0 * k))
    nonnegative = [value for value in roots if value >= 0.0]
    assert nonnegative
    return min(nonnegative)


def decimal_curve(x: Decimal, params: tuple[float, ...]) -> Decimal:
    a, b, c, d, e = map(lambda value: Decimal(str(value)), params)
    return (d * x + e) / (x * x + a * x + b) + c


def numerical_inverse(y: float, params: tuple[float, ...]) -> float:
    target = Decimal(str(y))
    lo = Decimal(0)
    hi = Decimal(1)
    while decimal_curve(hi, params) < target:
        hi *= 2
    for _ in range(240):
        mid = (lo + hi) / 2
        if decimal_curve(mid, params) < target:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2)


def numerical_derivative(x: float, params: tuple[float, ...]) -> float:
    point = Decimal(str(x))
    step = Decimal("1e-22") * (Decimal(1) + point)
    return float((decimal_curve(point + step, params) - decimal_curve(point - step, params)) / (2 * step))


def limit_peak(rgb: tuple[float, float, float], peak: float) -> tuple[float, float, float]:
    maximum = max(rgb)
    scale = peak / maximum if maximum > peak else 1.0
    return tuple(max(0.0, value) * scale for value in rgb)


def native_handoff(value: float, game_nits: float) -> float:
    return value * game_nits / ENCODER_REFERENCE_NITS


def compress_lut_srgb(value: float, compression: float) -> float:
    scaled = max(value, 0.0) * compression
    encoded = (min(scaled, 1.0) + max(scaled**0.75 - 1.0, 0.0)) * 0.5 / compression
    return min(max(encoded, 0.0), 1.0)


def decompress_lut_srgb(value: float, compression: float) -> float:
    value = min(max(value, 0.0), 1.0)
    if value <= 0.5 / compression:
        return value * 2.0
    return (value * 2.0) ** (4.0 / 3.0) * compression ** (1.0 / 3.0)


def neutwo_anchor(x: float, peak: float, clip: float, gray_in: float, gray_out: float) -> float:
    cc = clip * clip
    pp = peak * peak
    gg = gray_in * gray_in
    oo = gray_out * gray_out
    xx = x * x
    cc_minus_gg = cc - gg
    numerator = peak * gray_out * x * cc_minus_gg
    denominator_squared = cc_minus_gg * (xx * (cc * oo - pp * gg) + cc * gg * (pp - oo))
    return numerator / denominator_squared**0.5


def test_curve_inverse_and_derivative() -> None:
    for params in CAPTURED_CURVES.values():
        for x in (1e-6, 1e-4, 0.001, 0.01, 0.05, 0.13243948, 0.18, 0.5, 1.0, 4.0, 16.0, 64.0):
            y = native_curve(x, params)
            assert abs(native_inverse(y, params) - numerical_inverse(y, params)) <= 2e-6 * max(1.0, x)
            reference = numerical_derivative(x, params)
            assert abs(native_derivative(x, params) - reference) <= 2e-6 * max(1.0, abs(reference))


def test_reference_anchor_and_log_slope() -> None:
    target_native = REFERENCE_GRAY * FIXED_REFERENCE_NITS / ENCODER_REFERENCE_NITS
    reference_input = native_inverse(target_native)
    reference_output = native_curve(reference_input) * ENCODER_REFERENCE_NITS / FIXED_REFERENCE_NITS
    log_slope = reference_input * native_derivative(reference_input) / native_curve(reference_input)
    assert abs(reference_input - 0.13243948) < 2e-7
    assert abs(reference_output - REFERENCE_GRAY) < 2e-7
    assert abs(log_slope - 0.9406792) < 2e-6
    assert isfinite(native_inverse(1.0)) and native_inverse(1.0) >= 0.0


def test_mode_neutral_glue_is_finite_and_monotonic() -> None:
    # Shared PsychoV owns each internal neutral curve. This test starts at that
    # interface with two distinct finite monotonic mode outputs and verifies the
    # game-local limiter and 300-nit handoff cannot break monotonicity.
    peak = 1000.0 / 203.0
    inputs = [index / 64.0 * 32.0 for index in range(65)]
    upstream = {
        "psychov17": [peak * (1.0 - 1.0 / (1.0 + value / peak)) for value in inputs],
        "psychov22": [peak * value / (value + peak) for value in inputs],
    }
    for function in (
        "psychotm_test17(",
        "psychotm_test22(",
        "psychotm_test24(",
        "psychotm_test25(",
        "psychotm_test30(",
        "TLOUMapRenoDRT(",
        "TLOUMapNeutwo(",
    ):
        assert function in COMMON
    for values in upstream.values():
        output = [native_handoff(limit_peak((value, value, value), peak)[0], 203.0) for value in values]
        assert all(isfinite(value) and value >= 0.0 for value in output)
        assert all(left <= right for left, right in zip(output, output[1:]))


def test_game_nits_does_not_cancel() -> None:
    value = 0.18
    outputs = {game_nits: native_handoff(value, game_nits) for game_nits in (80.0, 203.0, 400.0)}
    for game_nits, output in outputs.items():
        assert abs(output / outputs[203.0] - game_nits / 203.0) < 1e-12
    assert COMMON.count("RENODX_DIFFUSE_WHITE_NITS / TLOU_NATIVE_ENCODER_REFERENCE_NITS") == 1


def test_uniform_peak_limit_preserves_ratios() -> None:
    limited = limit_peak((4.0, 2.0, 1.0), 2.0)
    assert limited == (2.0, 1.0, 0.5)
    assert limited[0] / limited[1] == 2.0
    assert limited[1] / limited[2] == 2.0


def test_neutwo_native_anchor_is_finite_and_monotonic() -> None:
    peak = 1000.0 / 203.0
    reference_input = native_inverse(REFERENCE_GRAY * FIXED_REFERENCE_NITS / ENCODER_REFERENCE_NITS)
    reference_output = native_curve(reference_input) * ENCODER_REFERENCE_NITS / FIXED_REFERENCE_NITS
    assert abs(neutwo_anchor(reference_input, peak, 100.0, reference_input, reference_output) - reference_output) < 1e-12
    values = [neutwo_anchor(index / 64.0 * 100.0, peak, 100.0, reference_input, reference_output) for index in range(65)]
    assert all(isfinite(value) and value >= 0.0 for value in values)
    assert all(left <= right for left, right in zip(values, values[1:]))
    assert abs(values[-1] - peak) < 1e-12


def test_lut_hdr_coordinate_round_trip() -> None:
    for compression in (1.0, 2.0, 4.0, 8.0, 16.0):
        maximum = 2.0 ** (4.0 / 3.0) * compression ** (1.0 / 3.0)
        for fraction in (0.0, 1e-6, 0.001, 0.05, 0.25, 0.5, 0.75, 0.999999):
            source = maximum * fraction
            encoded = compress_lut_srgb(source, compression)
            decoded = decompress_lut_srgb(encoded, compression)
            assert abs(decoded - source) <= 2e-12 * max(1.0, source)


def test_grading_lut_does_not_reapply_native_curve() -> None:
    reference_output = REFERENCE_GRAY
    assert native_inverse(reference_output) > reference_output
    assert "return state.neutral_sdr_bt709;" in COMMON
    assert "TLOUNativeCurveInverse(state.neutral_sdr_bt709" not in COMMON


def test_sparse_descriptor_update_and_copy_contract() -> None:
    candidates = {0x700: "t7", 0x800: "t8"}
    sparse: dict[tuple[int, int], tuple[str, object]] = {}

    def update(heap: int, first: int, values: list[tuple[str, object]]) -> None:
        for index, (kind, value) in enumerate(values, first):
            key = (heap, index)
            sparse.pop(key, None)
            if kind == "srv" and value in candidates:
                sparse[key] = (kind, candidates[value])
            elif kind == "cbv" and int(value) >= 832:
                sparse[key] = (kind, value)

    def copy(source: tuple[int, int], destination: tuple[int, int], count: int) -> None:
        copied = [sparse.get((source[0], source[1] + i)) for i in range(count)]
        for i, value in enumerate(copied):
            key = (destination[0], destination[1] + i)
            sparse.pop(key, None)
            if value is not None:
                sparse[key] = value

    update(1, 10, [("srv", 0x700), ("srv", 0x123), ("cbv", 1024)])
    assert sparse == {(1, 10): ("srv", "t7"), (1, 12): ("cbv", 1024)}
    copy((1, 10), (2, 20), 3)
    assert sparse[(2, 20)] == ("srv", "t7")
    assert sparse[(2, 22)] == ("cbv", 1024)
    update(2, 20, [("srv", 0x999)])
    assert (2, 20) not in sparse


def test_lut_dirty_key_contract() -> None:
    base = (
        1, 1000.0, 203.0, 1.0, 1.0, 1.0,
        31.0 / 32.0, 0.5 / 32.0, 8.0, 0.0,
        0x700, 0x701, 1, 0x800, 0x801, 1,
    )
    assert base == tuple(base)
    for index in range(len(base)):
        changed = list(base)
        changed[index] = changed[index] + 1
        assert tuple(changed) != base


def test_requested_modes_are_game_local_and_wired() -> None:
    expected_modes = {
        "TLOU_TONE_MAP_TYPE_VANILLA": "Vanilla",
        "TLOU_TONE_MAP_TYPE_PSYCHOV17": "PsychoV-17",
        "TLOU_TONE_MAP_TYPE_PSYCHOV22": "PsychoV-22",
        "TLOU_TONE_MAP_TYPE_PSYCHOV24": "PsychoV-24",
        "TLOU_TONE_MAP_TYPE_PSYCHOV25": "PsychoV-25",
        "TLOU_TONE_MAP_TYPE_PSYCHOV30": "PsychoV-30",
        "TLOU_TONE_MAP_TYPE_RENODRT": "RenoDRT",
        "TLOU_TONE_MAP_TYPE_NEUTWO": "NeuTwo",
    }
    for mode, label in expected_modes.items():
        assert mode in SHARED
        assert f'"{label}"' in ADDON
    for version, path in PSYCHOV_PATHS.items():
        source = path.read_text(encoding="utf-8")
        assert f"psychotm_test{version}(" in source
        assert "SPDX-License-Identifier: MIT" in source
    assert "psychov_test25_nrg.hlsli" in PSYCHOV_PATHS[25].read_text(encoding="utf-8")
    assert "config.mid_gray_value = reference_input" in COMMON
    assert "config.mid_gray_nits = reference_output * 100.f" in COMMON
    assert "sizeof(ShaderInjectData) == 48" in SHARED
    for mode, path in VARIANT_PATHS.items():
        wrapper = path.read_text(encoding="utf-8")
        assert f"#define TLOU_COMPILED_TONE_MAP_TYPE {mode}" in wrapper
        assert '#include "./lut_impl.hlsli"' in wrapper
    for shader in (
        "__lut_psychov17",
        "__lut_psychov22",
        "__lut_psychov24",
        "__lut_psychov25",
        "__lut_psychov30",
        "__lut_renodrt",
        "__lut_neutwo",
    ):
        assert shader in ADDON
    assert "AddRuntimeReplacement" not in ADDON
    assert "RemoveRuntimeReplacements" not in ADDON


def test_hdr_and_vanilla_gates_are_explicit() -> None:
    assert "if (tone_map_type == 0) return true;" in ADDON
    assert "native.output_mode != 3" in ADDON
    assert "native.lut_enabled <= 0.5f" in ADDON
    assert "EnsureLutPipeline" in ADDON.split("native.output_mode != 3", 1)[1]
    assert LUT.count("TLOUPrepareToneMapLut(") == 1
    assert LUT.count("TLOUFinalizeToneMap(") == 1
    assert "RENODX_UI_MAGIC" in GUI
    assert ": 1.f;" in GUI
    assert "PatchGuiConstants" in ADDON


def test_lut_path_does_not_replace_the_full_resolution_shader() -> None:
    game_path = ROOT / "src/games/thelastofuspart1"
    assert not list(game_path.glob("prepost*.hlsl"))
    assert "bool OnDispatch(" in ADDON
    assert "GetCurrentComputeShaderHash(compute_state)" in ADDON
    assert "reshade::register_event<reshade::addon_event::dispatch>(OnDispatch)" in ADDON
    assert "ResolveToneMapBindings" in ADDON
    assert "OnUpdateDescriptorTables" in ADDON
    assert "OnCopyDescriptorTables" in ADDON
    assert "candidate_views" in TRACKER
    assert "sparse_descriptors" in TRACKER
    assert "descriptor::trace_descriptor_tables" not in ADDON
    assert "UpdateBoundView(" in ADDON
    assert "device->update_descriptor_tables(1u, &update)" in ADDON
    assert "const bool rebuild_lut" in ADDON
    assert "data->lut_key = current_key" in ADDON
    assert "use_pipeline_layout_cloning" not in ADDON
    assert "force_pipeline_cloning" not in ADDON
    assert "renodx::mods::shader" not in ADDON
    assert "renodx::utils::shader::Use(reason);" in ADDON
    assert "QueueCompileTimeReplacement" in ADDON
    assert "TLOU_GUI_SHADER_HASH" in ADDON
    assert "RWTexture3D<float4> u0" in LUT
    assert "[numthreads(4, 4, 4)]" in LUT


def main() -> None:
    getcontext().prec = 80
    tests = (
        test_curve_inverse_and_derivative,
        test_reference_anchor_and_log_slope,
        test_mode_neutral_glue_is_finite_and_monotonic,
        test_game_nits_does_not_cancel,
        test_uniform_peak_limit_preserves_ratios,
        test_neutwo_native_anchor_is_finite_and_monotonic,
        test_lut_hdr_coordinate_round_trip,
        test_grading_lut_does_not_reapply_native_curve,
        test_sparse_descriptor_update_and_copy_contract,
        test_lut_dirty_key_contract,
        test_requested_modes_are_game_local_and_wired,
        test_hdr_and_vanilla_gates_are_explicit,
        test_lut_path_does_not_replace_the_full_resolution_shader,
    )
    for test in tests:
        test()
    print(f"thelastofuspart1 HDR tone-map contract: {len(tests)} checks passed")


if __name__ == "__main__":
    main()
