#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include "gerdos/core/operation.hpp"

namespace gerdos {

// Shared dtype conversion: the one semantic home for element conversion.
// Both engines compute in F32 and convert once per direction at the home
// boundary through these helpers, so engine pairs cannot disagree on
// conversion — only on F32 arithmetic in the last ulp. Mixed-dtype
// attempts convert explicitly with round-half-away-from-zero; there is no
// silent reinterpretation. Generic vocabulary only: no vendor-intrinsic
// spellings here or anywhere in the core.

// The bit pattern of an F16 value: sign, 5-bit exponent (bias 15),
// 10-bit mantissa.
using F16Bits = std::uint16_t;

// Round-half-away-from-zero: exact on values both dtypes represent.
[[nodiscard]] inline float f16_to_f32(F16Bits bits) noexcept {
    const auto sign = static_cast<std::uint32_t>(bits >> 15);
    const auto exponent = static_cast<std::uint32_t>((bits >> 10) & 0x1f);
    const auto mantissa = static_cast<std::uint32_t>(bits & 0x3ff);

    std::uint32_t f32 = 0;

    if (exponent == 0) {
        if (mantissa == 0) {
            f32 = sign << 31;
        } else {
            // Subnormal: value = mantissa * 2^-24, exact in F32.
            const float magnitude =
                static_cast<float>(mantissa) / 16777216.0f;
            return sign == 0 ? magnitude : -magnitude;
        }
    } else if (exponent == 31) {
        f32 = (sign << 31) | (0xffu << 23) | (mantissa << 13);
    } else {
        const auto unbiased =
            static_cast<std::int32_t>(exponent) - 15 + 127;
        f32 = (sign << 31) |
            (static_cast<std::uint32_t>(unbiased) << 23) | (mantissa << 13);
    }

    float value = 0.0f;
    std::memcpy(&value, &f32, sizeof(value));
    return value;
}

[[nodiscard]] inline F16Bits f32_to_f16(float value) noexcept {
    std::uint32_t raw = 0;
    std::memcpy(&raw, &value, sizeof(raw));

    const auto sign = (raw >> 31) & 1u;
    const auto exponent = static_cast<std::int32_t>((raw >> 23) & 0xffu);
    const auto mantissa = raw & 0x7fffffu;

    // NaN and infinities saturate to F16 infinities/NaN, never trap.
    if (exponent == 255) {
        if (mantissa == 0) {
            return static_cast<F16Bits>((sign << 15) | (31u << 10));
        }

        return static_cast<F16Bits>((sign << 15) | (31u << 10) | 1u);
    }

    const auto unbiased = exponent - 127 + 15;

    // Overflow saturates to infinity; deep underflow (past subnormals)
    // flushes to signed zero — both deterministic, both tested.
    if (unbiased >= 31) {
        return static_cast<F16Bits>((sign << 15) | (31u << 10));
    }

    if (unbiased <= 0) {
        if (unbiased < -10) {
            return static_cast<F16Bits>(sign << 15);
        }

        // Subnormal: round-half-away-from-zero on the mantissa tail —
        // ties (>= half) round up, never to even.
        const auto shift = static_cast<std::uint32_t>(1 - unbiased);
        const std::uint32_t full = mantissa | 0x800000u;
        std::uint32_t kept = full >> (shift + 13);
        const std::uint32_t tail = full & ((1u << (shift + 13)) - 1u);
        const std::uint32_t half = 1u << (shift + 12);

        if (tail >= half) {
            ++kept;
        }

        return static_cast<F16Bits>(
            (sign << 15) | (kept & 0x3ffu));
    }

    // Normal: round-half-away from the 13-bit tail.
    std::uint32_t half_bits =
        (sign << 15) | (static_cast<std::uint32_t>(unbiased) << 10) |
        (mantissa >> 13);
    const std::uint32_t tail = mantissa & 0x1fffu;

    if (tail >= 0x1000u) {
        ++half_bits;
    }

    return static_cast<F16Bits>(half_bits & 0xffffu);
}

// I8 saturates: values past [-128, 127] clamp, fractions round
// half-away-from-zero. Exact on values both dtypes represent.
[[nodiscard]] inline std::int8_t f32_to_i8(float value) noexcept {
    if (std::isnan(value)) {
        return 0;
    }

    const float rounded = value >= 0.0f ? std::floor(value + 0.5f)
                                        : std::ceil(value - 0.5f);

    if (rounded >= 127.0f) {
        return 127;
    }

    if (rounded <= -128.0f) {
        return -128;
    }

    return static_cast<std::int8_t>(rounded);
}

[[nodiscard]] inline float i8_to_f32(std::int8_t value) noexcept {
    return static_cast<float>(value);
}

// Decode one stored element to F32 for computation.
[[nodiscard]] inline float decode_element(
    WorkDtype dtype,
    const unsigned char* data,
    std::size_t index) noexcept {
    switch (dtype) {
    case WorkDtype::F32: {
        float value = 0.0f;
        std::memcpy(
            &value, data + index * sizeof(float), sizeof(value));
        return value;
    }
    case WorkDtype::F16: {
        F16Bits bits = 0;
        std::memcpy(&bits, data + index * sizeof(bits), sizeof(bits));
        return f16_to_f32(bits);
    }
    case WorkDtype::I8: {
        std::int8_t value = 0;
        std::memcpy(&value, data + index, sizeof(value));
        return i8_to_f32(value);
    }
    }

    return 0.0f;
}

// Encode one F32 computation result into stored dtype.
inline void encode_element(
    WorkDtype dtype,
    unsigned char* data,
    std::size_t index,
    float value) noexcept {
    switch (dtype) {
    case WorkDtype::F32: {
        std::memcpy(
            data + index * sizeof(float), &value, sizeof(value));
        return;
    }
    case WorkDtype::F16: {
        const F16Bits bits = f32_to_f16(value);
        std::memcpy(data + index * sizeof(bits), &bits, sizeof(bits));
        return;
    }
    case WorkDtype::I8: {
        const std::int8_t narrowed = f32_to_i8(value);
        std::memcpy(data + index, &narrowed, sizeof(narrowed));
        return;
    }
    }
}

// Decode a whole allocation to F32 scratch for one kernel launch.
inline void decode_all(
    WorkDtype dtype,
    const unsigned char* data,
    float* out,
    std::size_t elements) noexcept {
    for (std::size_t i = 0; i < elements; ++i) {
        out[i] = decode_element(dtype, data, i);
    }
}

// Encode whole F32 scratch back into stored dtype.
inline void encode_all(
    WorkDtype dtype,
    const float* values,
    unsigned char* data,
    std::size_t elements) noexcept {
    for (std::size_t i = 0; i < elements; ++i) {
        encode_element(dtype, data, i, values[i]);
    }
}

} // namespace gerdos
