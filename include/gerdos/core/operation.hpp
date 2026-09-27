#pragma once

#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "gerdos/core/ids.hpp"
#include "gerdos/core/physical_binding.hpp"

namespace gerdos {

// A resource requirement declares required runtime participation as a
// resource binding role and a minimum number of distinct resources in that
// role. Requirements do not select a concrete Resource or Device and do not
// classify the Operation.
struct ResourceRequirement {
    ResourceBindingRole role;
    std::size_t minimum;

    [[nodiscard]] constexpr bool valid() const noexcept {
        if (minimum == 0) {
            return false;
        }

        switch (role) {
        case ResourceBindingRole::COMPUTE:
        case ResourceBindingRole::TRANSFER:
            return true;
        default:
            return false;
        }
    }
};

// The form of a declared computation. Forms are backend-interpreted
// content: no gate inspects which form an Operation declares.
enum class WorkForm : std::uint8_t {
    // dst = dst * destination_scale + source_scale * src + constant.
    ELEMENTWISE_AFFINE,
    // dst = dst * destination_scale + source_scale * (A x B) + constant,
    // row-major, with A and B the first two consuming entries.
    MATRIX_PRODUCT,
    // dst[0] = dst[0] * destination_scale + source_scale * sum(src) +
    // constant; the remaining elements are left alone.
    REDUCE_SUM,
    // dst = dst * destination_scale + source_scale * exp(src) +
    // constant, elementwise.
    EXPONENTIAL,
    // dst[0] = dst[0] * destination_scale + source_scale * max(src) +
    // constant; the remaining elements are left alone.
    REDUCE_MAX,
    // dst[0] = dst[0] * destination_scale + source_scale * min(src) +
    // constant; the remaining elements are left alone.
    REDUCE_MIN,
    // dst = dst * destination_scale + source_scale * min(A, B) +
    // constant, elementwise over the first two consuming entries.
    ELEMENTWISE_MIN,
    // dst = dst * destination_scale + source_scale * max(A, B) +
    // constant, elementwise over the first two consuming entries.
    ELEMENTWISE_MAX,
    // dst = dst * destination_scale + source_scale * (P != 0 ? A : B)
    // + constant, elementwise over the first three consuming entries
    // (predicate, value A, value B).
    MASK_SELECT,
    // dst = dst * destination_scale + source_scale * V[clamp(I)] +
    // constant, elementwise over the first two consuming entries (value
    // table, index table). Indices truncate toward zero and clamp into
    // the table range.
    GATHER,
};

// The element type of a declared computation: generic dtype vocabulary.
// F32 is the default so every existing construction site keeps its
// meaning; F16 and I8 extend the algebra without splitting forms.
// Dtype is a parameter beside form — no gate inspects it.
enum class WorkDtype : std::uint8_t {
    F32,
    F16,
    I8,
};

// The byte width of one element of a dtype.
[[nodiscard]] constexpr std::size_t dtype_bytes(WorkDtype dtype) noexcept {
    switch (dtype) {
    case WorkDtype::F32:
        return 4;
    case WorkDtype::F16:
        return 2;
    case WorkDtype::I8:
        return 1;
    }

    return 4;
}

// The declared computation of an attempt, expressed as generic data-parallel
// semantics over the bound representations, repeated passes times. Exact
// copying is the elementwise form with destination_scale 0, source_scale 1,
// constant 0, passes 1. The consuming record of the same Data supplies src —
// or, for compute shapes, the first consuming entry. Shape fields are
// interpreted by form. The exponential form applies the affine wrapper
// around exp(src); the max-reduction form folds the operand maximum into
// dst[0] and leaves the remaining elements alone. The min-reduction form
// mirrors it; the two-operand min/max forms read the first two consuming
// entries; the predicate selection reads the first three (predicate, A,
// B); the gather form reads the value table and the index table from the
// first two. Gates never inspect the work; backends interpret it at the
// seam.
struct WorkDescription {
    std::size_t elements{0};
    std::size_t passes{0};
    float destination_scale{1.0f};
    float source_scale{0.0f};
    float constant{0.0f};
    WorkForm form{WorkForm::ELEMENTWISE_AFFINE};
    std::size_t rows{0};
    std::size_t inner{0};
    std::size_t columns{0};
    WorkDtype dtype{WorkDtype::F32};

    // Well-formed executable work: non-empty and safely sized. Every
    // engine stages F32 compute scratch (four bytes per element), so the
    // bound is the scratch width — not the stored width: a narrow stored
    // dtype does not admit counts the scratch cannot hold. Gates never
    // inspect the work; the seam rejects work that is not well-formed
    // rather than executing hostile arithmetic.
    [[nodiscard]] constexpr bool valid() const noexcept {
        constexpr auto max_bytes = std::numeric_limits<std::size_t>::max();
        constexpr auto limit = max_bytes / sizeof(float);

        if (dtype != WorkDtype::F32 && dtype != WorkDtype::F16 &&
            dtype != WorkDtype::I8) {
            return false;
        }

        switch (form) {
        case WorkForm::ELEMENTWISE_AFFINE:
        case WorkForm::REDUCE_SUM:
        case WorkForm::EXPONENTIAL:
        case WorkForm::REDUCE_MAX:
        case WorkForm::REDUCE_MIN:
        case WorkForm::ELEMENTWISE_MIN:
        case WorkForm::ELEMENTWISE_MAX:
        case WorkForm::MASK_SELECT:
        case WorkForm::GATHER:
            return elements > 0 && passes > 0 && elements <= limit;
        case WorkForm::MATRIX_PRODUCT:
            return rows > 0 && inner > 0 && columns > 0 && passes > 0 &&
                   rows <= limit / inner && inner <= limit / columns &&
                   rows <= limit / columns;
        }

        return false;
    }

    // The byte width of one stored element.
    [[nodiscard]] constexpr std::size_t storage_bytes() const noexcept {
        return dtype_bytes(dtype);
    }

    // The largest operand this work touches, in elements — the storage
    // every bound representation needs, times storage_bytes() for bytes.
    [[nodiscard]] constexpr std::size_t storage_elements() const noexcept {
        if (form != WorkForm::MATRIX_PRODUCT) {
            return elements;
        }

        const auto a = rows * inner;
        const auto b = inner * columns;
        const auto c = rows * columns;

        return a > b ? (a > c ? a : c) : (b > c ? b : c);
    }
};

// An Operation is one unit of declarative executable work: it references
// data and dependencies by identity, declares required runtime participation
// as resource requirements, and declares its computation as a work
// description. An Operation does not select a concrete Resource or Device,
// and no gate classifies it.
struct OperationDescription {
    OperationId id;
    std::vector<DataId> inputs;
    std::vector<DataId> outputs;
    std::vector<OperationId> dependencies;
    std::vector<ResourceRequirement> resource_requirements{};
    WorkDescription work{};
};

class Operation {
public:
    explicit Operation(OperationDescription description)
        : description_(std::move(description)) {}

    Operation(const Operation&) = delete;
    Operation& operator=(const Operation&) = delete;
    Operation(Operation&&) noexcept = default;
    Operation& operator=(Operation&&) noexcept = delete;

    [[nodiscard]] const OperationDescription&
    description() const noexcept {
        return description_;
    }

private:
    OperationDescription description_;
};

} // namespace gerdos