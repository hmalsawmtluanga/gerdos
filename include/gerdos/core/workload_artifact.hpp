#pragma once

#include <charconv>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "gerdos/core/data_registry.hpp"
#include "gerdos/core/device_registry.hpp"
#include "gerdos/core/operation_registry.hpp"
#include "gerdos/core/workload.hpp"

namespace gerdos {

// A parsed workload artifact (version 1): data identities with residency
// declarations plus the workload itself. Population into live registries
// is the caller's (devices must exist first — the artifact names homes
// by id, never declares hardware).
struct Artifact {
    Workload workload;
    struct ResidencySeed {
        DataId data;
        DataResidencyId residency;
        ResourceRef home;
        std::string representation;
        bool usable{false};
    };
    struct DataSeed {
        DataId id;
        std::string name;
        std::vector<ResidencySeed> residencies;
    };
    std::vector<DataSeed> data;
};

struct ArtifactError {
    std::size_t line{0};
    std::string reason;
};

[[nodiscard]] inline std::optional<std::uint64_t> parse_uint(
    std::string_view text) noexcept {
    if (text.empty()) {
        return std::nullopt;
    }

    for (const char c : text) {
        if (c < '0' || c > '9') {
            return std::nullopt;
        }
    }

    std::uint64_t value = 0;
    const auto* first = text.data();
    const auto* last = first + text.size();
    const auto parsed = std::from_chars(first, last, value);

    if (parsed.ec != std::errc{} || parsed.ptr != last) {
        return std::nullopt;
    }

    return value;
}

// Strict decimal grammar: optional sign, digits with at most one point,
// optional decimal exponent. Rejects strtof's nan/inf/hex forms — the
// artifact format speaks decimals, not C literals.
[[nodiscard]] inline std::optional<float> parse_float(
    std::string_view text) noexcept {
    if (text.empty()) {
        return std::nullopt;
    }

    std::size_t i = 0;

    if (text[i] == '+' || text[i] == '-') {
        ++i;
    }

    bool digits = false;
    bool point = false;

    for (; i < text.size(); ++i) {
        const char c = text[i];

        if (c >= '0' && c <= '9') {
            digits = true;
        } else if (c == '.' && !point) {
            point = true;
        } else {
            break;
        }
    }

    if (!digits) {
        return std::nullopt;
    }

    if (i < text.size() && (text[i] == 'e' || text[i] == 'E')) {
        ++i;

        if (i < text.size() && (text[i] == '+' || text[i] == '-')) {
            ++i;
        }

        bool exp_digits = false;

        for (; i < text.size(); ++i) {
            if (text[i] >= '0' && text[i] <= '9') {
                exp_digits = true;
            } else {
                break;
            }
        }

        if (!exp_digits) {
            return std::nullopt;
        }
    }

    if (i != text.size()) {
        return std::nullopt;
    }

    std::string owned{text};
    const char* first = owned.c_str();
    char* end = nullptr;
    const float value = std::strtof(first, &end);

    if (end == first || *end != 0) {
        return std::nullopt;
    }

    return value;
}

[[nodiscard]] inline std::vector<std::string> split_words(std::string_view line) {
    std::vector<std::string> words;
    std::size_t i = 0;

    while (i < line.size()) {
        while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) {
            ++i;
        }

        if (i >= line.size() || line[i] == '#') {
            break;
        }

        std::size_t end = i;

        while (end < line.size() && line[end] != ' ' && line[end] != '\t' && line[end] != '#') {
            ++end;
        }

        words.emplace_back(line.substr(i, end - i));
        i = end;
    }

    return words;
}

[[nodiscard]] inline std::optional<WorkForm> parse_form(std::string_view text) noexcept {
    if (text == "ELEMENTWISE_AFFINE" || text == "AFFINE") return WorkForm::ELEMENTWISE_AFFINE;
    if (text == "MATRIX_PRODUCT") return WorkForm::MATRIX_PRODUCT;
    if (text == "REDUCE_SUM") return WorkForm::REDUCE_SUM;
    if (text == "EXPONENTIAL") return WorkForm::EXPONENTIAL;
    if (text == "REDUCE_MAX") return WorkForm::REDUCE_MAX;
    if (text == "REDUCE_MIN") return WorkForm::REDUCE_MIN;
    if (text == "ELEMENTWISE_MIN") return WorkForm::ELEMENTWISE_MIN;
    if (text == "ELEMENTWISE_MAX") return WorkForm::ELEMENTWISE_MAX;
    if (text == "MASK_SELECT") return WorkForm::MASK_SELECT;
    if (text == "GATHER") return WorkForm::GATHER;
    return std::nullopt;
}

[[nodiscard]] inline std::optional<WorkDtype> parse_dtype(std::string_view text) noexcept {
    if (text == "F32") return WorkDtype::F32;
    if (text == "F16") return WorkDtype::F16;
    if (text == "I8") return WorkDtype::I8;
    return std::nullopt;
}

[[nodiscard]] inline std::optional<ResourceBindingRole> parse_role(std::string_view text) noexcept {
    if (text == "COMPUTE") return ResourceBindingRole::COMPUTE;
    if (text == "TRANSFER") return ResourceBindingRole::TRANSFER;
    return std::nullopt;
}

// Parse one artifact document. Returns the artifact, or the first error
// with its 1-based line number. Fail-closed: any malformed line refuses
// the whole document — never a partial workload.
struct ArtifactParse {
    bool ok{false};
    Artifact artifact;
    ArtifactError error;
};

[[nodiscard]] inline ArtifactParse parse_artifact(std::string_view document) {
    ArtifactParse result;
    Artifact artifact;
    bool have_version = false;
    bool have_workload = false;

    auto fail = [&](std::size_t line, const char* reason) {
        result.error = ArtifactError{line, reason};
        return result;
    };

    // Seeds are looked up by index, never by pointer: pushing new data
    // reallocates the seed vector, which would dangle any pointer that
    // later usable lines dereference.
    auto find_data = [&](DataId id) -> std::optional<std::size_t> {
        for (std::size_t i = 0; i < artifact.data.size(); ++i) {
            if (artifact.data[i].id == id) {
                return i;
            }
        }

        return std::nullopt;
    };

    std::size_t line_number = 0;
    std::size_t pos = 0;

    while (pos < document.size()) {
        std::size_t end = document.find('\n', pos);

        if (end == std::string_view::npos) {
            end = document.size();
        }

        ++line_number;
        std::string_view line = document.substr(pos, end - pos);

        // Text-format law: one trailing CR per line is a CRLF twin,
        // stripped before any other rule runs.
        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1);
        }

        const auto words = split_words(line);
        pos = end + 1;

        if (words.empty()) {
            continue;
        }

        const std::string_view kind = words[0];

        if (kind == "version") {
            if (words.size() != 2 || words[1] != "1" || have_version) {
                return fail(line_number, "version must be exactly 'version 1' once");
            }

            have_version = true;
            continue;
        }

        if (!have_version) {
            return fail(line_number, "first non-empty line must be 'version 1'");
        }

        if (kind == "workload") {
            if (words.size() != 2 || have_workload) {
                return fail(line_number, "workload names the stream once");
            }

            artifact.workload.name = std::string{words[1]};
            have_workload = true;
            continue;
        }

        if (kind == "data") {
            if (words.size() != 3) {
                return fail(line_number, "data needs <data_id> <name>");
            }

            const auto id = parse_uint(words[1]);

            if (!id.has_value() ||
                find_data(DataId{*id}).has_value()) {
                return fail(line_number, "data id must be a unique integer");
            }

            artifact.data.push_back(
                Artifact::DataSeed{DataId{*id}, std::string{words[2]}, {}});
            continue;
        }

        if (kind == "residency") {
            if (words.size() != 6) {
                return fail(line_number, "residency needs <data> <residency> <device> <resource> <repr>");
            }

            const auto data_id = parse_uint(words[1]);
            const auto res_id = parse_uint(words[2]);
            const auto device_id = parse_uint(words[3]);
            const auto resource_id = parse_uint(words[4]);
            const auto seed = data_id.has_value()
                ? find_data(DataId{*data_id})
                : std::optional<std::size_t>{};

            if (!data_id.has_value() || !res_id.has_value() ||
                !device_id.has_value() || !resource_id.has_value() ||
                !seed.has_value()) {
                return fail(line_number, "residency ids must be integers naming known data");
            }

            for (const auto& existing :
                 artifact.data[*seed].residencies) {
                if (existing.residency == DataResidencyId{*res_id}) {
                    return fail(line_number, "duplicate residency id");
                }
            }

            artifact.data[*seed].residencies.push_back(Artifact::ResidencySeed{
                DataId{*data_id},
                DataResidencyId{*res_id},
                ResourceRef{DeviceId{*device_id}, ResourceId{*resource_id}},
                std::string{words[5]},
                false,
            });
            continue;
        }

        if (kind == "usable") {
            if (words.size() != 3) {
                return fail(line_number, "usable needs <data_id> <residency_id>");
            }

            const auto data_id = parse_uint(words[1]);
            const auto res_id = parse_uint(words[2]);
            const auto seed = data_id.has_value()
                ? find_data(DataId{*data_id})
                : std::optional<std::size_t>{};

            if (!seed.has_value() || !res_id.has_value()) {
                return fail(line_number, "usable names known data with an integer residency");
            }

            bool found = false;

            for (auto& res : artifact.data[*seed].residencies) {
                if (res.residency == DataResidencyId{*res_id}) {
                    res.usable = true;
                    found = true;
                }
            }

            if (!found) {
                return fail(line_number, "usable names a declared residency");
            }

            continue;
        }

        if (kind == "op") {
            // Views into words[] (owned std::string storage, alive for
            // the whole line) — never substr() temporaries.
            auto need = [&](std::string_view key) -> std::optional<std::string_view> {
                for (const auto& word : words) {
                    const std::string_view view = word;

                    if (view.size() > key.size() && view.substr(0, key.size()) == key && view[key.size()] == '=') {
                        return view.substr(key.size() + 1);
                    }
                }

                return std::nullopt;
            };

            if (words.size() < 3) {
                return fail(line_number, "op needs <op_id> and key=value fields");
            }

            const auto op_id = parse_uint(words[1]);

            if (!op_id.has_value()) {
                return fail(line_number, "op id must be an integer");
            }

            for (const auto& existing : artifact.workload.operations) {
                if (existing.id == OperationId{*op_id}) {
                    return fail(line_number, "duplicate op id");
                }
            }

            auto parse_ids = [&](std::string_view text)
                -> std::optional<std::vector<std::uint64_t>> {
                if (text == "-") {
                    return std::vector<std::uint64_t>{};
                }

                std::vector<std::uint64_t> ids;
                std::size_t at = 0;

                while (at <= text.size()) {
                    std::size_t comma = text.find(',', at);

                    if (comma == std::string_view::npos) {
                        comma = text.size();
                    }

                    const auto id = parse_uint(text.substr(at, comma - at));

                    if (!id.has_value()) {
                        return std::nullopt;
                    }

                    ids.push_back(*id);
                    at = comma + 1;
                }

                return ids;
            };

            const auto inputs = need("inputs");
            const auto outputs = need("outputs");
            const auto deps = need("deps");
            const auto req = need("req");
            const auto form_text = need("form");
            const auto dtype_text = need("dtype");
            const auto elements = need("elements");
            const auto passes = need("passes");
            const auto ds = need("ds");
            const auto ss = need("ss");
            const auto constant = need("c");

            if (!inputs.has_value() || !outputs.has_value() || !deps.has_value() ||
                !req.has_value() || !form_text.has_value() || !dtype_text.has_value() ||
                !elements.has_value() || !passes.has_value() || !ds.has_value() ||
                !ss.has_value() || !constant.has_value()) {
                return fail(line_number, "op misses a required field");
            }

            const auto in_ids = parse_ids(*inputs);
            const auto out_ids = parse_ids(*outputs);
            const auto dep_ids = parse_ids(*deps);
            const auto form = parse_form(*form_text);
            const auto dtype = parse_dtype(*dtype_text);
            const auto element_count = parse_uint(*elements);
            const auto pass_count = parse_uint(*passes);
            const auto dst_scale = parse_float(*ds);
            const auto src_scale = parse_float(*ss);
            const auto bias = parse_float(*constant);

            if (!in_ids.has_value() || !out_ids.has_value() || !dep_ids.has_value() ||
                !form.has_value() || !dtype.has_value() || !element_count.has_value() ||
                !pass_count.has_value() || !dst_scale.has_value() ||
                !src_scale.has_value() || !bias.has_value()) {
                return fail(line_number, "op field value malformed");
            }

            std::vector<ResourceRequirement> requirements;
            {
                std::string_view rest = *req;

                while (!rest.empty()) {
                    std::size_t comma = rest.find(',');
                    std::string_view one = (comma == std::string_view::npos)
                        ? rest
                        : rest.substr(0, comma);
                    rest = (comma == std::string_view::npos)
                        ? std::string_view{}
                        : rest.substr(comma + 1);

                    const std::size_t colon = one.find(':');

                    if (colon == std::string_view::npos) {
                        return fail(line_number, "req entries are ROLE:min");
                    }

                    const auto role = parse_role(one.substr(0, colon));
                    const auto minimum = parse_uint(one.substr(colon + 1));

                    if (!role.has_value() || !minimum.has_value() || *minimum == 0) {
                        return fail(line_number, "req role unknown or minimum vacuous");
                    }

                    requirements.push_back(
                        ResourceRequirement{*role, static_cast<std::size_t>(*minimum)});
                }
            }

            if (requirements.empty()) {
                return fail(line_number, "op needs at least one requirement");
            }

            std::size_t rows = 0;
            std::size_t inner = 0;
            std::size_t columns = 0;

            if (*form == WorkForm::MATRIX_PRODUCT) {
                // Early refusal before any dereference: the analyzer
                // sees no path where an empty optional is read.
                const auto r = need("rows");
                const auto i = need("inner");
                const auto c = need("columns");

                if (!r.has_value() || !i.has_value() ||
                    !c.has_value()) {
                    return fail(line_number, "matrix form needs a complete shape");
                }

                const auto rv = parse_uint(*r);
                const auto iv = parse_uint(*i);
                const auto cv = parse_uint(*c);

                if (!rv.has_value() || !iv.has_value() ||
                    !cv.has_value() || *rv == 0 || *iv == 0 ||
                    *cv == 0) {
                    return fail(line_number, "matrix form needs a complete shape");
                }

                rows = static_cast<std::size_t>(*rv);
                inner = static_cast<std::size_t>(*iv);
                columns = static_cast<std::size_t>(*cv);
            }

            // Well-formedness is a parse-time refusal: the seam never
            // sees work that is not well-formed, and registration is
            // not execution — but an artifact declares executable work,
            // so hostile sizing is refused here with its line number.
            WorkDescription preview{
                static_cast<std::size_t>(*element_count),
                static_cast<std::size_t>(*pass_count),
                *dst_scale,
                *src_scale,
                *bias,
                *form,
                rows,
                inner,
                columns,
                *dtype,
            };

            if (!preview.valid()) {
                return fail(line_number, "work not well-formed");
            }

            OperationDescription description{
                OperationId{*op_id},
                {},
                {},
                {},
                std::move(requirements),
                WorkDescription{
                    static_cast<std::size_t>(*element_count),
                    static_cast<std::size_t>(*pass_count),
                    *dst_scale,
                    *src_scale,
                    *bias,
                    *form,
                    rows,
                    inner,
                    columns,
                    *dtype,
                },
            };

            for (const auto id : *in_ids) {
                description.inputs.push_back(DataId{id});
            }

            for (const auto id : *out_ids) {
                description.outputs.push_back(DataId{id});
            }

            for (const auto id : *dep_ids) {
                description.dependencies.push_back(OperationId{id});
            }

            artifact.workload.operations.push_back(std::move(description));
            continue;
        }

        return fail(line_number, "unknown line kind");
    }

    if (!have_workload) {
        return fail(line_number, "artifact names no workload");
    }

    result.ok = true;
    result.artifact = std::move(artifact);
    return result;
}

// Populate live registries from a parsed artifact. Devices must exist
// first (the artifact names homes by id). Returns false with the reason
// when a home is missing or a registry refuses the seed.
[[nodiscard]] inline bool populate_registries(
    const Artifact& artifact,
    DeviceRegistry& devices,
    DataRegistry& data,
    OperationRegistry& operations,
    std::string& reason) {
    (void)devices;

    for (const auto& seed : artifact.data) {
        auto* datum = data.create_data(DataDescription{seed.id, seed.name});

        if (datum == nullptr) {
            reason = "data refused";
            return false;
        }

        for (const auto& res : seed.residencies) {
            if (!datum->add_residency(DataResidency{DataResidencyDescription{
                    res.residency, res.data, res.home, res.representation}})) {
                reason = "residency refused";
                return false;
            }

            if (res.usable &&
                !datum->find_residency(res.residency)
                     ->set_state(DataResidencyState::VALID)) {
                reason = "usable refused";
                return false;
            }
        }
    }

    for (const auto& description : artifact.workload.operations) {
        if (operations.create_operation(description) == nullptr) {
            reason = "operation refused";
            return false;
        }
    }

    return true;
}

} // namespace gerdos

