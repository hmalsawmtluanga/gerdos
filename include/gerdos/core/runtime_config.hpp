#pragma once

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "gerdos/core/ids.hpp"

namespace gerdos {

// Runtime configuration: device identity and backend choice, resolved
// file-first (key=value text) with GERDOS_ environment overrides.
// Unknown keys are refused loudly; documented defaults match today's
// hardcoded behavior. Exploration thresholds and pool sizing stay
// hardcoded BY CONTRACT (determinism): they are not configuration.
struct RuntimeConfig {
    // Which device the accelerator backends serve. Default 200.
    DeviceId accelerator{DeviceId{200}};
};

struct ConfigError {
    std::size_t line{0};
    std::string reason;
};

struct ConfigParse {
    bool ok{false};
    RuntimeConfig config;
    ConfigError error;
};

[[nodiscard]] inline ConfigParse parse_config(std::string_view document) {
    ConfigParse result;
    RuntimeConfig config;

    auto fail = [&](std::size_t line, const char* reason) {
        result.error = ConfigError{line, reason};
        return result;
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
        pos = end + 1;

        // Text-format law: one trailing CR per line is a CRLF twin,
        // stripped before any other rule runs (a lone CR line is blank).
        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1);
        }

        std::size_t start = 0;

        while (start < line.size() && (line[start] == ' ' || line[start] == '\t')) {
            ++start;
        }

        if (start >= line.size() || line[start] == '#') {
            continue;
        }

        const std::size_t eq = line.find('=', start);

        if (eq == std::string_view::npos) {
            return fail(line_number, "config lines are key=value");
        }

        std::string key{line.substr(start, eq - start)};
        std::string value{line.substr(eq + 1)};

        while (!key.empty() && (key.back() == ' ' || key.back() == '\t')) {
            key.pop_back();
        }

        std::size_t vstart = 0;

        while (vstart < value.size() && (value[vstart] == ' ' || value[vstart] == '\t')) {
            ++vstart;
        }

        value = value.substr(vstart);

        while (!value.empty() && (value.back() == ' ' || value.back() == '\t' || value.back() == '\r')) {
            value.pop_back();
        }

        if (key == "accelerator") {
            std::uint64_t id = 0;

            if (value.empty()) {
                return fail(line_number, "accelerator must be a device id");
            }

            for (const char c : value) {
                if (c < '0' || c > '9') {
                    return fail(line_number, "accelerator must be a device id");
                }

                id = id * 10 + static_cast<std::uint64_t>(c - '0');
            }

            config.accelerator = DeviceId{id};
        } else {
            return fail(line_number, "unknown config key");
        }
    }

    result.ok = true;
    result.config = config;
    return result;
}

// Environment override: GERDOS_ACCELERATOR wins over the file value.
// Malformed environment is refused loudly (nullopt), never defaulted.
[[nodiscard]] inline std::optional<DeviceId> accelerator_from_env(const char* value) {
    if (value == nullptr || *value == 0) {
        return std::nullopt;
    }

    std::uint64_t id = 0;
    std::string text{value};

    if (text.empty()) {
        return std::nullopt;
    }

    for (const char c : text) {
        if (c < '0' || c > '9') {
            return std::nullopt;
        }
    }

    for (const char c : text) {
        id = id * 10 + static_cast<std::uint64_t>(c - '0');
    }

    return DeviceId{id};
}

} // namespace gerdos

