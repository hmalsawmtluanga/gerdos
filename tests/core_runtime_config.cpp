#include "test_check.hpp"

#include <string>

#include "gerdos/core/runtime_config.hpp"

int main() {
    using namespace gerdos;

    {
        const auto parsed = parse_config("# comment\naccelerator = 300\n");
        GERDOS_CHECK(parsed.ok);
        GERDOS_CHECK(parsed.config.accelerator == DeviceId{300});

        const auto defaults = parse_config("");
        GERDOS_CHECK(defaults.ok);
        GERDOS_CHECK(defaults.config.accelerator == DeviceId{200});
    }

    {
        const auto unknown = parse_config("mystery = 1\n");
        GERDOS_CHECK(!unknown.ok);
        GERDOS_CHECK(unknown.error.line == 1);

        const auto malformed = parse_config("accelerator = fast\n");
        GERDOS_CHECK(!malformed.ok);
        GERDOS_CHECK(malformed.error.line == 1);

        const auto no_equals = parse_config("accelerator\n");
        GERDOS_CHECK(!no_equals.ok);
    }

    {
        const auto good = accelerator_from_env("200");
        GERDOS_CHECK(good.has_value());
        GERDOS_CHECK(*good == DeviceId{200});
        GERDOS_CHECK(!accelerator_from_env("fast").has_value());
        GERDOS_CHECK(!accelerator_from_env("").has_value());
        GERDOS_CHECK(!accelerator_from_env(nullptr).has_value());
    }

    return 0;
}
