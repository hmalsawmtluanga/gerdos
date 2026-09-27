#pragma once

#include <cstdio>
#include <cstdlib>

namespace gerdos_test {

inline void check(bool condition, const char* expression) {
    if (!condition) {
        std::fprintf(stderr, "TEST FAILURE: %s\n", expression);
        std::abort();
    }
}

} // namespace gerdos_test

#define GERDOS_CHECK(condition) \
    ::gerdos_test::check((condition), #condition)
