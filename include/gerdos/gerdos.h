/* GERDOS C ABI: the stable consumer boundary. Opaque handles, no C++
 * templates across the boundary, no new semantics — every call forwards
 * to the audited core. Compile consumers with a C compiler.
 * Flagship backends stay untouched behind this shim. */
#ifndef GERDOS_H
#define GERDOS_H

#include <stddef.h>

/* Symbol export is part of the ABI contract: explicit export on
 * toolchains that hide symbols by default, empty elsewhere.
 * GERDOS_BUILD_CABI is defined when compiling the gerdos_c library
 * itself; consumers leave it undefined and import. */
#if defined(_WIN32) && defined(GERDOS_BUILD_CABI)
#define GERDOS_API __declspec(dllexport)
#elif defined(_WIN32)
#define GERDOS_API __declspec(dllimport)
#else
#define GERDOS_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gerdos_runtime gerdos_runtime;

/* Create a host-only runtime (CPU engine, empty registries). */
GERDOS_API gerdos_runtime* gerdos_create(void);

/* Destroy a runtime created by gerdos_create. */
GERDOS_API void gerdos_destroy(gerdos_runtime* runtime);

/* Load an artifact document (version-1 text). Returns 0 on success;
 * nonzero refuses with the line number (fail-closed, never partial). */
GERDOS_API int gerdos_load(gerdos_runtime* runtime, const char* text);

/* Plan, admit, execute, and measure every ready operation in schedule
 * order. Returns the number of coherent completions, or -1 on a
 * structural failure (nothing partial: the count is exact). */
GERDOS_API int gerdos_run(gerdos_runtime* runtime);

/* Sample one element of a device-homed record. Missing records read 0. */
GERDOS_API float gerdos_sample(
    gerdos_runtime* runtime,
    unsigned long long data,
    unsigned long long residency,
    size_t index);

/* Successful observation count (evidence growth, deterministic). */
GERDOS_API unsigned long long gerdos_evidence(const gerdos_runtime* runtime);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* GERDOS_H */
