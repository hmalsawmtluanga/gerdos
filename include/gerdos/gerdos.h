/* GERDOS C ABI: the stable consumer boundary. Opaque handles, no C++
 * templates across the boundary, no new semantics — every call forwards
 * to the audited core. Compile consumers with a C compiler.
 * Flagship backends stay untouched behind this shim. */
#ifndef GERDOS_H
#define GERDOS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gerdos_runtime gerdos_runtime;

/* Create a host-only runtime (CPU engine, empty registries). */
gerdos_runtime* gerdos_create(void);

/* Destroy a runtime created by gerdos_create. */
void gerdos_destroy(gerdos_runtime* runtime);

/* Load an artifact document (version-1 text). Returns 0 on success;
 * nonzero refuses with the line number (fail-closed, never partial). */
int gerdos_load(gerdos_runtime* runtime, const char* text);

/* Plan, admit, execute, and measure every ready operation in schedule
 * order. Returns the number of coherent completions, or -1 on a
 * structural failure (nothing partial: the count is exact). */
int gerdos_run(gerdos_runtime* runtime);

/* Sample one element of a device-homed record. Missing records read 0. */
float gerdos_sample(
    gerdos_runtime* runtime,
    unsigned long long data,
    unsigned long long residency,
    size_t index);

/* Successful observation count (evidence growth, deterministic). */
unsigned long long gerdos_evidence(const gerdos_runtime* runtime);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* GERDOS_H */
