#pragma once

#include <cstdint>
#include <vector>

#include "gerdos/core/execution.hpp"
#include "gerdos/core/ids.hpp"
#include "gerdos/core/operation.hpp"

namespace gerdos {

// The outcome of one attempt as reported by a backend. A completion carries
// the attempt identity, whether the work succeeded, and the observed
// duration of the work; the duration is evidence measured where the work
// happened. Terminal state transitions, execution effects, result recording,
// and measurement capture are applied by the runtime integration layer.
struct BackendCompletion {
    ExecutionId execution;
    bool succeeded;
    std::uint64_t duration_ns;
};

// The single seam between the core runtime and hardware-specific execution.
// Backends perform the work described by an Operation's attempt and report
// completions. Backend-specific state — vendor handles, queues, streams,
// events, device pointers, addresses — remains behind this interface and
// must never appear in core types.
class ExecutionBackend {
public:
    ExecutionBackend() = default;

    ExecutionBackend(const ExecutionBackend&) = delete;
    ExecutionBackend& operator=(const ExecutionBackend&) = delete;
    ExecutionBackend(ExecutionBackend&&) = delete;
    ExecutionBackend& operator=(ExecutionBackend&&) = delete;

    virtual ~ExecutionBackend() = default;

    // Begins performing the work of one admitted attempt. Rejected when the
    // backend cannot accept the attempt.
    [[nodiscard]] virtual bool submit(
        const Operation& operation,
        const Execution& execution) = 0;

    // Appends attempts completed since the previous call. Never blocks the
    // runtime on hardware; simulated backends advance their deterministic
    // model here.
    virtual void poll(std::vector<BackendCompletion>& completed) = 0;
};

} // namespace gerdos