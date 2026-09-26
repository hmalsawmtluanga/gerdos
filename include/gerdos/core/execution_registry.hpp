#pragma once

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "gerdos/core/execution.hpp"

namespace gerdos {

class ExecutionRegistry {
public:
    ExecutionRegistry() = default;

    ExecutionRegistry(const ExecutionRegistry&) = delete;
    ExecutionRegistry& operator=(const ExecutionRegistry&) = delete;
    ExecutionRegistry(ExecutionRegistry&&) = delete;
    ExecutionRegistry& operator=(ExecutionRegistry&&) = delete;

    [[nodiscard]] Execution* create_execution(
        ExecutionDescription description) {
        const auto id = description.id;

        if (!id.valid()) {
            return nullptr;
        }

        if (!description.operation.valid()) {
            return nullptr;
        }

        if (executions_.contains(id) || retired_ids_.contains(id)) {
            return nullptr;
        }

        auto execution =
            std::make_unique<Execution>(std::move(description));
        auto* execution_ptr = execution.get();

        executions_.emplace(id, std::move(execution));
        return execution_ptr;
    }

    [[nodiscard]] Execution* find_execution(
        ExecutionId id) noexcept {
        const auto it = executions_.find(id);

        if (it == executions_.end()) {
            return nullptr;
        }

        return it->second.get();
    }

    [[nodiscard]] const Execution* find_execution(
        ExecutionId id) const noexcept {
        const auto it = executions_.find(id);

        if (it == executions_.end()) {
            return nullptr;
        }

        return it->second.get();
    }

    [[nodiscard]] bool remove_execution(ExecutionId id) {
        const auto it = executions_.find(id);

        if (it == executions_.end()) {
            return false;
        }

        retired_ids_.insert(id);
        executions_.erase(it);
        return true;
    }

    [[nodiscard]] std::size_t execution_count() const noexcept {
        return executions_.size();
    }

    template <typename Fn>
    void for_each_execution(Fn&& fn) {
        for (auto& [id, execution] : executions_) {
            (void)id;
            fn(execution.get());
        }
    }

    template <typename Fn>
    void for_each_execution(Fn&& fn) const {
        for (const auto& [id, execution] : executions_) {
            (void)id;
            fn(execution.get());
        }
    }

private:
    std::unordered_map<
        ExecutionId,
        std::unique_ptr<Execution>> executions_;

    std::unordered_set<ExecutionId> retired_ids_;
};

} // namespace gerdos
