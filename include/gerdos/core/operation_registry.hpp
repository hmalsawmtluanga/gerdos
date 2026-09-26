#pragma once

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "gerdos/core/operation.hpp"

namespace gerdos {

class OperationRegistry {
public:
    OperationRegistry() = default;

    OperationRegistry(const OperationRegistry&) = delete;
    OperationRegistry& operator=(const OperationRegistry&) = delete;
    OperationRegistry(OperationRegistry&&) = delete;
    OperationRegistry& operator=(OperationRegistry&&) = delete;

    [[nodiscard]] Operation* create_operation(
        OperationDescription description) {
        const auto id = description.id;

        if (!id.valid()) {
            return nullptr;
        }

        for (const auto data_id : description.inputs) {
            if (!data_id.valid()) {
                return nullptr;
            }
        }

        for (const auto data_id : description.outputs) {
            if (!data_id.valid()) {
                return nullptr;
            }
        }

        for (const auto dependency : description.dependencies) {
            if (!dependency.valid()) {
                return nullptr;
            }
        }

        for (std::size_t i = 0;
             i < description.resource_requirements.size();
             ++i) {
            const auto& requirement =
                description.resource_requirements[i];

            if (!requirement.valid()) {
                return nullptr;
            }

            for (std::size_t j = 0; j < i; ++j) {
                if (description.resource_requirements[j].role ==
                    requirement.role) {
                    return nullptr;
                }
            }
        }

        if (operations_.contains(id) || retired_ids_.contains(id)) {
            return nullptr;
        }

        auto operation =
            std::make_unique<Operation>(std::move(description));
        auto* operation_ptr = operation.get();

        operations_.emplace(id, std::move(operation));
        return operation_ptr;
    }

    [[nodiscard]] Operation* find_operation(
        OperationId id) noexcept {
        const auto it = operations_.find(id);

        if (it == operations_.end()) {
            return nullptr;
        }

        return it->second.get();
    }

    [[nodiscard]] const Operation* find_operation(
        OperationId id) const noexcept {
        const auto it = operations_.find(id);

        if (it == operations_.end()) {
            return nullptr;
        }

        return it->second.get();
    }

    [[nodiscard]] bool remove_operation(OperationId id) {
        const auto it = operations_.find(id);

        if (it == operations_.end()) {
            return false;
        }

        retired_ids_.insert(id);
        operations_.erase(it);
        return true;
    }

    [[nodiscard]] std::size_t operation_count() const noexcept {
        return operations_.size();
    }

    template <typename Fn>
    void for_each_operation(Fn&& fn) {
        for (auto& [id, operation] : operations_) {
            (void)id;
            fn(operation.get());
        }
    }

    template <typename Fn>
    void for_each_operation(Fn&& fn) const {
        for (const auto& [id, operation] : operations_) {
            (void)id;
            fn(operation.get());
        }
    }

private:
    std::unordered_map<
        OperationId,
        std::unique_ptr<Operation>> operations_;

    std::unordered_set<OperationId> retired_ids_;
};

} // namespace gerdos
