#include "gerdos/gerdos.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "gerdos/core/binding_admissibility.hpp"
#include "gerdos/core/binding_planner.hpp"
#include "gerdos/core/device_registry.hpp"
#include "gerdos/core/execution_admission.hpp"
#include "gerdos/core/execution_registry.hpp"
#include "gerdos/core/executor.hpp"
#include "gerdos/core/operation_registry.hpp"
#include "gerdos/core/physical_binding_resolver.hpp"
#include "gerdos/core/physical_binding_validation.hpp"
#include "gerdos/core/workload_artifact.hpp"
#include "gerdos/cpu/cpu_backend.hpp"

struct gerdos_runtime {
    gerdos::DeviceRegistry devices;
    gerdos::DataRegistry data;
    gerdos::OperationRegistry operations;
    gerdos::ExecutionRegistry executions;
    gerdos::Topology topology;
    gerdos::MeasurementRegistry measurements;
    gerdos::CpuBackend backend;
    unsigned long long next_execution{1};

    gerdos_runtime() {
        auto* host = devices.create_device(
            gerdos::DeviceDescription{gerdos::DeviceId{100}, "host"});
        (void)host->add_resource(
            gerdos::Resource{
                gerdos::ResourceDescription{
                    gerdos::ResourceId{101},
                    gerdos::DeviceId{100},
                    gerdos::ResourceKind::MEMORY,
                    "ram",
                },
            });
        host->find_resource(gerdos::ResourceId{101})
            ->set_availability(gerdos::ResourceAvailability::AVAILABLE);
        (void)host->add_resource(
            gerdos::Resource{
                gerdos::ResourceDescription{
                    gerdos::ResourceId{102},
                    gerdos::DeviceId{100},
                    gerdos::ResourceKind::COMPUTE,
                    "compute",
                },
            });
        host->find_resource(gerdos::ResourceId{102})
            ->set_availability(gerdos::ResourceAvailability::AVAILABLE);
    }
};

extern "C" {

gerdos_runtime* gerdos_create() {
    try {
        return new gerdos_runtime{};
    } catch (...) {
        return nullptr;
    }
}

void gerdos_destroy(gerdos_runtime* runtime) {
    delete runtime;
}

int gerdos_load(gerdos_runtime* runtime, const char* text) {
    if (runtime == nullptr || text == nullptr) {
        return -1;
    }

    try {
        const auto parsed = gerdos::parse_artifact(text);

        if (!parsed.ok) {
            return static_cast<int>(parsed.error.line);
        }

        std::string reason;

        if (!gerdos::populate_registries(
                parsed.artifact, runtime->devices, runtime->data,
                runtime->operations, reason)) {
            return -1;
        }
    } catch (...) {
        return -1;
    }

    return 0;
}

int gerdos_run(gerdos_runtime* runtime) {
    if (runtime == nullptr) {
        return -1;
    }

    try {
        gerdos::Executor executor(
            runtime->executions, runtime->operations, runtime->devices,
            runtime->data, runtime->backend, &runtime->measurements);
        gerdos::BindingPlanner planner(
            runtime->devices, runtime->data, runtime->topology,
            runtime->measurements);

        const auto planned = planner.plan_attempts(runtime->operations);
        int coherent = 0;

        for (const auto& step : planned) {
            const auto* operation =
                runtime->operations.find_operation(step.id);

            if (operation == nullptr) {
                return -1;
            }

            gerdos::PhysicalBindingValidator validator;
            gerdos::BindingResolver resolver(
                runtime->devices, runtime->data);
            gerdos::BindingAdmissibilityValidator admissibility;
            gerdos::ExecutionAdmissionValidator admission(
                runtime->devices, runtime->data);

            if (!validator.validate(step.binding) ||
                !resolver.resolve(step.binding).fully_resolved() ||
                !admissibility.admissible(*operation, step.binding)) {
                return -1;
            }

            auto* execution = runtime->executions.create_execution(
                gerdos::ExecutionDescription{
                    gerdos::ExecutionId{runtime->next_execution++},
                    step.id,
                });

            if (!execution->bind(step.binding) ||
                !admission.admit(*execution).has_value() ||
                !executor.start(execution->description().id)) {
                return -1;
            }

            std::vector<gerdos::AttemptStatus> outcomes;
            executor.advance(outcomes);

            while (outcomes.empty()) {
                executor.advance(outcomes);
            }

            if (outcomes.size() != 1 ||
                outcomes.front().integrity !=
                    gerdos::AttemptIntegrity::COHERENT) {
                return -1;
            }

            ++coherent;
        }

        return coherent;
    } catch (...) {
        return -1;
    }
}

float gerdos_sample(
    gerdos_runtime* runtime,
    unsigned long long data,
    unsigned long long residency,
    size_t index) {
    if (runtime == nullptr) {
        return 0.0f;
    }

    try {
        return runtime->backend.sample(
            gerdos::DataResidencyRef{
                gerdos::DataId{data},
                gerdos::DataResidencyId{residency},
            },
            index);
    } catch (...) {
        return 0.0f;
    }
}

unsigned long long gerdos_evidence(const gerdos_runtime* runtime) {
    if (runtime == nullptr) {
        return 0;
    }

    try {
        return runtime->measurements
            .summarize(
                gerdos::ResourceRef{
                    gerdos::DeviceId{100}, gerdos::ResourceId{102},
                },
                gerdos::MeasurementQuantity::DURATION_NS)
            .succeeded_observations;
    } catch (...) {
        return 0;
    }
}

} /* extern "C" */
