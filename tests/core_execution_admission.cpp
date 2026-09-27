#include "test_check.hpp"

#include <type_traits>

#include "gerdos/core/execution_admission.hpp"

int main() {
    using namespace gerdos;

    // ---------------------------------------------------------------------
    // 1. Synthetic runtime state
    // ---------------------------------------------------------------------

    DeviceRegistry devices;
    DataRegistry data_registry;

    auto* device = devices.create_device(
        DeviceDescription{
            DeviceId{100},
            "Synthetic Device",
        });
    GERDOS_CHECK(device != nullptr);

    GERDOS_CHECK(device->add_resource(
        Resource{
            ResourceDescription{
                ResourceId{200},
                DeviceId{100},
                ResourceKind::COMPUTE,
                "Synthetic Compute",
            },
        }));

    GERDOS_CHECK(device->add_resource(
        Resource{
            ResourceDescription{
                ResourceId{201},
                DeviceId{100},
                ResourceKind::MEMORY,
                "Synthetic Memory",
            },
        }));

    Resource* compute = device->find_resource(ResourceId{200});
    Resource* memory = device->find_resource(ResourceId{201});

    GERDOS_CHECK(compute != nullptr);
    GERDOS_CHECK(memory != nullptr);

    compute->set_availability(ResourceAvailability::AVAILABLE);
    memory->set_availability(ResourceAvailability::AVAILABLE);

    const ResourceRef compute_ref{DeviceId{100}, ResourceId{200}};
    const ResourceRef memory_ref{DeviceId{100}, ResourceId{201}};

    Data* data_a = data_registry.create_data(
        DataDescription{
            DataId{300},
            "Data A",
        });
    GERDOS_CHECK(data_a != nullptr);

    GERDOS_CHECK(data_a->add_residency(
        DataResidency{
            DataResidencyDescription{
                DataResidencyId{400},
                DataId{300},
                memory_ref,
                "usable-input",
            },
        }));

    GERDOS_CHECK(data_a->add_residency(
        DataResidency{
            DataResidencyDescription{
                DataResidencyId{401},
                DataId{300},
                memory_ref,
                "fresh-output",
            },
        }));

    auto* usable_input = data_a->find_residency(DataResidencyId{400});
    auto* fresh_output = data_a->find_residency(DataResidencyId{401});

    GERDOS_CHECK(usable_input != nullptr);
    GERDOS_CHECK(fresh_output != nullptr);
    GERDOS_CHECK(usable_input->set_state(DataResidencyState::VALID));

    Data* data_b = data_registry.create_data(
        DataDescription{
            DataId{301},
            "Data B",
        });
    GERDOS_CHECK(data_b != nullptr);

    GERDOS_CHECK(data_b->add_residency(
        DataResidency{
            DataResidencyDescription{
                DataResidencyId{410},
                DataId{301},
                memory_ref,
                "stale-input",
            },
        }));

    auto* stale_input = data_b->find_residency(DataResidencyId{410});
    GERDOS_CHECK(stale_input != nullptr);
    GERDOS_CHECK(stale_input->set_state(DataResidencyState::VALID));
    GERDOS_CHECK(stale_input->set_state(DataResidencyState::STALE));

    Data* data_c = data_registry.create_data(
        DataDescription{
            DataId{302},
            "Data C",
        });
    GERDOS_CHECK(data_c != nullptr);

    GERDOS_CHECK(data_c->add_residency(
        DataResidency{
            DataResidencyDescription{
                DataResidencyId{420},
                DataId{302},
                memory_ref,
                "transferring-input",
            },
        }));

    auto* transferring_input =
        data_c->find_residency(DataResidencyId{420});
    GERDOS_CHECK(transferring_input != nullptr);
    GERDOS_CHECK(
        transferring_input->set_state(DataResidencyState::VALID));
    GERDOS_CHECK(
        transferring_input->set_state(DataResidencyState::TRANSFERRING));

    Data* data_d = data_registry.create_data(
        DataDescription{
            DataId{303},
            "Data D",
        });
    GERDOS_CHECK(data_d != nullptr);

    GERDOS_CHECK(data_d->add_residency(
        DataResidency{
            DataResidencyDescription{
                DataResidencyId{430},
                DataId{303},
                memory_ref,
                "fresh-input",
            },
        }));

    auto* fresh_input = data_d->find_residency(DataResidencyId{430});
    GERDOS_CHECK(fresh_input != nullptr);

    ExecutionAdmissionValidator admission_validator(
        devices,
        data_registry);

    // ---------------------------------------------------------------------
    // 2. Binding establishment requirements
    // ---------------------------------------------------------------------

    // No binding cannot be admitted.
    Execution unbound(
        ExecutionDescription{
            ExecutionId{500},
            OperationId{600},
        });

    GERDOS_CHECK(!admission_validator.admit(unbound).has_value());

    // Empty binding is structurally valid and fully resolved by 8F/8E,
    // but is not an executable physical realization.
    Execution empty_binding(
        ExecutionDescription{
            ExecutionId{501},
            OperationId{601},
        });

    GERDOS_CHECK(empty_binding.bind(PhysicalBinding{}));
    GERDOS_CHECK(!admission_validator.admit(empty_binding).has_value());

    // ---------------------------------------------------------------------
    // 3. A valid, resolvable, usable binding is admitted
    // ---------------------------------------------------------------------

    Execution valid(
        ExecutionDescription{
            ExecutionId{502},
            OperationId{602},
        });

    PhysicalBinding valid_binding;
    valid_binding.data.push_back(
        DataBinding{
            DataBindingRole::INPUT,
            DataResidencyRef{
                DataId{300},
                DataResidencyId{400},
            },
        });

    valid_binding.resources.push_back(
        ResourceBinding{
            ResourceBindingRole::COMPUTE,
            compute_ref,
        });

    GERDOS_CHECK(valid.bind(std::move(valid_binding)));

    const auto admission = admission_validator.admit(valid);

    GERDOS_CHECK(admission.has_value());
    GERDOS_CHECK(admission->execution() == ExecutionId{502});

    // Admission is observational: it does not change execution state.
    GERDOS_CHECK(valid.state() == ExecutionState::PENDING);
    GERDOS_CHECK(valid.has_binding());

    // Admission is what makes the RUNNING transition possible.
    GERDOS_CHECK(valid.admitted());
    GERDOS_CHECK(valid.set_state(ExecutionState::RUNNING));

    // ---------------------------------------------------------------------
    // 4. Consuming roles require a usable residency
    // ---------------------------------------------------------------------

    // A stale consuming residency must not be admitted.
    Execution stale_consumer(
        ExecutionDescription{
            ExecutionId{503},
            OperationId{603},
        });

    PhysicalBinding stale_binding;
    stale_binding.data.push_back(
        DataBinding{
            DataBindingRole::INPUT,
            DataResidencyRef{
                DataId{301},
                DataResidencyId{410},
            },
        });

    stale_binding.resources.push_back(
        ResourceBinding{
            ResourceBindingRole::COMPUTE,
            compute_ref,
        });

    GERDOS_CHECK(stale_consumer.bind(std::move(stale_binding)));
    GERDOS_CHECK(
        !admission_validator.admit(stale_consumer).has_value());

    // A transferring consuming residency must not be admitted.
    Execution transferring_consumer(
        ExecutionDescription{
            ExecutionId{504},
            OperationId{604},
        });

    PhysicalBinding transferring_binding;
    transferring_binding.data.push_back(
        DataBinding{
            DataBindingRole::SOURCE,
            DataResidencyRef{
                DataId{302},
                DataResidencyId{420},
            },
        });

    transferring_binding.resources.push_back(
        ResourceBinding{
            ResourceBindingRole::TRANSFER,
            compute_ref,
        });

    GERDOS_CHECK(
        transferring_consumer.bind(std::move(transferring_binding)));
    GERDOS_CHECK(
        !admission_validator.admit(transferring_consumer).has_value());

    // A fresh, not yet usable consuming residency must not be admitted.
    Execution fresh_consumer(
        ExecutionDescription{
            ExecutionId{505},
            OperationId{605},
        });

    PhysicalBinding fresh_binding;
    fresh_binding.data.push_back(
        DataBinding{
            DataBindingRole::INPUT,
            DataResidencyRef{
                DataId{303},
                DataResidencyId{430},
            },
        });

    fresh_binding.resources.push_back(
        ResourceBinding{
            ResourceBindingRole::COMPUTE,
            compute_ref,
        });

    GERDOS_CHECK(fresh_consumer.bind(std::move(fresh_binding)));
    GERDOS_CHECK(
        !admission_validator.admit(fresh_consumer).has_value());

    // ---------------------------------------------------------------------
    // 5. Producing roles do not require residency usability
    // ---------------------------------------------------------------------

    // A fresh output residency is about to be created or rewritten and is
    // therefore admitted while still UNAVAILABLE.
    Execution producer(
        ExecutionDescription{
            ExecutionId{506},
            OperationId{606},
        });

    PhysicalBinding producer_binding;
    producer_binding.data.push_back(
        DataBinding{
            DataBindingRole::OUTPUT,
            DataResidencyRef{
                DataId{300},
                DataResidencyId{401},
            },
        });

    producer_binding.resources.push_back(
        ResourceBinding{
            ResourceBindingRole::COMPUTE,
            compute_ref,
        });

    GERDOS_CHECK(producer.bind(std::move(producer_binding)));

    const auto producer_admission =
        admission_validator.admit(producer);

    GERDOS_CHECK(producer_admission.has_value());
    GERDOS_CHECK(
        producer_admission->execution() == ExecutionId{506});

    // ---------------------------------------------------------------------
    // 6. Referenced resources must be available
    // ---------------------------------------------------------------------

    // A binding onto an unavailable compute resource is rejected.
    compute->set_availability(ResourceAvailability::FAILED);

    Execution failed_compute(
        ExecutionDescription{
            ExecutionId{507},
            OperationId{607},
        });

    PhysicalBinding failed_compute_binding;
    failed_compute_binding.data.push_back(
        DataBinding{
            DataBindingRole::INPUT,
            DataResidencyRef{
                DataId{300},
                DataResidencyId{400},
            },
        });

    failed_compute_binding.resources.push_back(
        ResourceBinding{
            ResourceBindingRole::COMPUTE,
            compute_ref,
        });

    GERDOS_CHECK(
        failed_compute.bind(std::move(failed_compute_binding)));
    GERDOS_CHECK(
        !admission_validator.admit(failed_compute).has_value());

    compute->set_availability(ResourceAvailability::AVAILABLE);

    // A residency hosted on an unavailable resource is not effectively
    // usable, even in a producing role.
    memory->set_availability(ResourceAvailability::DRAINING);

    Execution draining_host(
        ExecutionDescription{
            ExecutionId{508},
            OperationId{608},
        });

    PhysicalBinding draining_host_binding;
    draining_host_binding.data.push_back(
        DataBinding{
            DataBindingRole::OUTPUT,
            DataResidencyRef{
                DataId{300},
                DataResidencyId{401},
            },
        });

    draining_host_binding.resources.push_back(
        ResourceBinding{
            ResourceBindingRole::COMPUTE,
            compute_ref,
        });

    GERDOS_CHECK(
        draining_host.bind(std::move(draining_host_binding)));
    GERDOS_CHECK(
        !admission_validator.admit(draining_host).has_value());

    memory->set_availability(ResourceAvailability::AVAILABLE);

    // ---------------------------------------------------------------------
    // 7. Structural and resolution requirements remain enforced
    // ---------------------------------------------------------------------

    // Structurally invalid references are rejected before resolution.
    Execution malformed(
        ExecutionDescription{
            ExecutionId{509},
            OperationId{609},
        });

    PhysicalBinding malformed_binding;
    malformed_binding.resources.push_back(
        ResourceBinding{
            ResourceBindingRole::COMPUTE,
            ResourceRef{
                DeviceId{},
                ResourceId{200},
            },
        });

    GERDOS_CHECK(malformed.bind(std::move(malformed_binding)));
    GERDOS_CHECK(!admission_validator.admit(malformed).has_value());

    // Structurally valid but unresolved references are rejected.
    Execution unresolved(
        ExecutionDescription{
            ExecutionId{510},
            OperationId{610},
        });

    PhysicalBinding unresolved_binding;
    unresolved_binding.resources.push_back(
        ResourceBinding{
            ResourceBindingRole::COMPUTE,
            ResourceRef{
                DeviceId{999},
                ResourceId{999},
            },
        });

    GERDOS_CHECK(unresolved.bind(std::move(unresolved_binding)));
    GERDOS_CHECK(!admission_validator.admit(unresolved).has_value());

    // ---------------------------------------------------------------------
    // 8. Admission applies to PENDING executions only
    // ---------------------------------------------------------------------

    Execution running(
        ExecutionDescription{
            ExecutionId{511},
            OperationId{611},
        });

    PhysicalBinding running_binding;
    running_binding.resources.push_back(
        ResourceBinding{
            ResourceBindingRole::COMPUTE,
            compute_ref,
        });

    GERDOS_CHECK(running.bind(std::move(running_binding)));
    GERDOS_CHECK(admission_validator.admit(running).has_value());
    GERDOS_CHECK(running.set_state(ExecutionState::RUNNING));
    GERDOS_CHECK(!admission_validator.admit(running).has_value());

    // ---------------------------------------------------------------------
    // 9. Admission evidence is execution-scoped
    // ---------------------------------------------------------------------

    Execution other(
        ExecutionDescription{
            ExecutionId{512},
            OperationId{612},
        });

    PhysicalBinding other_binding;
    other_binding.resources.push_back(
        ResourceBinding{
            ResourceBindingRole::COMPUTE,
            compute_ref,
        });

    GERDOS_CHECK(other.bind(std::move(other_binding)));

    const auto other_admission = admission_validator.admit(other);

    GERDOS_CHECK(other_admission.has_value());
    GERDOS_CHECK(other_admission->execution() == ExecutionId{512});
    GERDOS_CHECK(
        admission->execution() != other_admission->execution());

    // ---------------------------------------------------------------------
    // 10. Admission is mechanism, not convention
    // ---------------------------------------------------------------------

    // Evidence tokens cannot be fabricated.
    static_assert(
        !std::is_constructible_v<ExecutionAdmission, ExecutionId>);
    static_assert(
        !std::is_default_constructible_v<ExecutionAdmission>);

    // The RUNNING transition is rejected without admission evidence, even
    // for a well-formed, fully resolvable binding.
    Execution unadmitted(
        ExecutionDescription{
            ExecutionId{513},
            OperationId{613},
        });

    PhysicalBinding unadmitted_binding;
    unadmitted_binding.data.push_back(
        DataBinding{
            DataBindingRole::INPUT,
            DataResidencyRef{
                DataId{300},
                DataResidencyId{400},
            },
        });

    unadmitted_binding.resources.push_back(
        ResourceBinding{
            ResourceBindingRole::COMPUTE,
            compute_ref,
        });

    GERDOS_CHECK(unadmitted.bind(std::move(unadmitted_binding)));
    GERDOS_CHECK(!unadmitted.admitted());
    GERDOS_CHECK(!unadmitted.set_state(ExecutionState::RUNNING));
    GERDOS_CHECK(unadmitted.state() == ExecutionState::PENDING);

    // Admission establishes the evidence and the transition succeeds.
    GERDOS_CHECK(
        admission_validator.admit(unadmitted).has_value());
    GERDOS_CHECK(unadmitted.admitted());
    GERDOS_CHECK(unadmitted.set_state(ExecutionState::RUNNING));
    GERDOS_CHECK(unadmitted.state() == ExecutionState::RUNNING);

    // Rejected admission never establishes evidence.
    Execution never_admitted(
        ExecutionDescription{
            ExecutionId{514},
            OperationId{614},
        });

    GERDOS_CHECK(!admission_validator.admit(never_admitted).has_value());
    GERDOS_CHECK(!never_admitted.admitted());
    GERDOS_CHECK(
        !never_admitted.set_state(ExecutionState::RUNNING));

    return 0;
}