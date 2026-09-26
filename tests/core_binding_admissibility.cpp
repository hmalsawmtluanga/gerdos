#include "test_check.hpp"

#include "gerdos/core/binding_admissibility.hpp"
#include "gerdos/core/operation.hpp"

namespace {

using namespace gerdos;

PhysicalBinding compute_realization() {
    PhysicalBinding binding;
    binding.data.push_back(
        DataBinding{
            DataBindingRole::INPUT,
            DataResidencyRef{
                DataId{10},
                DataResidencyId{1},
            },
        });

    binding.data.push_back(
        DataBinding{
            DataBindingRole::OUTPUT,
            DataResidencyRef{
                DataId{20},
                DataResidencyId{2},
            },
        });

    binding.resources.push_back(
        ResourceBinding{
            ResourceBindingRole::COMPUTE,
            ResourceRef{
                DeviceId{1},
                ResourceId{1},
            },
        });

    return binding;
}

} // namespace

int main() {
    using namespace gerdos;

    BindingAdmissibilityValidator validator;

    // ---------------------------------------------------------------------
    // 1. A well-formed compute realization is admissible
    // ---------------------------------------------------------------------

    const Operation compute_operation{OperationDescription{
        OperationId{100},
        {DataId{10}},
        {DataId{20}},
        {},
        {
            ResourceRequirement{
                ResourceBindingRole::COMPUTE,
                1,
            },
        },
    }};

    GERDOS_CHECK(
        validator.admissible(compute_operation, compute_realization()));

    // ---------------------------------------------------------------------
    // 2. Data role consistency
    // ---------------------------------------------------------------------

    // A consuming entry for undeclared data is inadmissible.
    {
        PhysicalBinding binding = compute_realization();
        binding.data.push_back(
            DataBinding{
                DataBindingRole::INPUT,
                DataResidencyRef{
                    DataId{99},
                    DataResidencyId{3},
                },
            });

        GERDOS_CHECK(!validator.admissible(compute_operation, binding));
    }

    // A producing entry for a declared input is inadmissible.
    {
        PhysicalBinding binding = compute_realization();
        binding.data.push_back(
            DataBinding{
                DataBindingRole::OUTPUT,
                DataResidencyRef{
                    DataId{10},
                    DataResidencyId{3},
                },
            });

        GERDOS_CHECK(!validator.admissible(compute_operation, binding));
    }

    // A consuming entry for a declared output is inadmissible.
    {
        PhysicalBinding binding = compute_realization();
        binding.data.push_back(
            DataBinding{
                DataBindingRole::SOURCE,
                DataResidencyRef{
                    DataId{20},
                    DataResidencyId{3},
                },
            });

        GERDOS_CHECK(!validator.admissible(compute_operation, binding));
    }

    // A role value outside the data binding role domain is inadmissible.
    {
        PhysicalBinding binding = compute_realization();
        binding.data.push_back(
            DataBinding{
                static_cast<DataBindingRole>(255),
                DataResidencyRef{
                    DataId{10},
                    DataResidencyId{3},
                },
            });

        GERDOS_CHECK(!validator.admissible(compute_operation, binding));
    }

    // ---------------------------------------------------------------------
    // 3. Declared references must be covered
    // ---------------------------------------------------------------------

    // A declared input without a consuming entry is inadmissible.
    {
        PhysicalBinding binding = compute_realization();
        binding.data.erase(binding.data.begin());

        GERDOS_CHECK(!validator.admissible(compute_operation, binding));
    }

    // A declared output without a producing entry is inadmissible.
    {
        PhysicalBinding binding = compute_realization();
        binding.data.erase(binding.data.begin() + 1);

        GERDOS_CHECK(!validator.admissible(compute_operation, binding));
    }

    // ---------------------------------------------------------------------
    // 4. Resource requirements
    // ---------------------------------------------------------------------

    // A missing resource requirement role is inadmissible.
    {
        PhysicalBinding binding = compute_realization();
        binding.resources.clear();
        binding.resources.push_back(
            ResourceBinding{
                ResourceBindingRole::TRANSFER,
                ResourceRef{
                    DeviceId{1},
                    ResourceId{2},
                },
            });

        GERDOS_CHECK(!validator.admissible(compute_operation, binding));
    }

    // Declared minimums are enforced.
    const Operation two_compute_operation{OperationDescription{
        OperationId{101},
        {DataId{10}},
        {DataId{20}},
        {},
        {
            ResourceRequirement{
                ResourceBindingRole::COMPUTE,
                2,
            },
        },
    }};

    GERDOS_CHECK(
        !validator.admissible(two_compute_operation, compute_realization()));

    {
        PhysicalBinding binding = compute_realization();
        binding.resources.push_back(
            ResourceBinding{
                ResourceBindingRole::COMPUTE,
                ResourceRef{
                    DeviceId{1},
                    ResourceId{2},
                },
            });

        GERDOS_CHECK(
            validator.admissible(two_compute_operation, binding));
    }

    // Bindings beyond the declared requirements are permitted.
    {
        PhysicalBinding binding = compute_realization();
        binding.resources.push_back(
            ResourceBinding{
                ResourceBindingRole::TRANSFER,
                ResourceRef{
                    DeviceId{1},
                    ResourceId{3},
                },
            });

        GERDOS_CHECK(validator.admissible(compute_operation, binding));
    }

    // An invalid requirement is inadmissible.
    const Operation invalid_requirement_operation{OperationDescription{
        OperationId{102},
        {},
        {},
        {},
        {
            ResourceRequirement{
                ResourceBindingRole::COMPUTE,
                0,
            },
        },
    }};

    GERDOS_CHECK(
        !validator.admissible(invalid_requirement_operation, PhysicalBinding{}));

    // ---------------------------------------------------------------------
    // 5. A movement realization for data used as both input and output
    // ---------------------------------------------------------------------

    const Operation movement_operation{OperationDescription{
        OperationId{103},
        {DataId{10}},
        {DataId{10}},
        {},
        {
            ResourceRequirement{
                ResourceBindingRole::TRANSFER,
                1,
            },
        },
    }};

    PhysicalBinding movement_binding;
    movement_binding.data.push_back(
        DataBinding{
            DataBindingRole::SOURCE,
            DataResidencyRef{
                DataId{10},
                DataResidencyId{1},
            },
        });

    movement_binding.data.push_back(
        DataBinding{
            DataBindingRole::DESTINATION,
            DataResidencyRef{
                DataId{10},
                DataResidencyId{2},
            },
        });

    movement_binding.resources.push_back(
        ResourceBinding{
            ResourceBindingRole::TRANSFER,
            ResourceRef{
                DeviceId{1},
                ResourceId{1},
            },
        });

    GERDOS_CHECK(
        validator.admissible(movement_operation, movement_binding));

    // The same operation without its producing entry is inadmissible.
    {
        PhysicalBinding binding = movement_binding;
        binding.data.pop_back();

        GERDOS_CHECK(!validator.admissible(movement_operation, binding));
    }

    // ---------------------------------------------------------------------
    // 6. Operations without requirements place no demand
    // ---------------------------------------------------------------------

    const Operation unconstrained_operation{OperationDescription{
        OperationId{104},
        {},
        {},
        {},
        {},
    }};

    GERDOS_CHECK(
        validator.admissible(unconstrained_operation, PhysicalBinding{}));

    {
        PhysicalBinding binding;
        binding.resources.push_back(
            ResourceBinding{
                ResourceBindingRole::COMPUTE,
                ResourceRef{
                    DeviceId{1},
                    ResourceId{1},
                },
            });

        GERDOS_CHECK(
            validator.admissible(unconstrained_operation, binding));
    }

    // An undeclared data binding remains inadmissible without requirements.
    {
        PhysicalBinding binding;
        binding.data.push_back(
            DataBinding{
                DataBindingRole::INPUT,
                DataResidencyRef{
                    DataId{10},
                    DataResidencyId{1},
                },
            });

        GERDOS_CHECK(
            !validator.admissible(unconstrained_operation, binding));
    }

    return 0;
}