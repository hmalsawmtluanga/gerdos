#include <cassert>

#include "gerdos/core/physical_binding_validation.hpp"

namespace {

using namespace gerdos;

void test_empty_binding_is_valid() {
    PhysicalBindingValidator validator;
    const PhysicalBinding binding{};

    assert(validator.validate(binding));
}

void test_valid_data_roles() {
    PhysicalBindingValidator validator;

    for (const auto role : {
             DataBindingRole::INPUT,
             DataBindingRole::OUTPUT,
             DataBindingRole::SOURCE,
             DataBindingRole::DESTINATION,
         }) {
        PhysicalBinding binding;
        binding.data.push_back(
            DataBinding{
                role,
                DataResidencyRef{
                    DataId{1},
                    DataResidencyId{1},
                },
            });

        assert(validator.validate(binding));
    }
}

void test_valid_resource_roles() {
    PhysicalBindingValidator validator;

    for (const auto role : {
             ResourceBindingRole::COMPUTE,
             ResourceBindingRole::TRANSFER,
         }) {
        PhysicalBinding binding;
        binding.resources.push_back(
            ResourceBinding{
                role,
                ResourceRef{
                    DeviceId{1},
                    ResourceId{1},
                },
            });

        assert(validator.validate(binding));
    }
}

void test_multiple_same_role_bindings_are_valid() {
    PhysicalBindingValidator validator;

    PhysicalBinding binding;
    binding.data.push_back(
        DataBinding{
            DataBindingRole::INPUT,
            DataResidencyRef{
                DataId{1},
                DataResidencyId{1},
            },
        });
    binding.data.push_back(
        DataBinding{
            DataBindingRole::INPUT,
            DataResidencyRef{
                DataId{2},
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
    binding.resources.push_back(
        ResourceBinding{
            ResourceBindingRole::COMPUTE,
            ResourceRef{
                DeviceId{2},
                ResourceId{2},
            },
        });

    assert(validator.validate(binding));
}

void test_mixed_data_and_resource_bindings_are_valid() {
    PhysicalBindingValidator validator;

    PhysicalBinding binding;
    binding.data.push_back(
        DataBinding{
            DataBindingRole::INPUT,
            DataResidencyRef{
                DataId{1},
                DataResidencyId{1},
            },
        });
    binding.resources.push_back(
        ResourceBinding{
            ResourceBindingRole::TRANSFER,
            ResourceRef{
                DeviceId{2},
                ResourceId{3},
            },
        });

    assert(validator.validate(binding));
}

void test_zero_data_id_is_invalid() {
    PhysicalBindingValidator validator;

    PhysicalBinding binding;
    binding.data.push_back(
        DataBinding{
            DataBindingRole::INPUT,
            DataResidencyRef{
                DataId{},
                DataResidencyId{1},
            },
        });

    assert(!validator.validate(binding));
}

void test_zero_data_residency_id_is_invalid() {
    PhysicalBindingValidator validator;

    PhysicalBinding binding;
    binding.data.push_back(
        DataBinding{
            DataBindingRole::INPUT,
            DataResidencyRef{
                DataId{1},
                DataResidencyId{},
            },
        });

    assert(!validator.validate(binding));
}

void test_zero_device_id_is_invalid() {
    PhysicalBindingValidator validator;

    PhysicalBinding binding;
    binding.resources.push_back(
        ResourceBinding{
            ResourceBindingRole::COMPUTE,
            ResourceRef{
                DeviceId{},
                ResourceId{1},
            },
        });

    assert(!validator.validate(binding));
}

void test_zero_resource_id_is_invalid() {
    PhysicalBindingValidator validator;

    PhysicalBinding binding;
    binding.resources.push_back(
        ResourceBinding{
            ResourceBindingRole::COMPUTE,
            ResourceRef{
                DeviceId{1},
                ResourceId{},
            },
        });

    assert(!validator.validate(binding));
}

void test_invalid_data_role_is_invalid() {
    PhysicalBindingValidator validator;

    PhysicalBinding binding;
    binding.data.push_back(
        DataBinding{
            static_cast<DataBindingRole>(255),
            DataResidencyRef{
                DataId{1},
                DataResidencyId{1},
            },
        });

    assert(!validator.validate(binding));
}

void test_invalid_resource_role_is_invalid() {
    PhysicalBindingValidator validator;

    PhysicalBinding binding;
    binding.resources.push_back(
        ResourceBinding{
            static_cast<ResourceBindingRole>(255),
            ResourceRef{
                DeviceId{1},
                ResourceId{1},
            },
        });

    assert(!validator.validate(binding));
}

} // namespace

int main() {
    test_empty_binding_is_valid();
    test_valid_data_roles();
    test_valid_resource_roles();
    test_multiple_same_role_bindings_are_valid();
    test_mixed_data_and_resource_bindings_are_valid();

    test_zero_data_id_is_invalid();
    test_zero_data_residency_id_is_invalid();
    test_zero_device_id_is_invalid();
    test_zero_resource_id_is_invalid();
    test_invalid_data_role_is_invalid();
    test_invalid_resource_role_is_invalid();

    return 0;
}
