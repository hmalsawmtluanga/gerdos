#include "test_check.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "gerdos/core/binding_admissibility.hpp"
#include "gerdos/core/binding_planner.hpp"
#include "gerdos/core/execution_admission.hpp"
#include "gerdos/core/physical_binding_resolver.hpp"
#include "gerdos/core/physical_binding_validation.hpp"
#include "gerdos/core/executor.hpp"
#include "gerdos/core/operation_registry.hpp"
#include "gerdos/core/workload_artifact.hpp"
#include "gerdos/cpu/cpu_backend.hpp"

namespace {

using namespace gerdos;

struct Machine {
    DeviceRegistry devices;
    DataRegistry data;
    OperationRegistry operations;
    ExecutionRegistry executions;
    Topology topology;
    MeasurementRegistry measurements;

    Machine() {
        auto* host = devices.create_device(
            DeviceDescription{DeviceId{100}, "host"});
        (void)host->add_resource(
            Resource{ResourceDescription{
                ResourceId{101}, DeviceId{100}, ResourceKind::MEMORY, "ram"}});
        host->find_resource(ResourceId{101})->set_availability(
            ResourceAvailability::AVAILABLE);
        (void)host->add_resource(
            Resource{ResourceDescription{
                ResourceId{102}, DeviceId{100}, ResourceKind::COMPUTE, "compute"}});
        host->find_resource(ResourceId{102})->set_availability(
            ResourceAvailability::AVAILABLE);
    }
};

std::string load_text(const char* path) {
    std::ifstream input(path);
    GERDOS_CHECK(input.good());
    std::stringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

} // namespace

int main(int argc, char** argv) {
    using namespace gerdos;

    GERDOS_CHECK(argc == 2);
    const std::string directory{argv[1]};

    Machine machine;
    CpuBackend backend;
    Executor executor(
        machine.executions,
        machine.operations,
        machine.devices,
        machine.data,
        backend,
        &machine.measurements);
    BindingPlanner planner(
        machine.devices,
        machine.data,
        machine.topology,
        machine.measurements);

    // ---------------------------------------------------------------------
    // 1. The checked-in artifact loads and executes end to end
    // ---------------------------------------------------------------------

    {
        const auto parsed = parse_artifact(load_text((directory + "/softmax_prefix.gwd").c_str()));
        GERDOS_CHECK(parsed.ok);
        GERDOS_CHECK(parsed.artifact.workload.name == "softmax-prefix");
        GERDOS_CHECK(parsed.artifact.workload.operations.size() == 3);
        GERDOS_CHECK(parsed.artifact.data.size() == 4);

        std::string reason;
        GERDOS_CHECK(populate_registries(
            parsed.artifact, machine.devices, machine.data,
            machine.operations, reason));

        const auto planned = planner.plan_attempts(machine.operations);
        GERDOS_CHECK(planned.size() == 3);

        std::size_t attempt = 0;

        for (const auto& step : planned) {
            const auto* operation = machine.operations.find_operation(step.id);
            GERDOS_CHECK(operation != nullptr);
            PhysicalBindingValidator validator;
            BindingResolver resolver(machine.devices, machine.data);
            BindingAdmissibilityValidator admissibility;
            ExecutionAdmissionValidator admission(machine.devices, machine.data);
            GERDOS_CHECK(validator.validate(step.binding));
            GERDOS_CHECK(resolver.resolve(step.binding).fully_resolved());
            GERDOS_CHECK(admissibility.admissible(*operation, step.binding));
            auto* execution = machine.executions.create_execution(
                ExecutionDescription{ExecutionId{3000 + attempt}, step.id});
            ++attempt;
            GERDOS_CHECK(execution->bind(step.binding));
            GERDOS_CHECK(admission.admit(*execution).has_value());
            GERDOS_CHECK(executor.start(execution->description().id));
            std::vector<AttemptStatus> outcomes;
            executor.advance(outcomes);

            while (outcomes.empty()) {
                executor.advance(outcomes);
            }

            GERDOS_CHECK(outcomes.size() == 1);
            GERDOS_CHECK(outcomes.front().integrity == AttemptIntegrity::COHERENT);
        }

        // Closed forms: T = [6,1,1,1,1,1], E = 1 + exp(T), S = sum(E).
        // The sum accumulates left-to-right in F32 (not the regrouped
        // closed form), so it compares within one ulp, printed first.
        const float e6 = std::exp(6.0f);
        const float e1 = std::exp(1.0f);
        const float t0 = backend.sample(DataResidencyRef{DataId{701}, DataResidencyId{7002}}, 0);
        const float e0 = backend.sample(DataResidencyRef{DataId{702}, DataResidencyId{7003}}, 0);
        const float e1v = backend.sample(DataResidencyRef{DataId{702}, DataResidencyId{7003}}, 1);
        const float sum = backend.sample(DataResidencyRef{DataId{703}, DataResidencyId{7004}}, 0);
        std::printf("artifact chain: T0=%f E0=%f E1=%f S=%f\n", t0, e0, e1v, sum);
        std::printf("expected:       6.0 %f %f %f\n", 1.0f + e6, 1.0f + e1, (1.0f + e6) + 5.0f * (1.0f + e1));
        std::fflush(stdout);
        GERDOS_CHECK(t0 == 6.0f);
        GERDOS_CHECK(e0 == 1.0f + e6);
        GERDOS_CHECK(e1v == 1.0f + e1);
        const float expected_sum = (1.0f + e6) + 5.0f * (1.0f + e1);
        GERDOS_CHECK(
            sum == expected_sum ||
            std::fabs(sum - expected_sum) <=
                std::numeric_limits<float>::epsilon() * expected_sum * 4);

        // Evidence-growth check (Phase 4 thesis guard): the log holds
        // exactly the three successful compute observations — no more,
        // no fewer. Growth is accounted per attempt, deterministically.
        const auto evidence = machine.measurements.summarize(
            ResourceRef{DeviceId{100}, ResourceId{102}},
            MeasurementQuantity::DURATION_NS);
        std::printf("evidence: %llu successful observations\n", (unsigned long long)evidence.succeeded_observations);
        std::fflush(stdout);
        GERDOS_CHECK(evidence.succeeded_observations == 3);
    }

    // ---------------------------------------------------------------------
    // 2. Fail-closed parsing: every corruption refused with its line
    // ---------------------------------------------------------------------

    {
        const char* cases[] = {
            "workload x\n",
            "version 2\nworkload x\n",
            "version 1\nversion 1\n",
            "version 1\nworkload x\nbogus 1 2\n",
            "version 1\nworkload x\ndata 1\n",
            "version 1\nworkload x\ndata 1 a\ndata 1 b\n",
            "version 1\nworkload x\ndata 1 a\nresidency 1 1 1 1\n",
            "version 1\nworkload x\ndata 1 a\nresidency 9 1 100 101 r\n",
            "version 1\nworkload x\nusable 1 1\n",
            "version 1\nworkload x\ndata 1 a\nresidency 1 1 100 101 r\nusable 1 2\n",
            "version 1\nworkload x\nop 1 inputs=- outputs=- deps=- req=COMPUTE:1 form=BOGUS dtype=F32 elements=1 passes=1 ds=0 ss=1 c=0\n",
            "version 1\nworkload x\nop 1 inputs=- outputs=- deps=- req=COMPUTE:1 form=AFFINE dtype=F64 elements=1 passes=1 ds=0 ss=1 c=0\n",
            "version 1\nworkload x\nop 1 inputs=- outputs=- deps=- req=FLY:1 form=AFFINE dtype=F32 elements=1 passes=1 ds=0 ss=1 c=0\n",
            "version 1\nworkload x\nop 1 inputs=- outputs=- deps=- req=COMPUTE:0 form=AFFINE dtype=F32 elements=1 passes=1 ds=0 ss=1 c=0\n",
            "version 1\nworkload x\nop 1 inputs=- outputs=- deps=- req=COMPUTE:1 form=AFFINE dtype=F32 elements=0 passes=1 ds=0 ss=1 c=0\n",
            "version 1\nworkload x\nop 1 inputs=- outputs=- deps=- req=COMPUTE:1 form=AFFINE dtype=F32 elements=1 passes=1 ds=zero ss=1 c=0\n",
            "version 1\nworkload x\nop 1 inputs=- outputs=- deps=- req=COMPUTE:1 form=MATRIX_PRODUCT dtype=F32 elements=1 passes=1 ds=0 ss=1 c=0\n",
            "version 1\nworkload x\nop 1 inputs=- outputs=- deps=- form=AFFINE dtype=F32 elements=1 passes=1 ds=0 ss=1 c=0\n",
            "version 1\nworkload x\nop 1 inputs=- outputs=- deps=- req=COMPUTE:1 form=AFFINE dtype=F32 elements=1 passes=1 ds=0 ss=1 c=0\nop 1 inputs=- outputs=- deps=- req=COMPUTE:1 form=AFFINE dtype=F32 elements=1 passes=1 ds=0 ss=1 c=0\n",
            "version 1\n",
        };

        for (const char* text : cases) {
            const auto parsed = parse_artifact(text);
            GERDOS_CHECK(!parsed.ok);
            GERDOS_CHECK(parsed.error.line >= 1);
            GERDOS_CHECK(!parsed.error.reason.empty());
        }
    }

    return 0;
}

