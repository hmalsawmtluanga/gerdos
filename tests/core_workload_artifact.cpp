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
#include "gerdos/adapters/model_adapter.hpp"
#include "gerdos/adapters/signal_adapter.hpp"
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
            "version 1\nworkload x\nop 1 inputs=- outputs=- deps=- req=COMPUTE:1 form=AFFINE dtype=F32 elements=1 passes=1 ds=nan ss=1 c=0\n",
            "version 1\nworkload x\nop 1 inputs=- outputs=- deps=- req=COMPUTE:1 form=AFFINE dtype=F32 elements=1 passes=1 ds=inf ss=1 c=0\n",
            "version 1\nworkload x\nop 1 inputs=- outputs=- deps=- req=COMPUTE:1 form=AFFINE dtype=F32 elements=1 passes=1 ds=0x1p3 ss=1 c=0\n",
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

        // Decimal exponents stay accepted under the strict grammar.
        const auto exponent =
            parse_artifact("version 1\nworkload x\nop 1 inputs=- outputs=- deps=- req=COMPUTE:1 form=AFFINE dtype=F32 elements=1 passes=1 ds=1e3 ss=1 c=0\n");
        GERDOS_CHECK(exponent.ok);

        // Text-format law: CRLF parses exactly like its LF twin, a lone
        // CR line is blank, and a CR anywhere else is refused loudly.
        const auto crlf = parse_artifact(
            "version 1\r\n\r\nworkload x\r\nop 1 inputs=- outputs=- deps=- req=COMPUTE:1 form=AFFINE dtype=F32 elements=1 passes=1 ds=1e3 ss=1 c=0\r\n");
        GERDOS_CHECK(crlf.ok);
        GERDOS_CHECK(crlf.artifact.workload.operations.size() == 1);

        const auto stray_cr = parse_artifact(
            "versio\rn 1\nworkload x\nop 1 inputs=- outputs=- deps=- req=COMPUTE:1 form=AFFINE dtype=F32 elements=1 passes=1 ds=1e3 ss=1 c=0\n");
        GERDOS_CHECK(!stray_cr.ok);
    }

    // ---------------------------------------------------------------------
    // 3. Weight-scale pressure stages through capacity-fitting rounds
    // ---------------------------------------------------------------------

    {
        // Four 1 MiB weight records against a 2 MiB ram budget: the full
        // plan cannot fit, so the workload executes in sequenced rounds
        // with explicit eviction between them. Each round is planned,
        // admitted, executed, and measured independently; eviction
        // preserves every logical Data; over-budget single operations
        // would still refuse (none exists here — each op needs 1 MiB).
        Machine pressured;
        pressured.devices.find_device(DeviceId{100})
            ->find_resource(ResourceId{101})
            ->set_capacity(2u << 20);

        CpuBackend pressured_backend;
        Executor pressured_executor(
            pressured.executions,
            pressured.operations,
            pressured.devices,
            pressured.data,
            pressured_backend,
            &pressured.measurements);
        BindingPlanner pressured_planner(
            pressured.devices,
            pressured.data,
            pressured.topology,
            pressured.measurements);

        const auto pressured_parsed = parse_artifact(
            load_text((directory + "/staged_weights.gwd").c_str()));
        GERDOS_CHECK(pressured_parsed.ok);
        GERDOS_CHECK(
            pressured_parsed.artifact.workload.operations.size() == 4);

        std::string pressured_reason;
        GERDOS_CHECK(populate_registries(
            pressured_parsed.artifact, pressured.devices, pressured.data,
            pressured.operations, pressured_reason));

        // The whole workload does not fit: capacity admits only a
        // prefix. Rounds sequence explicitly: plan, execute, evict.
        const auto full_plan =
            pressured_planner.plan_attempts(pressured.operations);
        GERDOS_CHECK(full_plan.size() < 4);
        GERDOS_CHECK(!full_plan.empty());

        // Round structure: one op per round (each needs a full 1 MiB
        // against the 2 MiB budget shared with its output record... in
        // fact each reduce writes one scalar into a fresh record, so
        // demand per op is input 1 MiB + output scalar; two consecutive
        // rounds' inputs (2 MiB) exactly fill the budget).
        std::size_t rounds = 0;
        std::size_t executed = 0;
        std::size_t base_id = 4000;
        std::vector<std::pair<DataId, DataResidencyId>> evicted;

        while (executed < 4) {
            const auto round =
                pressured_planner.plan_attempts(pressured.operations);
            GERDOS_CHECK(!round.empty());
            GERDOS_CHECK(round.size() <= 2);
            ++rounds;

            for (const auto& step : round) {
                const auto* operation =
                    pressured.operations.find_operation(step.id);
                GERDOS_CHECK(operation != nullptr);
                PhysicalBindingValidator validator;
                BindingResolver resolver(
                    pressured.devices, pressured.data);
                BindingAdmissibilityValidator admissibility;
                ExecutionAdmissionValidator admission(
                    pressured.devices, pressured.data);
                GERDOS_CHECK(validator.validate(step.binding));
                GERDOS_CHECK(
                    resolver.resolve(step.binding).fully_resolved());
                GERDOS_CHECK(
                    admissibility.admissible(*operation, step.binding));
                auto* execution = pressured.executions.create_execution(
                    ExecutionDescription{
                        ExecutionId{base_id}, step.id});
                ++base_id;
                GERDOS_CHECK(execution->bind(step.binding));
                GERDOS_CHECK(admission.admit(*execution).has_value());
                GERDOS_CHECK(
                    pressured_executor.start(execution->description().id));
                std::vector<AttemptStatus> outcomes;
                pressured_executor.advance(outcomes);

                while (outcomes.empty()) {
                    pressured_executor.advance(outcomes);
                }

                GERDOS_CHECK(outcomes.size() == 1);
                GERDOS_CHECK(
                    outcomes.front().integrity ==
                    AttemptIntegrity::COHERENT);
                ++executed;

                // Evict the consumed weights record: explicit removal
                // preserves the logical Data object. Removal of a live
                // record with no claim succeeds; the Data stays.
                for (const auto input : operation->description().inputs) {
                    auto* datum = pressured.data.find_data(input);
                    GERDOS_CHECK(datum != nullptr);

                    // One residency per weights record: remove it, keep
                    // the Data.
                    std::vector<DataResidencyId> records;
                    datum->for_each_residency(
                        [&](const DataResidency* record) {
                            records.push_back(record->description().id);
                        });

                    for (const auto record : records) {
                        if (datum->remove_residency(record)) {
                            evicted.emplace_back(input, record);
                        }
                    }
                }
            }
        }

        GERDOS_CHECK(executed == 4);
        GERDOS_CHECK(!evicted.empty());

        // Eviction preserved every logical Data object.
        for (const auto& [data_id, record] : evicted) {
            (void)record;
            GERDOS_CHECK(pressured.data.find_data(data_id) != nullptr);
        }

        // Each partial is the sum of 262144 fresh 1.0 elements. The
        // partial records are 7102/7202/7302/7402 (data id with the
        // trailing digit replaced), not data*10+1.
        const std::pair<std::uint64_t, std::uint64_t> partials[] = {
            {711, 7102},
            {721, 7202},
            {731, 7302},
            {741, 7402},
        };

        for (const auto& [data_id, residency_id] : partials) {
            const float value = pressured_backend.sample(
                DataResidencyRef{
                    DataId{data_id}, DataResidencyId{residency_id}},
                0);
            std::printf(
                "partial %llu: %f (expect 262144)\n",
                (unsigned long long)data_id,
                value);
            std::fflush(stdout);
            GERDOS_CHECK(value == 262144.0f);
        }

        // Evidence grew once per executed attempt: four observations.
        const auto pressured_evidence = pressured.measurements.summarize(
            ResourceRef{DeviceId{100}, ResourceId{102}},
            MeasurementQuantity::DURATION_NS);
        std::printf(
            "pressured evidence: %llu observations in %zu rounds\n",
            (unsigned long long)
                pressured_evidence.succeeded_observations,
            rounds);
        std::fflush(stdout);
        GERDOS_CHECK(pressured_evidence.succeeded_observations == 4);
    }

    // ---------------------------------------------------------------------
    // 4. Removal preconditions hold under pressure
    // ---------------------------------------------------------------------

    {
        // A residency with a claimed update refuses removal; a Data with
        // a claimed residency refuses removal; terminal-state executions
        // allow removal afterwards. Exercised through the effects layer,
        // like real attempts do — claims cannot be forged.
        Machine guarded;
        CpuBackend guarded_backend;
        Executor guarded_executor(
            guarded.executions,
            guarded.operations,
            guarded.devices,
            guarded.data,
            guarded_backend,
            &guarded.measurements);

        auto* datum = guarded.data.create_data(
            DataDescription{DataId{800}, "guarded"});
        (void)datum->add_residency(
            DataResidency{
                DataResidencyDescription{
                    DataResidencyId{8001},
                    DataId{800},
                    ResourceRef{DeviceId{100}, ResourceId{101}},
                    "guarded",
                },
            });
        (void)datum->find_residency(DataResidencyId{8001})
            ->set_state(DataResidencyState::VALID);

        (void)guarded.operations.create_operation(OperationDescription{
            OperationId{880},
            {DataId{800}},
            {DataId{800}},
            {},
            {ResourceRequirement{ResourceBindingRole::COMPUTE, 1}},
            WorkDescription{4, 1, 0.0f, 1.0f, 0.0f},
        });

        BindingPlanner guarded_planner(
            guarded.devices,
            guarded.data,
            guarded.topology,
            guarded.measurements);
        const auto* operation =
            guarded.operations.find_operation(OperationId{880});
        const auto binding = guarded_planner.plan(*operation);
        GERDOS_CHECK(binding.has_value());

        auto* execution = guarded.executions.create_execution(
            ExecutionDescription{ExecutionId{3800}, OperationId{880}});
        GERDOS_CHECK(execution->bind(*binding));
        GERDOS_CHECK(guarded_executor.start(ExecutionId{3800}));

        // In-flight: the claimed record refuses removal, and the Data
        // with the claimed residency refuses removal.
        GERDOS_CHECK(
            !guarded.data.find_data(DataId{800})->remove_residency(
                DataResidencyId{8001}));
        GERDOS_CHECK(!guarded.data.remove_data(DataId{800}));

        std::vector<AttemptStatus> outcomes;
        guarded_executor.advance(outcomes);

        while (outcomes.empty()) {
            guarded_executor.advance(outcomes);
        }

        GERDOS_CHECK(outcomes.size() == 1);
        GERDOS_CHECK(
            outcomes.front().integrity == AttemptIntegrity::COHERENT);

        // Terminal: removal succeeds; the retired id stays retired.
        GERDOS_CHECK(
            guarded.data.find_data(DataId{800})->remove_residency(
                DataResidencyId{8001}));
        GERDOS_CHECK(
            guarded.data.find_data(DataId{800}) != nullptr);
    }

    // ---------------------------------------------------------------------
    // 5. The decoder prefix translates, executes, and verifies
    // ---------------------------------------------------------------------

    {
        // The exactly-expressible layer scope as model steps: LINEAR
        // projection, AFFINE shift, REDUCE_SUM fold, EXPONENTIAL lift,
        // REDUCE_MAX peak, mean-as-sum-with-1/N, plus the refusals that
        // bound the scope (RESIDUAL_ADD needs two sources, DIVIDE and
        // LAYER_NORM need division/variance, SOFTMAX/ATTENTION need the
        // closing divide). Every step translates exactly or is refused
        // loudly — never approximated.
        using gerdos::adapters::adapt;
        using gerdos::adapters::Adaptation;
        using gerdos::adapters::ModelOp;
        using gerdos::adapters::ModelStep;
        const std::vector<ModelStep> layer{
            ModelStep{ModelOp::LINEAR, 600, 601, 0, 602, 2, 3, 2},
            ModelStep{ModelOp::AFFINE, 610, 0, 0, 610, 6, 0, 0, 1.0f, 2.0f, 1.0f},
            ModelStep{ModelOp::REDUCE_SUM, 611, 0, 0, 611, 6},
            ModelStep{ModelOp::EXPONENTIAL, 611, 0, 0, 612, 6},
            ModelStep{ModelOp::REDUCE_MAX, 612, 0, 0, 613, 6},
            ModelStep{ModelOp::REDUCE_MEAN, 612, 0, 0, 614, 6},
            ModelStep{ModelOp::REDUCE_MEAN, 620, 0, 0, 621, 6},
            ModelStep{ModelOp::RESIDUAL_ADD, 610, 611, 0, 615, 6},
            ModelStep{ModelOp::DIVIDE, 612, 613, 0, 616, 6},
            ModelStep{ModelOp::LAYER_NORM, 610, 0, 0, 617, 6},
            ModelStep{ModelOp::SOFTMAX, 612, 0, 0, 618, 6},
            ModelStep{ModelOp::ATTENTION, 611, 612, 0, 619, 6},
        };

        const Adaptation translated = adapt("decoder layer", layer);
        GERDOS_CHECK(translated.workload.operations.size() == 7);
        GERDOS_CHECK(translated.refused.size() == 5);
        GERDOS_CHECK(translated.refused[0].find("RESIDUAL_ADD") == 0);
        GERDOS_CHECK(translated.refused[1].find("DIVIDE") == 0);
        GERDOS_CHECK(translated.refused[2].find("LAYER_NORM") == 0);
        GERDOS_CHECK(translated.refused[3].find("SOFTMAX") == 0);
        GERDOS_CHECK(translated.refused[4].find("ATTENTION") == 0);

        GERDOS_CHECK(translated.workload.operations[0].work.form == WorkForm::MATRIX_PRODUCT);
        GERDOS_CHECK(translated.workload.operations[5].work.form == WorkForm::REDUCE_SUM);
        GERDOS_CHECK(translated.workload.operations[5].work.source_scale == 1.0f / 6.0f);

        // The translated prefix executes through the full gate chain
        // with the workload artifact cross-checking identical values:
        // the artifact file declares the same seven operations.
        Machine layered;
        CpuBackend layered_backend;
        Executor layered_executor(
            layered.executions,
            layered.operations,
            layered.devices,
            layered.data,
            layered_backend,
            &layered.measurements);
        BindingPlanner layered_planner(
            layered.devices,
            layered.data,
            layered.topology,
            layered.measurements);

        const auto layered_parsed = parse_artifact(
            load_text((directory + "/decoder_prefix.gwd").c_str()));
        GERDOS_CHECK(layered_parsed.ok);
        GERDOS_CHECK(layered_parsed.artifact.workload.operations.size() == 7);

        std::string layered_reason;
        GERDOS_CHECK(populate_registries(
            layered_parsed.artifact, layered.devices, layered.data,
            layered.operations, layered_reason));

        const auto layered_plan = layered_planner.plan_attempts(layered.operations);
        GERDOS_CHECK(layered_plan.size() == 7);

        // Hand-rolled CPU reference (outside GERDOS): the same chain
        // computed directly. Bit-exact where integer shapes govern
        // (projection, shift, fold); tolerance-documented where float
        // reductions accumulate (sums and means within 4 ulps).
        float ref_proj[4];
        float ref_exp[6];

        for (std::size_t i = 0; i < 4; ++i) {
            ref_proj[i] = 3.0f;
        }


        std::size_t layered_attempt = 0;

        for (const auto& step : layered_plan) {
            const auto* operation = layered.operations.find_operation(step.id);
            GERDOS_CHECK(operation != nullptr);
            PhysicalBindingValidator validator;
            BindingResolver resolver(layered.devices, layered.data);
            BindingAdmissibilityValidator admissibility;
            ExecutionAdmissionValidator admission(layered.devices, layered.data);
            GERDOS_CHECK(validator.validate(step.binding));
            GERDOS_CHECK(resolver.resolve(step.binding).fully_resolved());
            GERDOS_CHECK(admissibility.admissible(*operation, step.binding));
            auto* execution = layered.executions.create_execution(
                ExecutionDescription{ExecutionId{6000 + layered_attempt}, step.id});
            ++layered_attempt;
            GERDOS_CHECK(execution->bind(step.binding));
            GERDOS_CHECK(admission.admit(*execution).has_value());
            GERDOS_CHECK(layered_executor.start(execution->description().id));
            std::vector<AttemptStatus> outcomes;
            layered_executor.advance(outcomes);

            while (outcomes.empty()) {
                layered_executor.advance(outcomes);
            }

            GERDOS_CHECK(outcomes.size() == 1);
            GERDOS_CHECK(outcomes.front().integrity == AttemptIntegrity::COHERENT);
        }

        // Projection: ones(2x3) x ones(3x2) = [[3,3],[3,3]] exactly.
        for (std::size_t i = 0; i < 4; ++i) {
            GERDOS_CHECK(layered_backend.sample(DataResidencyRef{DataId{602}, DataResidencyId{6003}}, i) == ref_proj[i]);
        }
        // Shift: fresh 1.0 through dst*1 + src*2 + 1 = 4.0 exactly.
        GERDOS_CHECK(layered_backend.sample(DataResidencyRef{DataId{610}, DataResidencyId{6010}}, 0) == 4.0f);

        // Reference exponentials off the raised record.
        for (std::size_t i = 0; i < 6; ++i) {
            const float r = layered_backend.sample(DataResidencyRef{DataId{611}, DataResidencyId{6011}}, i);
            ref_exp[i] = std::exp(r);
        }

        const float e0 = layered_backend.sample(DataResidencyRef{DataId{612}, DataResidencyId{6012}}, 0);
        const float e1 = layered_backend.sample(DataResidencyRef{DataId{612}, DataResidencyId{6012}}, 1);
        const float peak = layered_backend.sample(DataResidencyRef{DataId{613}, DataResidencyId{6013}}, 0);
        const float mean = layered_backend.sample(DataResidencyRef{DataId{614}, DataResidencyId{6014}}, 0);
        const float norm_mean = layered_backend.sample(DataResidencyRef{DataId{621}, DataResidencyId{6021}}, 0);
        float ref_sum = 0.0f;

        for (std::size_t i = 0; i < 6; ++i) {
            ref_sum += ref_exp[i];
        }

        std::printf("layer: E0=%f S+=%f M=%f mean=%f norm=%f\n", e0, e1, peak, mean, norm_mean);
        std::printf("reference: E0=%f S+=%f M=%f mean=%f\n", ref_exp[0], ref_exp[1], ref_exp[0], ref_sum / 6.0f);
        std::fflush(stdout);
        GERDOS_CHECK(e0 == ref_exp[0]);
        GERDOS_CHECK(e1 == ref_exp[1]);
        GERDOS_CHECK(peak == ref_exp[0]);
        GERDOS_CHECK(std::fabs(mean - ref_sum / 6.0f) <= std::numeric_limits<float>::epsilon() * (ref_sum / 6.0f) * 4);
       // ---------------------------------------------------------------------
    // 6. The signal chain translates, executes, and verifies
    // ---------------------------------------------------------------------

    {
        // The second consumer family end to end: normalize, masked
        // select, and histogram translate exactly; thresholding is
        // refused (no comparison form). Fresh 1.0 allocations double as
        // the always-true predicate and the gather index — no seeding,
        // every value closed-form exact.
        using gerdos::adapters::adapt_signal;
        using gerdos::adapters::SignalAdaptation;
        using gerdos::adapters::SignalOp;
        using gerdos::adapters::SignalStep;

        const std::vector<SignalStep> chain{
            SignalStep{SignalOp::SIGNAL_NORMALIZE, 900, 0, 0, 901, 6, 0.0f, 0.5f, 0.25f},
            SignalStep{SignalOp::SIGNAL_THRESHOLD, 901, 0, 0, 910, 6},
            SignalStep{SignalOp::SIGNAL_SELECT, 910, 901, 912, 913, 6},
            SignalStep{SignalOp::SIGNAL_HISTOGRAM, 913, 920, 0, 921, 6},
        };

        const SignalAdaptation signal_translated = adapt_signal("signal", chain);
        GERDOS_CHECK(signal_translated.workload.operations.size() == 3);
        GERDOS_CHECK(signal_translated.refused.size() == 1);
        GERDOS_CHECK(signal_translated.refused[0].find("SIGNAL_THRESHOLD") == 0);
        GERDOS_CHECK(signal_translated.workload.operations[0].work.form == WorkForm::ELEMENTWISE_AFFINE);
        GERDOS_CHECK(signal_translated.workload.operations[1].work.form == WorkForm::MASK_SELECT);
        GERDOS_CHECK(signal_translated.workload.operations[2].work.form == WorkForm::GATHER);

        Machine signaled;
        CpuBackend signaled_backend;
        Executor signaled_executor(
            signaled.executions,
            signaled.operations,
            signaled.devices,
            signaled.data,
            signaled_backend,
            &signaled.measurements);
        BindingPlanner signaled_planner(
            signaled.devices,
            signaled.data,
            signaled.topology,
            signaled.measurements);

        const auto signaled_parsed = parse_artifact(
            load_text((directory + "/signal_chain.gwd").c_str()));
        GERDOS_CHECK(signaled_parsed.ok);
        GERDOS_CHECK(signaled_parsed.artifact.workload.operations.size() == 3);

        std::string signaled_reason;
        GERDOS_CHECK(populate_registries(
            signaled_parsed.artifact, signaled.devices, signaled.data,
            signaled.operations, signaled_reason));

        const auto signaled_plan = signaled_planner.plan_attempts(signaled.operations);
        GERDOS_CHECK(signaled_plan.size() == 3);

        std::size_t signaled_attempt = 0;

        for (const auto& step : signaled_plan) {
            const auto* operation = signaled.operations.find_operation(step.id);
            GERDOS_CHECK(operation != nullptr);
            PhysicalBindingValidator validator;
            BindingResolver resolver(signaled.devices, signaled.data);
            BindingAdmissibilityValidator admissibility;
            ExecutionAdmissionValidator admission(signaled.devices, signaled.data);
            GERDOS_CHECK(validator.validate(step.binding));
            GERDOS_CHECK(resolver.resolve(step.binding).fully_resolved());
            GERDOS_CHECK(admissibility.admissible(*operation, step.binding));
            auto* execution = signaled.executions.create_execution(
                ExecutionDescription{ExecutionId{7000 + signaled_attempt}, step.id});
            ++signaled_attempt;
            GERDOS_CHECK(execution->bind(step.binding));
            GERDOS_CHECK(admission.admit(*execution).has_value());
            GERDOS_CHECK(signaled_executor.start(execution->description().id));
            std::vector<AttemptStatus> outcomes;
            signaled_executor.advance(outcomes);

            while (outcomes.empty()) {
                signaled_executor.advance(outcomes);
            }

            GERDOS_CHECK(outcomes.size() == 1);
            GERDOS_CHECK(outcomes.front().integrity == AttemptIntegrity::COHERENT);
        }

        // Closed forms: normalize maps 1.0 -> 0.75; select picks 0.75
        // over dark 1.0 under the always-true predicate; gather reads
        // position 1 (0.75) into every slot. All bit-exact.
        for (std::size_t i = 0; i < 6; ++i) {
            GERDOS_CHECK(signaled_backend.sample(DataResidencyRef{DataId{901}, DataResidencyId{9002}}, i) == 0.75f);
            GERDOS_CHECK(signaled_backend.sample(DataResidencyRef{DataId{913}, DataResidencyId{9013}}, i) == 0.75f);
            GERDOS_CHECK(signaled_backend.sample(DataResidencyRef{DataId{921}, DataResidencyId{9021}}, i) == 0.75f);
        }

        std::printf("signal: 0.75 exact across normalize/select/gather, 3 ops coherent\n");
        std::fflush(stdout);

        const auto signaled_evidence = signaled.measurements.summarize(
            ResourceRef{DeviceId{100}, ResourceId{102}},
            MeasurementQuantity::DURATION_NS);
        GERDOS_CHECK(signaled_evidence.succeeded_observations == 3);
    }

     GERDOS_CHECK(norm_mean == 1.0f);
    }


    // ---------------------------------------------------------------------
    // 7. The tiny model translates, executes, and verifies
    // ---------------------------------------------------------------------

    {
        // The smallest honest model: a linear classifier over fresh-1.0
        // features. LINEAR projection to scores, AFFINE bias shift,
        // second-layer map, min/max/mean over the score vector — every
        // value closed-form exact. The decision (which class) is
        // refused: ARGMAX names the gap. This section also pins the
        // surplus rule: record 102 is sized 6 by its matmul producer
        // and read by a 2-wide consumer, which must keep — never wipe.
        using gerdos::adapters::adapt;
        using gerdos::adapters::Adaptation;
        using gerdos::adapters::ModelOp;
        using gerdos::adapters::ModelStep;

        const std::vector<ModelStep> tiny{
            ModelStep{ModelOp::AFFINE, 100, 0, 0, 100, 6, 0, 0, 1.0f, 1.0f, 0.0f},
            ModelStep{ModelOp::LINEAR, 100, 101, 0, 102, 2, 3, 1},
            ModelStep{ModelOp::AFFINE, 102, 0, 0, 103, 2, 0, 0, 0.0f, 1.0f, 0.5f},
            ModelStep{ModelOp::AFFINE, 103, 0, 0, 104, 2, 0, 0, 0.0f, 2.0f, 1.0f},
            ModelStep{ModelOp::REDUCE_MIN, 104, 0, 0, 105, 2},
            ModelStep{ModelOp::REDUCE_MAX, 104, 0, 0, 106, 2},
            ModelStep{ModelOp::REDUCE_MEAN, 104, 0, 0, 107, 2},
            ModelStep{ModelOp::ARGMAX, 104, 0, 0, 108, 2},
        };

        const Adaptation translated = adapt("tiny model", tiny);
        GERDOS_CHECK(translated.workload.operations.size() == 7);
        GERDOS_CHECK(translated.refused.size() == 1);
        GERDOS_CHECK(translated.refused[0].find("ARGMAX") == 0);
        GERDOS_CHECK(translated.workload.operations[1].work.form == WorkForm::MATRIX_PRODUCT);
        GERDOS_CHECK(translated.workload.operations[1].work.rows == 2);
        GERDOS_CHECK(translated.workload.operations[1].work.inner == 3);
        GERDOS_CHECK(translated.workload.operations[1].work.columns == 1);

        Machine modeled;
        CpuBackend modeled_backend;
        Executor modeled_executor(
            modeled.executions,
            modeled.operations,
            modeled.devices,
            modeled.data,
            modeled_backend,
            &modeled.measurements);
        BindingPlanner modeled_planner(
            modeled.devices,
            modeled.data,
            modeled.topology,
            modeled.measurements);

        const auto modeled_parsed = parse_artifact(
            load_text((directory + "/tiny_model.gwd").c_str()));
        GERDOS_CHECK(modeled_parsed.ok);
        GERDOS_CHECK(modeled_parsed.artifact.workload.operations.size() == 7);

        std::string modeled_reason;
        GERDOS_CHECK(populate_registries(
            modeled_parsed.artifact, modeled.devices, modeled.data,
            modeled.operations, modeled_reason));

        const auto modeled_plan = modeled_planner.plan_attempts(modeled.operations);
        GERDOS_CHECK(modeled_plan.size() == 7);

        std::size_t modeled_attempt = 0;

        for (const auto& step : modeled_plan) {
            const auto* operation = modeled.operations.find_operation(step.id);
            GERDOS_CHECK(operation != nullptr);
            PhysicalBindingValidator validator;
            BindingResolver resolver(modeled.devices, modeled.data);
            BindingAdmissibilityValidator admissibility;
            ExecutionAdmissionValidator admission(modeled.devices, modeled.data);
            GERDOS_CHECK(validator.validate(step.binding));
            GERDOS_CHECK(resolver.resolve(step.binding).fully_resolved());
            GERDOS_CHECK(admissibility.admissible(*operation, step.binding));
            auto* execution = modeled.executions.create_execution(
                ExecutionDescription{ExecutionId{8000 + modeled_attempt}, step.id});
            ++modeled_attempt;
            GERDOS_CHECK(execution->bind(step.binding));
            GERDOS_CHECK(admission.admit(*execution).has_value());
            GERDOS_CHECK(modeled_executor.start(execution->description().id));
            std::vector<AttemptStatus> outcomes;
            modeled_executor.advance(outcomes);

            while (outcomes.empty()) {
                modeled_executor.advance(outcomes);
            }

            GERDOS_CHECK(outcomes.size() == 1);
            GERDOS_CHECK(outcomes.front().integrity == AttemptIntegrity::COHERENT);
        }

        // Closed forms: W = 2.0; scores = [6,6]; biased = [6.5,6.5];
        // second = [14,14]; min = max = mean = 14. All bit-exact.
        GERDOS_CHECK(modeled_backend.sample(DataResidencyRef{DataId{100}, DataResidencyId{1001}}, 0) == 2.0f);
        GERDOS_CHECK(modeled_backend.sample(DataResidencyRef{DataId{102}, DataResidencyId{1003}}, 0) == 6.0f);
        GERDOS_CHECK(modeled_backend.sample(DataResidencyRef{DataId{102}, DataResidencyId{1003}}, 1) == 6.0f);
        GERDOS_CHECK(modeled_backend.sample(DataResidencyRef{DataId{103}, DataResidencyId{1004}}, 0) == 6.5f);
        GERDOS_CHECK(modeled_backend.sample(DataResidencyRef{DataId{104}, DataResidencyId{1005}}, 0) == 14.0f);
        GERDOS_CHECK(modeled_backend.sample(DataResidencyRef{DataId{105}, DataResidencyId{1006}}, 0) == 14.0f);
        GERDOS_CHECK(modeled_backend.sample(DataResidencyRef{DataId{106}, DataResidencyId{1007}}, 0) == 14.0f);
        GERDOS_CHECK(modeled_backend.sample(DataResidencyRef{DataId{107}, DataResidencyId{1008}}, 0) == 14.0f);

        std::printf("tiny: W=2 scores=6 biased=6.5 layer2=14 stats=14, 7 ops coherent\n");
        std::fflush(stdout);

        const auto modeled_evidence = modeled.measurements.summarize(
            ResourceRef{DeviceId{100}, ResourceId{102}},
            MeasurementQuantity::DURATION_NS);
        GERDOS_CHECK(modeled_evidence.succeeded_observations == 7);
    }

    return 0;
}

