import random

import yaml
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.orchestrator import Orchestrator
from src.telemetry import AvailabilityState, ChargingState, FleetTelemetrySimulator, NetworkType, VehicleTelemetry
from study import _final_window_metrics, _inverse_participation_weights, _mean_std, _select_clients, run_one


def test_accuracy_carries_forward_when_all_selected_clients_drop(monkeypatch):
    config = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    config["simulation"]["fleet_size"] = 1
    config["simulation"]["clients_per_round"] = 1
    config["simulation"]["num_rounds"] = 2
    config["simulation"]["batch_size"] = 2

    def always_drop(self, client_id, is_baseline):
        return False, "forced dropout for test"

    monkeypatch.setattr(
        "src.telemetry.FleetTelemetrySimulator.simulate_mid_round_dynamics",
        always_drop,
    )
    labels = torch.arange(10).repeat(2)
    loader = DataLoader(
        TensorDataset(torch.randn(20, 1, 28, 28), labels),
        batch_size=10,
    )
    result = run_one(
        config,
        policy="A",
        rho=0.0,
        seed=42,
        client_loaders=[loader],
        test_loader=loader,
    )

    assert result["wasted_client_rounds"] == 2
    assert [row["successful_count"] for row in result["round_history"]] == [0, 0]
    assert [row["test_accuracy"] for row in result["round_history"]] == [
        result["initial_accuracy_pct"],
        result["initial_accuracy_pct"],
    ]
    assert 0.0 <= result["worst_class_accuracy_pct"] <= 100.0
    assert result["worst_class_accuracy_pct"] == min(result["per_class_accuracy_pct"])


def test_final_window_averages_accuracy_and_each_class():
    history = [
        {"test_accuracy": float(round_num), "per_class_accuracy": [float(round_num)] * 10}
        for round_num in range(1, 11)
    ]
    accuracy, per_class = _final_window_metrics(history, 5)
    assert accuracy == 8.0
    assert per_class == [8.0] * 10


def test_mean_std_includes_student_t_95_percent_interval():
    result = _mean_std([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result["n"] == 5
    assert result["ci95_lower"] < result["mean"] < result["ci95_upper"]
    assert round(result["ci95_upper"] - result["mean"], 3) == round(
        2.776 * result["std"] / (5 ** 0.5), 3
    )


def test_policy_d_prioritizes_stale_clients_without_breaking_guardrails():
    fleet = {
        f"ev_client_{index:02d}": VehicleTelemetry(
            client_id=f"ev_client_{index:02d}",
            charging_state=ChargingState.LEVEL_2_CHARGING,
            battery_percentage=80.0,
            network_type=NetworkType.UNMETERED_WIFI,
            bandwidth_mbps=30.0,
            availability_state=AvailabilityState.CHARGING_WIFI,
        )
        for index in range(3)
    }
    assumptions = yaml.safe_load(open("config.yaml", encoding="utf-8"))["assumptions"]
    participation = {cid: 1 for cid in fleet}
    last_participation = {"ev_client_00": 0, "ev_client_01": 10, "ev_client_02": 10}
    selected, eligible_count, _ = _select_clients(
        "D",
        fleet,
        Orchestrator(),
        clients_per_round=1,
        rng=random.Random(4),
        assumptions=assumptions,
        participation=participation,
        last_participation=last_participation,
        round_num=100,
    )
    assert eligible_count == 3
    assert selected == ["ev_client_00"]


def test_p2p_selects_only_connectable_parked_wifi_pairs():
    fleet = {
        "ev_client_00": VehicleTelemetry(
            client_id="ev_client_00",
            charging_state=ChargingState.DISCHARGING,
            battery_percentage=80.0,
            network_type=NetworkType.UNMETERED_WIFI,
            bandwidth_mbps=30.0,
            availability_state=AvailabilityState.PARKED_WIFI,
        ),
        "ev_client_01": VehicleTelemetry(
            client_id="ev_client_01",
            charging_state=ChargingState.DISCHARGING,
            battery_percentage=80.0,
            network_type=NetworkType.UNMETERED_WIFI,
            bandwidth_mbps=30.0,
            availability_state=AvailabilityState.PARKED_WIFI,
        ),
        "ev_client_02": VehicleTelemetry(
            client_id="ev_client_02",
            charging_state=ChargingState.DISCHARGING,
            battery_percentage=80.0,
            network_type=NetworkType.CELLULAR_5G,
            bandwidth_mbps=30.0,
            availability_state=AvailabilityState.DRIVING,
        ),
    }
    assumptions = yaml.safe_load(open("config.yaml", encoding="utf-8"))["assumptions"]
    selected, connectable_count, _ = _select_clients(
        "P2P",
        fleet,
        Orchestrator(),
        clients_per_round=5,
        rng=random.Random(4),
        assumptions=assumptions,
        participation={cid: 0 for cid in fleet},
        last_participation={cid: 0 for cid in fleet},
        round_num=1,
    )
    assert connectable_count == 2
    assert len(selected) == 2
    assert all(fleet[cid].availability_state == AvailabilityState.PARKED_WIFI for cid in selected)


def test_p2p_two_selects_four_connectable_clients():
    fleet = {
        f"ev_client_{index:02d}": VehicleTelemetry(
            client_id=f"ev_client_{index:02d}",
            charging_state=ChargingState.LEVEL_2_CHARGING,
            battery_percentage=80.0,
            network_type=NetworkType.UNMETERED_WIFI,
            bandwidth_mbps=30.0,
            availability_state=AvailabilityState.PARKED_WIFI,
        )
        for index in range(6)
    }
    assumptions = yaml.safe_load(open("config.yaml", encoding="utf-8"))["assumptions"]
    selected, _, _ = _select_clients(
        "P2P-2", fleet, Orchestrator(), 5, random.Random(5), assumptions,
        {cid: 0 for cid in fleet}, {cid: 0 for cid in fleet}, 1,
    )
    assert len(selected) == 4


def test_inverse_participation_weights_and_oracle_group_balance():
    updates = {
        "ev_client_00": ([torch.tensor([1.0]).numpy()], 100),
        "ev_client_01": ([torch.tensor([2.0]).numpy()], 100),
        "ev_client_02": ([torch.tensor([3.0]).numpy()], 100),
    }
    participation = {"ev_client_00": 1, "ev_client_01": 9, "ev_client_02": 1}
    profiles = {"ev_client_00": "heavy-use", "ev_client_01": "heavy-use", "ev_client_02": "light-use"}
    realistic = _inverse_participation_weights(updates, participation, profiles)
    assert realistic[0][1] == 100.0
    assert realistic[1][1] < realistic[0][1]
    oracle = _inverse_participation_weights(updates, participation, profiles, oracle_group_balance=True)
    heavy_weight = sum(weight for cid, (_, weight) in zip(updates, oracle) if profiles[cid] == "heavy-use")
    light_weight = sum(weight for cid, (_, weight) in zip(updates, oracle) if profiles[cid] == "light-use")
    assert abs(heavy_weight - light_weight) < 1e-12


def test_policy_f_adds_only_heavy_charging_wifi_fallback():
    fleet = {
        "ev_client_00": VehicleTelemetry(
            client_id="ev_client_00", charging_state=ChargingState.LEVEL_2_CHARGING,
            battery_percentage=40, network_type=NetworkType.UNMETERED_WIFI,
            bandwidth_mbps=30, usage_profile="heavy-use",
            availability_state=AvailabilityState.CHARGING_WIFI,
        ),
        "ev_client_01": VehicleTelemetry(
            client_id="ev_client_01", charging_state=ChargingState.LEVEL_2_CHARGING,
            battery_percentage=40, network_type=NetworkType.UNMETERED_WIFI,
            bandwidth_mbps=30, usage_profile="heavy-use",
            availability_state=AvailabilityState.CHARGING_WIFI,
        ),
        "ev_client_02": VehicleTelemetry(
            client_id="ev_client_02", charging_state=ChargingState.LEVEL_2_CHARGING,
            battery_percentage=80, network_type=NetworkType.UNMETERED_WIFI,
            bandwidth_mbps=30, usage_profile="light-use",
            availability_state=AvailabilityState.CHARGING_WIFI,
        ),
        "ev_client_03": VehicleTelemetry(
            client_id="ev_client_03", charging_state=ChargingState.DISCHARGING,
            battery_percentage=80, network_type=NetworkType.UNMETERED_WIFI,
            bandwidth_mbps=30, usage_profile="heavy-use",
            availability_state=AvailabilityState.PARKED_WIFI,
        ),
    }
    config = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    selected, pool_count, _ = _select_clients(
        "F", fleet, Orchestrator(), 5, random.Random(9), config["assumptions"],
        {cid: 0 for cid in fleet}, {cid: 0 for cid in fleet}, 1,
    )
    assert pool_count == 4
    assert set(selected) == set(fleet)


def test_policy_f_fallback_requires_wifi_and_non_driving_state():
    config = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    assumptions = config["assumptions"]
    fleet = {
        "ev_client_00": VehicleTelemetry(
            client_id="ev_client_00", charging_state=ChargingState.DISCHARGING,
            battery_percentage=80, network_type=NetworkType.UNMETERED_WIFI,
            bandwidth_mbps=30, usage_profile="heavy-use",
            availability_state=AvailabilityState.PARKED_WIFI,
        ),
        "ev_client_01": VehicleTelemetry(
            client_id="ev_client_01", charging_state=ChargingState.LEVEL_2_CHARGING,
            battery_percentage=80, network_type=NetworkType.UNMETERED_WIFI,
            bandwidth_mbps=30, usage_profile="light-use",
            availability_state=AvailabilityState.CHARGING_WIFI,
        ),
        "ev_client_02": VehicleTelemetry(
            client_id="ev_client_02", charging_state=ChargingState.DISCHARGING,
            battery_percentage=80, network_type=NetworkType.CELLULAR_5G,
            bandwidth_mbps=30, usage_profile="heavy-use",
            availability_state=AvailabilityState.DRIVING,
        ),
    }
    selected, pool_count, _ = _select_clients(
        "F", fleet, Orchestrator(), 5, random.Random(11), assumptions,
        {cid: 0 for cid in fleet}, {cid: 0 for cid in fleet}, 1,
    )
    assert pool_count == 2
    assert set(selected) == {"ev_client_00", "ev_client_01"}


def test_policies_share_the_same_exogenous_availability_trace():
    config = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    config["simulation"]["fleet_size"] = 1
    config["simulation"]["clients_per_round"] = 1
    config["simulation"]["num_rounds"] = 2
    config["simulation"]["batch_size"] = 2
    config["simulation"]["final_metric_rounds"] = 2
    labels = torch.arange(10).repeat(2)
    loader = DataLoader(
        TensorDataset(torch.randn(20, 1, 28, 28), labels),
        batch_size=10,
    )

    random_result = run_one(config, "A", 0.5, 42, [loader], loader)
    guardrail_result = run_one(config, "C", 0.5, 42, [loader], loader)

    assert random_result["availability_trace_sha256"] == guardrail_result["availability_trace_sha256"]


def test_selected_client_count_does_not_shift_future_dropout_draws():
    config = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    assumptions = config["assumptions"]
    kwargs = {
        "num_clients": 4,
        "seed": 18,
        "profile_share": assumptions["profile_share"],
        "initial_state_probabilities": assumptions["initial_state_probabilities"],
        "transition_matrices": assumptions["availability_transition_matrices"],
        "dropout_assumptions": assumptions["midround_dropout"],
    }
    random_run = FleetTelemetrySimulator(**kwargs)
    guardrail_run = FleetTelemetrySimulator(**kwargs)

    random_run.advance_round(1)
    guardrail_run.advance_round(1)
    for client_id in random_run.clients:
        random_run.simulate_mid_round_dynamics(client_id, is_baseline=True)
    guardrail_run.simulate_mid_round_dynamics("ev_client_00", is_baseline=False)

    random_run.advance_round(2)
    guardrail_run.advance_round(2)
    assert random_run.dropout_draws == guardrail_run.dropout_draws


def test_profile_assignments_are_randomized_by_seed_and_reproducible():
    config = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    assumptions = config["assumptions"]
    fleets = [
        FleetTelemetrySimulator(
            num_clients=20,
            seed=seed,
            profile_share=assumptions["profile_share"],
            initial_state_probabilities=assumptions["initial_state_probabilities"],
            transition_matrices=assumptions["availability_transition_matrices"],
        )
        for seed in (42, 43, 61)
    ]
    same_seed = FleetTelemetrySimulator(
        num_clients=20,
        seed=42,
        profile_share=assumptions["profile_share"],
        initial_state_probabilities=assumptions["initial_state_probabilities"],
        transition_matrices=assumptions["availability_transition_matrices"],
    )
    assert fleets[0].usage_profiles == same_seed.usage_profiles
    assert len({tuple(sorted(fleet.usage_profiles.items())) for fleet in fleets}) > 1