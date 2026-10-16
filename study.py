"""Controlled experiments for availability-aware EV client selection."""

from copy import deepcopy
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.model import EdgeCNN, get_model_size_bytes, get_parameters, partition_dataset, set_parameters, train
from src.orchestrator import Orchestrator
from src.telemetry import AvailabilityState, ChargingState, FleetTelemetrySimulator, NetworkType, VehicleTelemetry


POLICIES = {
    "A": "random",
    "B": "overselect",
    "C": "guardrail",
    "D": "guardrail_staleness",
    "P2P": "peer_to_peer",
    "E1": "inverse_participation_weighted",
    "E2": "oracle_group_balanced_weighted",
    "F": "guardrail_heavy_fallback",
    "P2P-2": "two_connectable_peer_pairs",
}
POLICY_NAMES = {
    "A": "Random selection",
    "B": "Random over-selection",
    "C": "Guardrail selection",
    "D": "Guardrail + staleness bonus",
    "P2P": "Connectable peer pairs",
    "E1": "Inverse-participation weighted (realistic)",
    "E2": "Group-balanced weighting (oracle)",
    "F": "Guardrail + heavy-use battery fallback",
    "P2P-2": "Two connectable peer pairs",
}


def _evaluate_by_class(model: EdgeCNN, loader: DataLoader, device: torch.device) -> Tuple[float, List[float]]:
    model.eval()
    correct = torch.zeros(10, dtype=torch.long)
    total = torch.zeros(10, dtype=torch.long)
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            predictions = outputs.argmax(dim=1).cpu()
            for label in range(10):
                mask = labels == label
                total[label] += mask.sum()
                correct[label] += (predictions[mask] == labels[mask]).sum()
    class_accuracy = [
        float(correct[index] / total[index]) if total[index] else 0.0
        for index in range(10)
    ]
    return float(correct.sum() / total.sum()), class_accuracy


def _survival_probability(telemetry: Any, assumptions: dict) -> float:
    dropout = assumptions["midround_dropout"]
    probability = 1.0
    if telemetry.charging_state.value == "DISCHARGING":
        probability *= 1.0 - dropout["driving_probability"]
    if telemetry.battery_percentage < dropout["critical_battery_pct"]:
        probability *= 1.0 - dropout["low_battery_probability"]
    if (
        telemetry.network_type == NetworkType.METERED_4G
        or telemetry.bandwidth_mbps < dropout["low_bandwidth_mbps"]
    ):
        probability *= 1.0 - dropout["poor_network_probability"]
    return probability


def _select_clients(
    policy: str,
    fleet: Dict[str, Any],
    orchestrator: Orchestrator,
    clients_per_round: int,
    rng: random.Random,
    assumptions: dict,
    participation: Dict[str, int],
    last_participation: Dict[str, int],
    round_num: int,
) -> Tuple[List[str], int, float]:
    client_ids = sorted(fleet)
    if policy == "A":
        selected = rng.sample(client_ids, min(clients_per_round, len(client_ids)))
        return selected, len(client_ids), sum(_survival_probability(fleet[cid], assumptions) for cid in selected)
    if policy == "B":
        mean_survival = mean(_survival_probability(fleet[cid], assumptions) for cid in client_ids)
        floor = assumptions["over_selection"]["survival_floor_for_sizing"]
        target = assumptions["over_selection"]["expected_survivors_target"]
        requested = min(len(client_ids), math.ceil(target / max(mean_survival, floor)))
        selected = rng.sample(client_ids, requested)
        expected = sum(_survival_probability(fleet[cid], assumptions) for cid in selected)
        return selected, len(client_ids), expected
    if policy in ("C", "E1", "E2"):
        eligible, _ = orchestrator.filter_eligible_clients(fleet)
        selected = rng.sample(eligible, min(clients_per_round, len(eligible)))
        expected = sum(_survival_probability(fleet[cid], assumptions) for cid in selected)
        return selected, len(eligible), expected
    if policy == "D":
        eligible, _ = orchestrator.filter_eligible_clients(fleet)
        count = min(clients_per_round, len(eligible))
        bonus = assumptions["policy_d"]["staleness_bonus"]
        remaining = list(eligible)
        selected = []
        while len(selected) < count:
            weights = [
                1.0 + bonus * (round_num - last_participation[cid])
                for cid in remaining
            ]
            chosen = rng.choices(remaining, weights=weights, k=1)[0]
            selected.append(chosen)
            remaining.remove(chosen)
        expected = sum(_survival_probability(fleet[cid], assumptions) for cid in selected)
        return selected, len(eligible), expected
    if policy == "F":
        eligible, _ = orchestrator.filter_eligible_clients(fleet)
        eligible_heavy = [cid for cid in eligible if fleet[cid].usage_profile == "heavy-use"]
        selection_pool = list(eligible)
        fallback = assumptions["policy_f"]["fallback_battery_pct"]
        min_heavy = assumptions["policy_f"]["min_heavy_eligible"]
        if len(eligible_heavy) < min_heavy:
            eligible_set = set(eligible)
            for cid, telemetry in fleet.items():
                if (
                    cid not in eligible_set
                    and telemetry.usage_profile == "heavy-use"
                    and telemetry.availability_state in (
                        AvailabilityState.CHARGING_WIFI,
                        AvailabilityState.PARKED_WIFI,
                    )
                    and telemetry.network_type == NetworkType.UNMETERED_WIFI
                    and telemetry.battery_percentage >= fallback
                ):
                    selection_pool.append(cid)
        selected = rng.sample(selection_pool, min(clients_per_round, len(selection_pool)))
        expected = sum(_survival_probability(fleet[cid], assumptions) for cid in selected)
        return selected, len(selection_pool), expected
    if policy in ("P2P", "P2P-2"):
        connectable = [
            cid for cid in client_ids
            if fleet[cid].availability_state.value == "PARKED_WIFI"
            and fleet[cid].network_type == NetworkType.UNMETERED_WIFI
        ]
        pairs = assumptions["p2p"]["pairs_per_round"]
        if policy == "P2P-2":
            pairs = assumptions["p2p"]["pairs_per_round_two"]
        count = min(2 * pairs, len(connectable))
        selected = rng.sample(connectable, count - count % 2)
        return selected, len(connectable), float(len(selected))
    raise ValueError(f"Unknown policy: {policy}")


def _client_class_counts(client_loaders: Sequence[DataLoader]) -> List[List[int]]:
    counts = []
    for loader in client_loaders:
        dataset = loader.dataset
        if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
            targets = dataset.dataset.targets[dataset.indices]
        elif hasattr(dataset, "tensors"):
            targets = dataset.tensors[1]
        else:
            raise TypeError("Client dataset must expose labels through Subset or TensorDataset")
        counts.append(np.bincount(np.asarray(targets, dtype=int), minlength=10).tolist())
    return counts


def _aggregate_updates(updates: Sequence[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
    total_examples = sum(weight for _, weight in updates)
    return [
        sum(parameters[layer] * (weight / total_examples) for parameters, weight in updates)
        for layer in range(len(updates[0][0]))
    ]


def _aggregate_weighted_updates(updates: Sequence[Tuple[List[np.ndarray], float]]) -> List[np.ndarray]:
    total_weight = sum(weight for _, weight in updates)
    if not updates or total_weight <= 0:
        raise ValueError("Weighted aggregation requires updates with positive total weight")
    return [
        sum(parameters[layer] * (weight / total_weight) for parameters, weight in updates)
        for layer in range(len(updates[0][0]))
    ]


def _inverse_participation_weights(
    updates: Dict[str, Tuple[List[np.ndarray], int]],
    participation: Dict[str, int],
    profiles: Dict[str, str],
    oracle_group_balance: bool = False,
) -> List[Tuple[List[np.ndarray], float]]:
    raw_weights = {
        cid: sample_count / max(participation[cid], 1)
        for cid, (_, sample_count) in updates.items()
    }
    if not oracle_group_balance:
        return [(updates[cid][0], weight) for cid, weight in raw_weights.items()]
    groups = {
        profile: [cid for cid in raw_weights if profiles[cid] == profile]
        for profile in ("heavy-use", "light-use")
    }
    present_groups = [cids for cids in groups.values() if cids]
    group_share = 1.0 / len(present_groups) if present_groups else 0.0
    weights = {}
    for cids in present_groups:
        group_total = sum(raw_weights[cid] for cid in cids)
        for cid in cids:
            weights[cid] = group_share * raw_weights[cid] / group_total
    return [(updates[cid][0], weights[cid]) for cid in weights]


def _final_window_metrics(round_history: Sequence[dict], window_size: int) -> Tuple[float, List[float]]:
    final_window = round_history[-window_size:]
    return (
        mean(row["test_accuracy"] for row in final_window),
        [
            mean(row["per_class_accuracy"][label] for row in final_window)
            for label in range(10)
        ],
    )


def run_one(
    config: dict,
    policy: str,
    rho: float,
    seed: int,
    client_loaders: Sequence[DataLoader],
    test_loader: DataLoader,
    thresholds: Optional[dict] = None,
) -> Dict[str, Any]:
    sim_cfg = config["simulation"]
    telemetry_cfg = config["telemetry"]
    assumptions = config["assumptions"]
    rules = deepcopy(config["guardrails"])
    if thresholds:
        rules["power"].update(thresholds.get("power", {}))
        rules["network"].update(thresholds.get("network", {}))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(sim_cfg["torch_num_threads"])
    device = torch.device("cpu")
    fleet = FleetTelemetrySimulator(
        num_clients=sim_cfg["fleet_size"],
        seed=seed,
        battery_capacity_kwh=telemetry_cfg["battery_capacity_kwh"],
        training_energy_per_round_kwh=telemetry_cfg["training_energy_per_round_kwh"],
        transmission_power_w=telemetry_cfg["transmission_power_w"],
        profile_share=assumptions["profile_share"],
        initial_state_probabilities=assumptions["initial_state_probabilities"],
        transition_matrices=assumptions["availability_transition_matrices"],
        dropout_assumptions=assumptions["midround_dropout"],
    )
    orchestrator = Orchestrator(
        min_battery_general=rules["power"]["min_battery_general"],
        min_battery_dc_fast=rules["power"]["min_battery_dc_fast"],
        required_network=NetworkType(rules["network"]["required_network_type"]),
        min_bandwidth_mbps=rules["network"]["min_bandwidth_mbps"],
    )
    model = EdgeCNN().to(device)
    initial_accuracy, initial_classes = _evaluate_by_class(model, test_loader, device)
    last_accuracy, last_classes = initial_accuracy, initial_classes
    participation = {cid: 0 for cid in fleet.clients}
    successful_participation = {cid: 0 for cid in fleet.clients}
    last_participation = {cid: 0 for cid in fleet.clients}
    client_class_counts = _client_class_counts(client_loaders)
    round_history = []
    wasted_client_rounds = 0
    rounds_to_90 = None
    model_size_bytes = get_model_size_bytes(model)
    availability_trace = []
    coverage_since_eval = [0] * 10
    coverage_rounds_since_eval = 0

    for round_num in range(1, sim_cfg["num_rounds"] + 1):
        fleet.advance_round(round_num)
        availability_trace.append([
            fleet.clients[cid].availability_state.value
            for cid in sorted(fleet.clients)
        ])
        selected, available_count, expected_survivors = _select_clients(
            policy,
            fleet.clients,
            orchestrator,
            sim_cfg["clients_per_round"],
            fleet.selection_rng,
            assumptions,
            participation,
            last_participation,
            round_num,
        )
        normal_eligible, _ = orchestrator.filter_eligible_clients(fleet.clients)
        eligible_by_profile = {
            profile: sum(fleet.clients[cid].usage_profile == profile for cid in normal_eligible)
            for profile in ("heavy-use", "light-use")
        }
        if policy in ("P2P", "P2P-2"):
            connectable = [
                telemetry
                for telemetry in fleet.clients.values()
                if telemetry.availability_state == AvailabilityState.PARKED_WIFI
                and telemetry.network_type == NetworkType.UNMETERED_WIFI
            ]
            eligible_by_profile = {
                profile: sum(telemetry.usage_profile == profile for telemetry in connectable)
                for profile in ("heavy-use", "light-use")
            }
        if policy == "F":
            eligible_by_profile["heavy-use"] += max(0, available_count - len(normal_eligible))
        updates = {}
        failed_this_round = 0
        for cid in selected:
            participation[cid] += 1
            last_participation[cid] = round_num
            survived, _ = fleet.simulate_mid_round_dynamics(
                cid, is_baseline=policy not in ("C", "D", "E1", "E2", "F", "P2P", "P2P-2")
            )
            if not survived:
                failed_this_round += 1
                fleet.record_resource_usage(cid, 1, model_size_bytes, completed=False)
                continue

            local_model = EdgeCNN().to(device)
            set_parameters(local_model, get_parameters(model))
            train(
                local_model,
                client_loaders[int(cid.rsplit("_", 1)[1])],
                epochs=sim_cfg["local_epochs"],
                learning_rate=sim_cfg["learning_rate"],
                device=device,
            )
            updates[cid] = (
                get_parameters(local_model),
                len(client_loaders[int(cid.rsplit("_", 1)[1])].dataset),
            )
            successful_participation[cid] += 1
            fleet.record_resource_usage(cid, sim_cfg["local_epochs"], model_size_bytes, completed=True)

        wasted_client_rounds += failed_this_round
        peer_pairs_merged = 0
        if policy in ("P2P", "P2P-2"):
            aggregate_updates = []
            for pair_start in range(0, len(selected), 2):
                pair = selected[pair_start:pair_start + 2]
                if len(pair) != 2 or any(cid not in updates for cid in pair):
                    continue
                pair_updates = [updates[cid] for cid in pair]
                peer_pairs_merged += 1
                pair_examples = sum(weight for _, weight in pair_updates)
                aggregate_updates.append((
                    [
                        sum(
                            parameters[index] * (weight / pair_examples)
                            for parameters, weight in pair_updates
                        )
                        for index in range(len(pair_updates[0][0]))
                    ],
                    pair_examples,
                ))
        elif policy in ("E1", "E2"):
            weighted_updates = _inverse_participation_weights(
                updates,
                participation,
                fleet.usage_profiles,
                oracle_group_balance=policy == "E2",
            )
            aggregate_updates = list(updates.values())
        else:
            aggregate_updates = list(updates.values())

        if aggregate_updates:
            if policy in ("E1", "E2"):
                aggregated = _aggregate_weighted_updates(weighted_updates)
            else:
                aggregated = _aggregate_updates(aggregate_updates)
            set_parameters(model, aggregated)
        round_class_coverage = [
            sum(client_class_counts[int(cid.rsplit("_", 1)[1])][label] > 0 for cid in selected)
            for label in range(10)
        ]
        selected_class_examples = [
            sum(
                client_class_counts[int(cid.rsplit("_", 1)[1])][label]
                for cid in selected
            )
            for label in range(10)
        ]
        selected_example_total = sum(selected_class_examples)
        successful_class_examples = [
            sum(
                client_class_counts[int(cid.rsplit("_", 1)[1])][label]
                for cid in updates
            )
            for label in range(10)
        ]
        selected_class_mix_pct = [
            (count / selected_example_total * 100.0) if selected_example_total else 0.0
            for count in selected_class_examples
        ]
        coverage_since_eval = [
            coverage_since_eval[label] + round_class_coverage[label]
            for label in range(10)
        ]
        coverage_rounds_since_eval += 1
        evaluation_round = (
            round_num % sim_cfg["evaluation_interval"] == 0
            or round_num >= sim_cfg["num_rounds"] - sim_cfg["final_metric_rounds"] + 1
            or round_num == sim_cfg["num_rounds"]
        )
        class_coverage_window = None
        if evaluation_round:
            if aggregate_updates:
                last_accuracy, last_classes = _evaluate_by_class(model, test_loader, device)
            class_coverage_window = [
                value / coverage_rounds_since_eval
                for value in coverage_since_eval
            ]
            coverage_since_eval = [0] * 10
            coverage_rounds_since_eval = 0
            if rounds_to_90 is None and last_accuracy >= 0.90:
                rounds_to_90 = round_num

        summary = fleet.get_summary_metrics()
        round_history.append({
            "round": round_num,
            "test_accuracy": last_accuracy * 100.0,
            "worst_class_accuracy": min(last_classes) * 100.0,
            "heavy_class_accuracy": mean(last_classes[5:]) * 100.0,
            "selected_count": len(selected),
            "successful_count": len(updates),
            "aggregated_client_count": len(aggregate_updates),
            "peer_pairs_merged": peer_pairs_merged,
            "wasted_client_rounds": failed_this_round,
            "eligible_count": available_count,
            "eligible_pool_by_profile": eligible_by_profile,
            "selected_clients": selected,
            "expected_survivors": expected_survivors,
            "class_coverage": round_class_coverage,
            "selected_class_examples": selected_class_examples,
            "successful_class_examples": successful_class_examples,
            "selected_class_mix_pct": selected_class_mix_pct,
            "class_coverage_window": class_coverage_window,
            "evaluated": evaluation_round,
            "per_class_accuracy": [value * 100.0 for value in last_classes],
            "total_energy_kwh": summary["total_energy_kwh"],
            "metered_mb": summary["total_metered_mb"],
        })

    summary = fleet.get_summary_metrics()
    participation_values = list(participation.values())
    selected_client_rounds = sum(participation_values)
    successful_client_rounds = sum(row["successful_count"] for row in round_history)
    selected_class_mass = [
        sum(
            participation[cid] * client_class_counts[int(cid.rsplit("_", 1)[1])][label]
            for cid in participation
        )
        for label in range(10)
    ]
    successful_class_mass = [
        sum(
            successful_participation[cid] * client_class_counts[int(cid.rsplit("_", 1)[1])][label]
            for cid in successful_participation
        )
        for label in range(10)
    ]
    total_successful_class_mass = sum(successful_class_mass)
    successful_class_mix_pct = [
        (100.0 * value / total_successful_class_mass) if total_successful_class_mass else 0.0
        for value in successful_class_mass
    ]
    total_selected_class_mass = sum(selected_class_mass)
    selected_class_mix_pct = [
        100.0 * value / total_selected_class_mass if total_selected_class_mass else 0.0
        for value in selected_class_mass
    ]
    final_window_accuracy, final_window_classes = _final_window_metrics(
        round_history,
        sim_cfg["final_metric_rounds"],
    )
    light_class_accuracy = mean(final_window_classes[:5])
    heavy_class_accuracy = mean(final_window_classes[5:])
    worst_class_label = min(range(10), key=lambda label: final_window_classes[label])
    participation_square_sum = sum(value * value for value in participation_values)
    jain_index = (
        (sum(participation_values) ** 2) / (len(participation_values) * participation_square_sum)
        if participation_square_sum
        else 1.0
    )
    coverage_accuracy_correlation = []
    for label in range(10):
        measured_rows = [row for row in round_history if row["evaluated"]]
        coverage_values = [row["class_coverage_window"][label] for row in measured_rows]
        accuracy_values = [row["per_class_accuracy"][label] for row in measured_rows]
        if np.std(coverage_values) > 0 and np.std(accuracy_values) > 0:
            coverage_accuracy_correlation.append(float(np.corrcoef(coverage_values, accuracy_values)[0, 1]))
    return {
        "policy": policy,
        "policy_name": POLICY_NAMES[policy],
        "rho": rho,
        "seed": seed,
        "initial_accuracy_pct": initial_accuracy * 100.0,
        "final_accuracy_pct": final_window_accuracy,
        "last_round_accuracy_pct": last_accuracy * 100.0,
        "rounds_to_90": rounds_to_90,
        "expected_survivors_per_round": mean([row["expected_survivors"] for row in round_history]),
        "mean_class_accuracy_pct": mean(final_window_classes),
        "worst_class_accuracy_pct": min(final_window_classes),
        "worst_class_label": worst_class_label,
        "worst_class_is_heavy_group": worst_class_label >= 5,
        "light_class_accuracy_pct": light_class_accuracy,
        "heavy_class_accuracy_pct": heavy_class_accuracy,
                "class_accuracy_gap_heavy_minus_light_pct": heavy_class_accuracy - light_class_accuracy,
        "per_class_accuracy_pct": final_window_classes,
        "coverage_accuracy_correlation_mean": mean(coverage_accuracy_correlation) if coverage_accuracy_correlation else None,
        "wasted_client_rounds": wasted_client_rounds,
        "total_energy_kwh": summary["total_energy_kwh"],
        "metered_mb": summary["total_metered_mb"],
        "participation_min_rounds": min(participation_values),
        "participation_max_rounds": max(participation_values),
        "vehicles_below_5_rounds": sum(value < 5 for value in participation_values),
        "any_vehicle_below_5_rounds": any(value < 5 for value in participation_values),
        "jain_participation_index": jain_index,
        "participation_weighted_class_mix_pct": successful_class_mix_pct,
        "participation_weighted_heavy_class_share_pct": sum(successful_class_mix_pct[5:]),
        "selected_class_mass_pct": selected_class_mix_pct,
        "successful_class_mass_pct": successful_class_mix_pct,
        "successful_participation_by_client": successful_participation,
        "selected_client_rounds": selected_client_rounds,
        "successful_client_rounds": successful_client_rounds,
        "energy_kwh_per_selected_client_round": summary["total_energy_kwh"] / max(selected_client_rounds, 1),
        "metered_mb_per_selected_client_round": summary["total_metered_mb"] / max(selected_client_rounds, 1),
        "participation_by_client": participation,
        "profile_map": fleet.usage_profiles,
        "availability_trace_sha256": hashlib.sha256(
            json.dumps(availability_trace, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "round_history": round_history,
        "fleet_profiles": fleet.usage_profiles,
        "participation_profile_summary": {
            profile: {
                "mean": mean(profile_counts.values()) if profile_counts else None,
                "counts": profile_counts,
            }
            for profile in ("heavy-use", "light-use")
            for profile_counts in [{
                cid: participation[cid]
                for cid, value in fleet.usage_profiles.items()
                if value == profile
            }]
        },
        "telemetry_summary": summary,
    }


_T_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045,
}


def _mean_std(values: Sequence[Optional[float]]) -> dict:
    present = [float(value) for value in values if value is not None]
    average = mean(present) if present else None
    deviation = stdev(present) if len(present) > 1 else (0.0 if present else None)
    critical = _T_975.get(len(present) - 1, 1.96) if present else None
    half_width = critical * deviation / math.sqrt(len(present)) if len(present) > 1 else None
    return {
        "mean": average,
        "std": deviation,
        "ci95_lower": average - half_width if average is not None and half_width is not None else None,
        "ci95_upper": average + half_width if average is not None and half_width is not None else None,
        "n": len(present),
    }


def _paired_interval(values: Sequence[float], bootstrap_samples: int = 10_000, seed: int = 0) -> dict:
    summary = _mean_std(values)
    present = np.asarray(values, dtype=float)
    if len(present) == 0:
        return {**summary, "bootstrap_ci95_lower": None, "bootstrap_ci95_upper": None}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(present), size=(bootstrap_samples, len(present)))
    bootstrap_means = present[indices].mean(axis=1)
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
    return {
        **summary,
        "bootstrap_ci95_lower": float(lower),
        "bootstrap_ci95_upper": float(upper),
        "bootstrap_samples": bootstrap_samples,
    }


def _one_sided_pvalue(values: Sequence[float], alternative: str) -> float:
    from scipy.stats import ttest_1samp

    if not values:
        return 1.0
    result = ttest_1samp(values, popmean=0.0, alternative=alternative)
    if np.isnan(result.pvalue):
        return 1.0
    return float(result.pvalue)


def _holm_adjust(hypothesis_tests: Dict[str, dict]) -> dict:
    ordered = sorted(hypothesis_tests.items(), key=lambda pair: pair[1]["raw_pvalue"])
    count = len(ordered)
    adjusted_so_far = 0.0
    adjusted = {}
    for rank, (name, result) in enumerate(ordered):
        corrected = min(1.0, (count - rank) * result["raw_pvalue"])
        adjusted_so_far = max(adjusted_so_far, corrected)
        adjusted[name] = {**result, "holm_pvalue": adjusted_so_far}
    return {name: adjusted[name] for name in hypothesis_tests}


def _summarize(results: Sequence[dict]) -> dict:
    metrics = [
        "final_accuracy_pct",
        "mean_class_accuracy_pct",
        "rounds_to_90",
        "worst_class_accuracy_pct",
        "light_class_accuracy_pct",
        "heavy_class_accuracy_pct",
        "class_accuracy_gap_heavy_minus_light_pct",
        "coverage_accuracy_correlation_mean",
        "wasted_client_rounds",
        "total_energy_kwh",
        "metered_mb",
        "participation_min_rounds",
        "participation_max_rounds",
        "vehicles_below_5_rounds",
        "any_vehicle_below_5_rounds",
        "jain_participation_index",
        "participation_weighted_heavy_class_share_pct",
        "energy_kwh_per_selected_client_round",
        "metered_mb_per_selected_client_round",
        "expected_survivors_per_round",
    ]
    summary = {
        metric: _mean_std([result[metric] for result in results])
        for metric in metrics
    }
    for label in range(10):
        summary[f"class_{label}_accuracy_pct"] = _mean_std([
            result["per_class_accuracy_pct"][label]
            for result in results
        ])
    summary["worst_class_heavy_group_frequency_pct"] = _mean_std([
        100.0 * result["worst_class_is_heavy_group"]
        for result in results
    ])
    summary["rounds_to_90_reach_fraction_pct"] = _mean_std([
        100.0 * (result["rounds_to_90"] is not None)
        for result in results
    ])
    reached = [result["rounds_to_90"] for result in results if result["rounds_to_90"] is not None]
    summary["rounds_to_90_median_if_reached"] = {
        "median": float(np.median(reached)) if reached else None,
        "reached_seeds": len(reached),
        "total_seeds": len(results),
    }
    summary["runs_any_vehicle_below_5_rounds_pct"] = _mean_std([
        100.0 * result["any_vehicle_below_5_rounds"]
        for result in results
    ])
    return summary


def _summarize_sensitivity(results: Sequence[dict]) -> dict:
    metrics = [
        "final_accuracy_pct",
        "mean_class_accuracy_pct",
        "worst_class_accuracy_pct",
        "class_accuracy_gap_heavy_minus_light_pct",
        "total_energy_kwh",
        "metered_mb",
        "participation_max_rounds",
    ]
    return {
        metric: _mean_std([result[metric] for result in results])
        for metric in metrics
    }


def _paired_differences(results: Sequence[dict], policies: Sequence[str]) -> dict:
    paired = {}
    by_key = {(row["rho"], row["policy"], row["seed"]): row for row in results}
    metrics = [
        "final_accuracy_pct",
        "worst_class_accuracy_pct",
        "heavy_class_accuracy_pct",
        "class_accuracy_gap_heavy_minus_light_pct",
        "participation_weighted_heavy_class_share_pct",
        "vehicles_below_5_rounds",
        "jain_participation_index",
        "total_energy_kwh",
        "metered_mb",
    ]
    for rho in sorted({row["rho"] for row in results}):
        for policy in policies:
            if policy == "A":
                continue
            differences = {}
            for metric in metrics:
                values = []
                for seed in sorted({row["seed"] for row in results if row["rho"] == rho}):
                    candidate = by_key.get((rho, policy, seed))
                    baseline = by_key.get((rho, "A", seed))
                    if candidate is not None and baseline is not None:
                        values.append(candidate[metric] - baseline[metric])
                differences[metric] = _paired_interval(
                    values,
                    bootstrap_samples=10_000,
                    seed=int(hashlib.sha256(f"{rho}|{policy}|{metric}".encode("ascii")).hexdigest()[:8], 16),
                )
            paired[f"rho={rho:g}/policy={policy}-minus-A"] = differences
    return paired


def _registered_hypotheses(results: Sequence[dict]) -> dict:
    by_key = {(row["rho"], row["policy"], row["seed"]): row for row in results}
    seeds = sorted({row["seed"] for row in results if row["rho"] == 1.0})

    def paired(policy: str, metric: str, against: str = "A") -> List[float]:
        return [
            by_key[(1.0, policy, seed)][metric] - by_key[(1.0, against, seed)][metric]
            for seed in seeds
            if (1.0, policy, seed) in by_key and (1.0, against, seed) in by_key
        ]

    h1_worst = paired("C", "worst_class_accuracy_pct")
    h1_heavy = paired("C", "heavy_class_accuracy_pct")
    h1_p = max(
        _one_sided_pvalue(h1_worst, "less"),
        _one_sided_pvalue(h1_heavy, "less"),
    )
    h2_tests = []
    for policy in ("E1", "F"):
        h2_tests.extend([
            _one_sided_pvalue(paired(policy, "worst_class_accuracy_pct", "C"), "greater"),
            _one_sided_pvalue(paired(policy, "heavy_class_accuracy_pct", "C"), "greater"),
        ])
    h2_p = max(h2_tests) if h2_tests else 1.0
    h3_tests = [
        _one_sided_pvalue(paired("B", "metered_mb", "C"), "greater"),
        _one_sided_pvalue(paired("B", "total_energy_kwh", "C"), "greater"),
    ]
    h3_p = max(h3_tests) if all(name in by_key for name in [(1.0, "B", seeds[0]), (1.0, "C", seeds[0])]) else 1.0
    h4_tests = [
        _one_sided_pvalue(paired("C", "vehicles_below_5_rounds"), "greater"),
        _one_sided_pvalue(paired("D", "vehicles_below_5_rounds"), "greater"),
    ]
    tests = {
        "H1_C_reduces_worst_and_heavy_class_accuracy": {
            "raw_pvalue": h1_p,
            "component_pvalues": {
                "worst_class": _one_sided_pvalue(h1_worst, "less"),
                "classes_5_9": _one_sided_pvalue(h1_heavy, "less"),
            },
            "direction": "C - A < 0 for both outcomes; intersection-union p-value is the maximum component p-value",
        },
        "H2_E1_and_F_improve_both_accuracy_metrics_vs_C": {
            "raw_pvalue": h2_p,
            "component_pvalues": h2_tests,
            "direction": "E1 and F - C > 0 for both outcomes; intersection-union p-value is the maximum component p-value",
        },
        "H3_B_uses_more_metered_data_and_energy_than_C": {
            "raw_pvalue": h3_p,
            "component_pvalues": {
                "metered_mb": h3_tests[0],
                "energy_kwh": h3_tests[1],
            },
            "direction": "B - C > 0 for both resources; intersection-union p-value is the maximum component p-value",
        },
        "H4_C_and_D_have_more_vehicles_below_5_rounds_than_A": {
            "raw_pvalue": max(h4_tests),
            "component_pvalues": {
                "C_vs_A": h4_tests[0],
                "D_vs_A": h4_tests[1],
            },
            "direction": "C and D - A > 0; intersection-union p-value is the maximum component p-value",
        },
    }
    return {
        "primary_rho": 1.0,
        "paired_seeds": seeds,
        "familywise_method": "Holm step-down across H1-H4; one-sided paired t-tests; composite hypotheses use intersection-union max p-value",
        "hypotheses": _holm_adjust(tests),
        "other_comparisons": "Exploratory; no multiplicity adjustment claimed.",
    }


def run_staleness_bonus_diagnostic(config: dict, seeds: Optional[Sequence[int]] = None) -> dict:
    """Compare C and D selection fairness without local-training runtime."""
    sim_cfg = config["simulation"]
    assumptions = config["assumptions"]
    if seeds is None:
        seeds = config["study"]["seeds"]
    configured_bonus = assumptions["policy_d"]["staleness_bonus"]
    bonus_values = [configured_bonus, 20.0]
    runs = []
    for seed in seeds:
        for policy, bonus in [("C", None), ("D", configured_bonus), ("D", 20.0)]:
            fleet = FleetTelemetrySimulator(
                num_clients=sim_cfg["fleet_size"],
                seed=seed,
                battery_capacity_kwh=config["telemetry"]["battery_capacity_kwh"],
                training_energy_per_round_kwh=config["telemetry"]["training_energy_per_round_kwh"],
                transmission_power_w=config["telemetry"]["transmission_power_w"],
                profile_share=assumptions["profile_share"],
                initial_state_probabilities=assumptions["initial_state_probabilities"],
                transition_matrices=assumptions["availability_transition_matrices"],
                dropout_assumptions=assumptions["midround_dropout"],
            )
            orchestrator = Orchestrator(
                min_battery_general=config["guardrails"]["power"]["min_battery_general"],
                min_battery_dc_fast=config["guardrails"]["power"]["min_battery_dc_fast"],
                required_network=NetworkType(config["guardrails"]["network"]["required_network_type"]),
                min_bandwidth_mbps=config["guardrails"]["network"]["min_bandwidth_mbps"],
            )
            local_assumptions = deepcopy(assumptions)
            if policy == "D":
                local_assumptions["policy_d"]["staleness_bonus"] = bonus
            participation = {cid: 0 for cid in fleet.clients}
            last_participation = {cid: 0 for cid in fleet.clients}
            for round_num in range(1, sim_cfg["num_rounds"] + 1):
                fleet.advance_round(round_num)
                selected, _, _ = _select_clients(
                    policy,
                    fleet.clients,
                    orchestrator,
                    sim_cfg["clients_per_round"],
                    fleet.selection_rng,
                    local_assumptions,
                    participation,
                    last_participation,
                    round_num,
                )
                for cid in selected:
                    participation[cid] += 1
                    last_participation[cid] = round_num
            values = list(participation.values())
            runs.append({
                "seed": seed,
                "policy": policy,
                "staleness_bonus": bonus if policy == "D" else 0.0,
                "participation_by_client": participation,
                "fleet_profiles": fleet.usage_profiles,
                "participation_min_rounds": min(values),
                "participation_max_rounds": max(values),
                "participation_spread_rounds": max(values) - min(values),
                "vehicles_below_5_rounds": sum(value < 5 for value in values),
                "any_vehicle_below_5_rounds": any(value < 5 for value in values),
                "heavy_use_mean_participation": mean(
                    participation[cid]
                    for cid, profile in fleet.usage_profiles.items()
                    if profile == "heavy-use"
                ),
                "light_use_mean_participation": mean(
                    participation[cid]
                    for cid, profile in fleet.usage_profiles.items()
                    if profile == "light-use"
                ),
            })
    grouped = {}
    for run in runs:
        key = f"{run['policy']}/bonus={run['staleness_bonus']:g}"
        grouped.setdefault(key, []).append(run)
    summary = {
        key: {
            metric: _mean_std([run[metric] for run in group])
            for metric in (
                "participation_min_rounds",
                "participation_max_rounds",
                "participation_spread_rounds",
                "vehicles_below_5_rounds",
                "any_vehicle_below_5_rounds",
                "heavy_use_mean_participation",
                "light_use_mean_participation",
            )
        }
        for key, group in grouped.items()
    }
    paired_differences = {}
    for bonus in bonus_values:
        paired_differences[f"D/bonus={bonus:g}-minus-C"] = {
            metric: _mean_std([
                next(
                    run[metric]
                    for run in runs
                    if run["seed"] == seed
                    and run["policy"] == "D"
                    and run["staleness_bonus"] == bonus
                ) - next(
                    run[metric]
                    for run in runs
                    if run["seed"] == seed and run["policy"] == "C"
                )
                for seed in seeds
            ])
            for metric in (
                "participation_min_rounds",
                "participation_spread_rounds",
                "vehicles_below_5_rounds",
                "heavy_use_mean_participation",
                "light_use_mean_participation",
            )
        }
    return {
        "description": "Selection-only diagnostic; no model training or resource accounting.",
        "seeds": list(seeds),
        "num_rounds": sim_cfg["num_rounds"],
        "bonus_values": [configured_bonus, 20.0],
        "summary_mean_std_ci95": summary,
        "paired_differences_vs_C": paired_differences,
        "runs": runs,
    }


def refresh_saved_report(path: str) -> dict:
    """Correct derived percentages in a saved result artifact and rebuild summaries."""
    with open(path, "r", encoding="utf-8") as results_file:
        report = json.load(results_file)
    results = report.get("results", [])
    if not results:
        raise ValueError("Results artifact has no per-run results to refresh")
    for result in results:
        per_class = result["per_class_accuracy_pct"]
        result["worst_class_accuracy_pct"] = min(per_class)
        participation = result["participation_by_client"]
        counts = list(participation.values())
        result["participation_min_rounds"] = min(counts)
        result["participation_max_rounds"] = max(counts)
        result["vehicles_below_5_rounds"] = sum(count < 5 for count in counts)
        result["any_vehicle_below_5_rounds"] = any(count < 5 for count in counts)
        result["selected_client_rounds"] = sum(counts)
        result["successful_client_rounds"] = sum(
            row["successful_count"] for row in result["round_history"]
        )
        result["energy_kwh_per_selected_client_round"] = result["total_energy_kwh"] / max(sum(counts), 1)
        result["metered_mb_per_selected_client_round"] = result["metered_mb"] / max(sum(counts), 1)
    partition_cache = {}
    sim_cfg = report["config"]["simulation"]
    assumptions = report["config"]["assumptions"]
    for result in results:
        cache_key = (result["rho"], result["seed"])
        if cache_key not in partition_cache:
            profile_map = result["fleet_profiles"]
            loaders, _ = partition_dataset(
                num_clients=sim_cfg["fleet_size"],
                batch_size=sim_cfg["batch_size"],
                data_dir=sim_cfg["data_dir"],
                seed=result["seed"],
                usage_profiles=[profile_map[f"ev_client_{index:02d}"] for index in range(sim_cfg["fleet_size"])],
                rho=result["rho"],
                max_client_examples=sim_cfg["max_client_examples"],
                test_sample_size=sim_cfg["test_sample_size"],
                dirichlet_alpha=assumptions["non_iid_dirichlet_alpha"],
                favored_class_weight=assumptions["profile_class_allocation_weight"]["favored"],
                non_favored_class_weight=assumptions["profile_class_allocation_weight"]["non_favored"],
            )
            partition_cache[cache_key] = _client_class_counts(loaders)
        client_counts = partition_cache[cache_key]
        participation_mass = [
            sum(
                result["participation_by_client"][f"ev_client_{index:02d}"] * client_counts[index][label]
                for index in range(sim_cfg["fleet_size"])
            )
            for label in range(10)
        ]
        total_mass = sum(participation_mass)
        result["participation_weighted_class_mix_pct"] = [
            count / total_mass * 100.0 if total_mass else 0.0
            for count in participation_mass
        ]
        result["participation_weighted_light_class_share_pct"] = sum(result["participation_weighted_class_mix_pct"][:5])
        result["participation_weighted_heavy_class_share_pct"] = sum(result["participation_weighted_class_mix_pct"][5:])
    grouped = {}
    for result in results:
        key = f"rho={result['rho']:g}/policy={result['policy']}"
        grouped.setdefault(key, []).append(result)
    report["summary_mean_std"] = {key: _summarize(group) for key, group in grouped.items()}
    policies = list(dict.fromkeys(result["policy"] for result in results))
    report["paired_policy_differences_vs_A"] = _paired_differences(results, policies)
    report["profile_assignments_fixed_across_seeds"] = len({
        tuple(sorted(result["fleet_profiles"].items()))
        for result in results
        if result["policy"] == "A"
    }) == 1
    report["metric_refresh"] = {
        "worst_class_accuracy_pct": "Recomputed as min(per_class_accuracy_pct); removed erroneous extra factor of 100.",
        "participation_diagnostics": "Participation minima, max-min spread, under-five counts, and selected-client-round-normalized resource metrics were recomputed from per-run records.",
        "participation_weighted_class_mix_pct": "Recomputed from each seed's recorded client profiles, partition labels, and per-client participation counts.",
        "profile_assignment_note": "Historical profile assignments are retained as recorded. The current simulator now uses seed-invariant client-ID assignments; this artifact predates that correction and reports profile_assignments_fixed_across_seeds=false.",
    }
    with open(path, "w", encoding="utf-8") as results_file:
        json.dump(report, results_file, indent=2)
    return report


def run_study(config: dict) -> dict:
    sim_cfg = config["simulation"]
    sweep_cfg = config["study"]
    seeds = sweep_cfg["seeds"]
    rho_values = sweep_cfg["rho_values"]
    policies = sweep_cfg["policies"]
    output_dir = config["benchmarks"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    data_cache = {}

    def get_data(experiment_config: dict, rho: float, seed: int) -> Tuple[Sequence[DataLoader], DataLoader]:
        experiment_sim = experiment_config["simulation"]
        assumptions = experiment_config["assumptions"]
        cache_key = (
            experiment_sim["fleet_size"], assumptions["profile_share"], rho, seed,
            experiment_sim["max_client_examples"],
        )
        if cache_key not in data_cache:
            profile_fleet = FleetTelemetrySimulator(
                num_clients=experiment_sim["fleet_size"],
                seed=seed,
                profile_share=assumptions["profile_share"],
                initial_state_probabilities=assumptions["initial_state_probabilities"],
                transition_matrices=assumptions["availability_transition_matrices"],
                dropout_assumptions=assumptions["midround_dropout"],
            )
            data_cache[cache_key] = partition_dataset(
                num_clients=experiment_sim["fleet_size"],
                batch_size=experiment_sim["batch_size"],
                data_dir=experiment_sim["data_dir"],
                seed=seed,
                usage_profiles=[profile_fleet.usage_profiles[f"ev_client_{index:02d}"] for index in range(experiment_sim["fleet_size"])],
                rho=rho,
                max_client_examples=experiment_sim["max_client_examples"],
                test_sample_size=experiment_sim["test_sample_size"],
                dirichlet_alpha=assumptions["non_iid_dirichlet_alpha"],
                favored_class_weight=assumptions["profile_class_allocation_weight"]["favored"],
                non_favored_class_weight=assumptions["profile_class_allocation_weight"]["non_favored"],
            )
        return data_cache[cache_key]

    results = []
    run_lookup = {}
    core_policies = list(policies) + (["P2P"] if sweep_cfg.get("p2p_policy", False) else [])
    experiment_count = len(seeds) * len(rho_values) * len(core_policies)
    print(f"Core sweep: {experiment_count} runs ({len(seeds)} seeds, {sim_cfg['num_rounds']} rounds each).", flush=True)
    for rho in rho_values:
        for seed in seeds:
            loaders, test_loader = get_data(config, rho, seed)
            for policy in core_policies:
                result = run_one(config, policy, rho, seed, loaders, test_loader)
                results.append(result)
                run_lookup[(rho, policy, seed)] = result
                print(
                    f"  rho={rho:g} policy={policy} seed={seed}: "
                    f"final-10-avg={result['final_accuracy_pct']:.2f}% "
                    f"waste={result['wasted_client_rounds']} "
                    f"energy={result['total_energy_kwh']:.3f} kWh",
                    flush=True,
                )

    sensitivity_results = []
    sensitivity_seeds = sweep_cfg.get("sensitivity_seeds", seeds)
    for sensitivity_name, values in sweep_cfg["sensitivity"].items():
        for value in values:
            for seed in sensitivity_seeds:
                experiment_config = deepcopy(config)
                if sensitivity_name == "profile_share":
                    experiment_config["assumptions"]["profile_share"] = value
                elif sensitivity_name == "driving_dropout_probability":
                    experiment_config["assumptions"]["midround_dropout"]["driving_probability"] = value
                elif sensitivity_name == "low_battery_dropout_probability":
                    experiment_config["assumptions"]["midround_dropout"]["low_battery_probability"] = value
                elif sensitivity_name == "poor_network_dropout_probability":
                    experiment_config["assumptions"]["midround_dropout"]["poor_network_probability"] = value
                elif sensitivity_name == "fleet_size":
                    experiment_config["simulation"]["fleet_size"] = int(value)
                elif sensitivity_name == "clients_per_round":
                    experiment_config["simulation"]["clients_per_round"] = int(value)
                    experiment_config["assumptions"]["over_selection"]["expected_survivors_target"] = int(value)
                else:
                    raise ValueError(f"Unknown sensitivity parameter: {sensitivity_name}")
                loaders, test_loader = get_data(experiment_config, 1.0, seed)
                result = run_one(experiment_config, "C", 1.0, seed, loaders, test_loader)
                sensitivity_results.append({
                    "parameter": sensitivity_name,
                    "value": value,
                    "rho": 1.0,
                    "policy": "C",
                    "seed": seed,
                    "final_accuracy_pct": result["final_accuracy_pct"],
                    "mean_class_accuracy_pct": result["mean_class_accuracy_pct"],
                    "worst_class_accuracy_pct": result["worst_class_accuracy_pct"],
                    "class_accuracy_gap_heavy_minus_light_pct": result["class_accuracy_gap_heavy_minus_light_pct"],
                    "total_energy_kwh": result["total_energy_kwh"],
                    "metered_mb": result["metered_mb"],
                    "participation_max_rounds": result["participation_max_rounds"],
                })
            print(f"  sensitivity {sensitivity_name}={value} complete", flush=True)

    sensitivity = sweep_cfg["threshold_sensitivity"]
    for threshold_name, values in sensitivity.items():
        if threshold_name == "min_battery_general":
            section, config_key = "power", "min_battery_general"
        elif threshold_name == "min_battery_dc_fast":
            section, config_key = "power", "min_battery_dc_fast"
        else:
            section, config_key = "network", "min_bandwidth_mbps"
        baseline = config["guardrails"][section][config_key]
        for value in values:
            if value == baseline:
                for seed in sensitivity_seeds:
                    result = run_lookup[(1.0, "C", seed)]
                    sensitivity_results.append({
                    "parameter": threshold_name,
                        "threshold": threshold_name,
                        "value": value,
                        "rho": 1.0,
                        "policy": "C",
                        "seed": seed,
                        "final_accuracy_pct": result["final_accuracy_pct"],
                        "mean_class_accuracy_pct": result["mean_class_accuracy_pct"],
                        "worst_class_accuracy_pct": result["worst_class_accuracy_pct"],
                        "class_accuracy_gap_heavy_minus_light_pct": result["class_accuracy_gap_heavy_minus_light_pct"],
                        "participation_max_rounds": result["participation_max_rounds"],
                        "heavy_class_accuracy_pct": result["heavy_class_accuracy_pct"],
                        "total_energy_kwh": result["total_energy_kwh"],
                        "metered_mb": result["metered_mb"],
                    })
                continue
            for seed in sensitivity_seeds:
                loaders, test_loader = get_data(config, 1.0, seed)
                threshold_patch = {section: {config_key: value}}
                result = run_one(config, "C", 1.0, seed, loaders, test_loader, threshold_patch)
                sensitivity_results.append({
                    "parameter": threshold_name,
                    "threshold": threshold_name,
                    "value": value,
                    "rho": 1.0,
                    "policy": "C",
                    "seed": seed,
                    "final_accuracy_pct": result["final_accuracy_pct"],
                    "mean_class_accuracy_pct": result["mean_class_accuracy_pct"],
                    "worst_class_accuracy_pct": result["worst_class_accuracy_pct"],
                    "class_accuracy_gap_heavy_minus_light_pct": result["class_accuracy_gap_heavy_minus_light_pct"],
                    "participation_max_rounds": result["participation_max_rounds"],
                    "heavy_class_accuracy_pct": result["heavy_class_accuracy_pct"],
                    "total_energy_kwh": result["total_energy_kwh"],
                    "metered_mb": result["metered_mb"],
                })
            print(f"  sensitivity {threshold_name}={value} complete", flush=True)

    grouped = {}
    for result in results:
        key = f"rho={result['rho']:g}/policy={result['policy']}"
        grouped.setdefault(key, []).append(result)
    trace_groups = {}
    for result in results:
        trace_groups.setdefault((result["rho"], result["seed"]), set()).add(result["availability_trace_sha256"])
    if any(len(trace_hashes) != 1 for trace_hashes in trace_groups.values()):
        raise RuntimeError("Paired policy runs did not share the same availability trace")
    summary = {key: _summarize(group) for key, group in grouped.items()}
    sensitivity_summary = {}
    for item in sensitivity_results:
        key = f"{item['parameter']}={item['value']}"
        sensitivity_summary.setdefault(key, []).append(item)

    report = {
        "hypotheses": sweep_cfg["hypotheses"],
        "assumptions": config["assumptions"],
        "config": config,
        "summary_mean_std": summary,
        "paired_policy_differences_vs_A": _paired_differences(results, core_policies),
        "paired_availability_trace_count": len(trace_groups),
        "profile_assignments_fixed_across_seeds": len({
            tuple(sorted(result["fleet_profiles"].items()))
            for result in results
            if result["policy"] == "A"
        }) == 1,
        "sensitivity_mean_std": {
            key: _summarize_sensitivity(group)
            for key, group in sensitivity_summary.items()
        },
        "sensitivity_results": sensitivity_results,
        "results": results,
        "staleness_bonus_diagnostic": run_staleness_bonus_diagnostic(config),
    }
    results_path = os.path.join(output_dir, "availability_study.json")
    with open(results_path, "w", encoding="utf-8") as results_file:
        json.dump(report, results_file, indent=2)

    print("\nMean, sample standard deviation, and 95% confidence interval across paired seeds")
    print("rho  policy  final-10 accuracy (%) [95% CI]   mean-class (%) [95% CI]    worst class (%) [95% CI]   light/heavy gap (pp)     energy (kWh) [95% CI]  metered MB [95% CI]")
    for key, metrics in summary.items():
        rho_text, policy_text = key.split("/")
        def cell(metric: str, precision: int = 2) -> str:
            entry = metrics[metric]
            if entry["mean"] is None:
                return "n/a"
            if entry["ci95_lower"] is None:
                return f"{entry['mean']:.{precision}f} (n=1; CI unavailable)"
            return f"{entry['mean']:.{precision}f} +/- {entry['std']:.{precision}f} [{entry['ci95_lower']:.{precision}f}, {entry['ci95_upper']:.{precision}f}]"
        print(
            f"{rho_text[4:]:<4} {policy_text[7:]:<7} "
            f"{cell('final_accuracy_pct'):<40} {cell('mean_class_accuracy_pct'):<38} "
            f"{cell('worst_class_accuracy_pct'):<38} {cell('class_accuracy_gap_heavy_minus_light_pct'):<24} "
            f"{cell('total_energy_kwh', 3):<40} {cell('metered_mb', 2)}"
        )
    print("\nSensitivity at rho=1, policy C (mean [95% CI], sensitivity seed set)")
    print("parameter value  final accuracy (%) [95% CI]    worst class (%) [95% CI]   energy (kWh) [95% CI]")
    for key, metrics in report["sensitivity_mean_std"].items():
        def sensitivity_cell(metric: str, precision: int = 2) -> str:
            entry = metrics[metric]
            if entry["mean"] is None:
                return "n/a"
            if entry["ci95_lower"] is None:
                return f"{entry['mean']:.{precision}f} (CI unavailable)"
            return f"{entry['mean']:.{precision}f} +/- {entry['std']:.{precision}f} [{entry['ci95_lower']:.{precision}f}, {entry['ci95_upper']:.{precision}f}]"
        threshold_name, value = key.split("=", 1)
        print(
            f"{threshold_name:<30} {value:<8} "
            f"{sensitivity_cell('final_accuracy_pct'):<40} "
            f"{sensitivity_cell('worst_class_accuracy_pct'):<38} "
            f"{sensitivity_cell('total_energy_kwh', 3)}"
        )
    print(f"\nDetailed results: {results_path}", flush=True)
    return report


def _git_metadata(config_path: str) -> dict:
    config_file = Path(config_path).resolve()
    repository_root = config_file.parent
    git_executable = shutil.which("git")
    if not git_executable and os.name == "nt":
        candidate = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git" / "cmd" / "git.exe"
        if not candidate.exists():
            candidate = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Git" / "cmd" / "git.exe"
        if candidate.exists():
            git_executable = str(candidate)
    if not git_executable:
        return {
            "commit": None,
            "config_is_committed_and_matches_head": False,
            "error": "Git executable not found on PATH or in standard Windows install locations",
        }
    try:
        commit = subprocess.run(
            [git_executable, "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        config_diff = subprocess.run(
            [git_executable, "diff", "--quiet", "HEAD", "--", "config.yaml"],
            cwd=repository_root,
            check=False,
            capture_output=True,
        )
        return {
            "commit": commit,
            "config_is_committed_and_matches_head": config_diff.returncode == 0,
            "error": config_diff.stderr.decode("utf-8", errors="replace") if config_diff.returncode not in (0, 1) else None,
        }
    except (OSError, subprocess.CalledProcessError) as error:
        return {
            "commit": None,
            "config_is_committed_and_matches_head": False,
            "error": str(error),
        }


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2)
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(temporary_path, path)


def _make_run_path(root: Path, job: dict) -> Path:
    if job["kind"] == "sensitivity":
        value = str(job["parameter_value"]).replace(".", "p")
        return (
            root / "sensitivity" / job["parameter"] / value
            / f"rho_{job['rho']:g}" / f"seed_{job['seed']}" / f"{job['policy']}.json"
        )
    rho = str(job["rho"]).replace(".", "p")
    return root / "primary" / f"rho_{rho}" / f"seed_{job['seed']}" / f"{job['policy']}.json"


def _run_seed_bundle(arguments: tuple) -> List[str]:
    config, jobs, config_hash, git_commit, run_root_text = arguments
    run_root = Path(run_root_text)
    cache = {}
    completed_paths = []
    for job in jobs:
        result_path = _make_run_path(run_root, job)
        run_config = job["config"]
        experiment_fingerprint = _sha256_json({
            "config_hash": config_hash,
            "seed": job["seed"],
            "rho": job["rho"],
            "policy": job["policy"],
            "kind": job["kind"],
            "parameter": job.get("parameter"),
            "parameter_value": job.get("parameter_value"),
            "experiment_config": run_config,
        })
        if result_path.exists():
            try:
                existing = json.loads(result_path.read_text(encoding="utf-8"))
                if existing.get("run_metadata", {}).get("experiment_fingerprint") == experiment_fingerprint:
                    completed_paths.append(str(result_path))
                    continue
            except (OSError, json.JSONDecodeError):
                pass

        sim_cfg = run_config["simulation"]
        assumptions = run_config["assumptions"]
        fleet = FleetTelemetrySimulator(
            num_clients=sim_cfg["fleet_size"],
            seed=job["seed"],
            battery_capacity_kwh=run_config["telemetry"]["battery_capacity_kwh"],
            training_energy_per_round_kwh=run_config["telemetry"]["training_energy_per_round_kwh"],
            transmission_power_w=run_config["telemetry"]["transmission_power_w"],
            profile_share=assumptions["profile_share"],
            initial_state_probabilities=assumptions["initial_state_probabilities"],
            transition_matrices=assumptions["availability_transition_matrices"],
            dropout_assumptions=assumptions["midround_dropout"],
        )
        profile_map = fleet.usage_profiles
        data_key = (
            sim_cfg["fleet_size"],
            assumptions["profile_share"],
            job["rho"],
            job["seed"],
            sim_cfg["max_client_examples"],
            tuple(sorted(profile_map.items())),
        )
        if data_key not in cache:
            cache[data_key] = partition_dataset(
                num_clients=sim_cfg["fleet_size"],
                batch_size=sim_cfg["batch_size"],
                data_dir=sim_cfg["data_dir"],
                seed=job["seed"],
                usage_profiles=[profile_map[f"ev_client_{index:02d}"] for index in range(sim_cfg["fleet_size"])],
                rho=job["rho"],
                max_client_examples=sim_cfg["max_client_examples"],
                test_sample_size=sim_cfg["test_sample_size"],
                dirichlet_alpha=assumptions["non_iid_dirichlet_alpha"],
                favored_class_weight=assumptions["profile_class_allocation_weight"]["favored"],
                non_favored_class_weight=assumptions["profile_class_allocation_weight"]["non_favored"],
            )
        train_loaders, test_loader = cache[data_key]
        result = run_one(
            run_config,
            job["policy"],
            job["rho"],
            job["seed"],
            train_loaders,
            test_loader,
        )
        result["profile_map"] = profile_map
        result["eligible_pool_by_profile_by_round"] = [
            {
                "round": round_record["round"],
                **round_record["eligible_pool_by_profile"],
            }
            for round_record in result["round_history"]
        ]
        result["selected_training_example_mass_by_class_by_round"] = [
            {
                "round": round_record["round"],
                "selected": round_record["selected_class_examples"],
                "successful": round_record["successful_class_examples"],
            }
            for round_record in result["round_history"]
        ]
        result["per_class_accuracy_by_evaluated_round"] = [
            {"round": round_record["round"], "accuracy_pct": round_record["per_class_accuracy"]}
            for round_record in result["round_history"]
            if round_record["evaluated"]
        ]
        result.update({
            "kind": job["kind"],
            "sensitivity_parameter": job.get("parameter"),
            "sensitivity_value": job.get("parameter_value"),
        })
        envelope = {
            "run_metadata": {
                "config_sha256": config_hash,
                "git_commit": git_commit,
                "experiment_fingerprint": experiment_fingerprint,
            },
            "result": result,
        }
        _write_json_atomic(result_path, envelope)
        completed_paths.append(str(result_path))
    return completed_paths


def _build_jobs(config: dict, mode: str, output_root: Path) -> List[dict]:
    study_cfg = config["study"]
    if mode == "pilot":
        pilot_cfg = study_cfg["pilot"]
        experiment_config = deepcopy(config)
        experiment_config["simulation"]["num_rounds"] = pilot_cfg["num_rounds"]
        experiment_config["simulation"]["final_metric_rounds"] = min(
            pilot_cfg["num_rounds"], config["simulation"]["final_metric_rounds"]
        )
        return [
            {
                "kind": "pilot",
                "seed": seed,
                "rho": rho,
                "policy": policy,
                "config": deepcopy(experiment_config),
            }
            for seed in pilot_cfg["seeds"]
            for rho in pilot_cfg["rho_values"]
            for policy in pilot_cfg["policies"]
        ]

    jobs = []
    for seed in study_cfg["seeds"]:
        for rho in study_cfg["primary_base_rho_values"]:
            for policy in study_cfg["primary_base_policies"]:
                jobs.append({
                    "kind": "primary",
                    "seed": seed,
                    "rho": rho,
                    "policy": policy,
                    "config": deepcopy(config),
                })
        for rho in study_cfg["primary_extended_rho_values"]:
            for policy in study_cfg["primary_extended_policies"]:
                jobs.append({
                    "kind": "primary",
                    "seed": seed,
                    "rho": rho,
                    "policy": policy,
                    "config": deepcopy(config),
                })

    for seed in study_cfg["sensitivity_seeds"]:
        for parameter, values in study_cfg["sensitivity"].items():
            for value in values:
                for policy in study_cfg["sensitivity_policies"]:
                    experiment_config = deepcopy(config)
                    if parameter == "profile_share":
                        experiment_config["assumptions"]["profile_share"] = float(value)
                    elif parameter == "driving_dropout_probability":
                        experiment_config["assumptions"]["midround_dropout"]["driving_probability"] = float(value)
                    elif parameter == "clients_per_round":
                        experiment_config["simulation"]["clients_per_round"] = int(value)
                    elif parameter == "policy_f_battery_threshold":
                        experiment_config["assumptions"]["policy_f"]["fallback_battery_pct"] = float(value)
                    else:
                        raise ValueError(f"Unknown final sensitivity parameter: {parameter}")
                    jobs.append({
                        "kind": "sensitivity",
                        "seed": seed,
                        "rho": study_cfg["sensitivity_rho"],
                        "policy": policy,
                        "parameter": parameter,
                        "parameter_value": value,
                        "config": experiment_config,
                    })
    return jobs


def _staleness_gate(config: dict) -> dict:
    assumptions = deepcopy(config["assumptions"])
    assumptions["policy_d"]["staleness_bonus"] = config["study"]["staleness_gate_bonus"]
    client_ids = [f"ev_client_{index:02d}" for index in range(20)]
    fleet = {
        cid: VehicleTelemetry(
            client_id=cid,
            charging_state=ChargingState.LEVEL_2_CHARGING,
            battery_percentage=80.0,
            network_type=NetworkType.UNMETERED_WIFI,
            bandwidth_mbps=30.0,
            usage_profile="heavy-use" if int(cid[-2:]) < 10 else "light-use",
            availability_state=AvailabilityState.CHARGING_WIFI,
        )
        for cid in client_ids
    }
    orchestrator = Orchestrator()
    policy_counts = {}
    for policy in ("C", "D"):
        participation = {cid: 0 for cid in client_ids}
        last_participation = {cid: 0 for cid in client_ids}
        rng = random.Random(90210)
        local_assumptions = assumptions if policy == "D" else config["assumptions"]
        for round_num in range(1, 101):
            selected, eligible_count, _ = _select_clients(
                policy,
                fleet,
                orchestrator,
                5,
                rng,
                local_assumptions,
                participation,
                last_participation,
                round_num,
            )
            if eligible_count != len(client_ids):
                raise AssertionError("Staleness gate fixture must keep all 20 clients eligible")
            for cid in selected:
                participation[cid] += 1
                last_participation[cid] = round_num
        policy_counts[policy] = participation
    c_spread = max(policy_counts["C"].values()) - min(policy_counts["C"].values())
    d_spread = max(policy_counts["D"].values()) - min(policy_counts["D"].values())
    reordered = policy_counts["C"] != policy_counts["D"]
    return {
        "eligible_pool_size": len(client_ids),
        "rounds": 100,
        "clients_per_round": 5,
        "bonus": config["study"]["staleness_gate_bonus"],
        "C_participation": policy_counts["C"],
        "D_participation": policy_counts["D"],
        "C_spread": c_spread,
        "D_spread": d_spread,
        "participation_reordered": reordered,
        "passed": reordered and d_spread < c_spread,
    }


def _validate_final_report(report: dict, expected_runs: int) -> dict:
    required_fields = {
        "profile_map",
        "availability_trace_sha256",
        "eligible_pool_by_profile_by_round",
        "selected_training_example_mass_by_class_by_round",
        "participation_by_client",
        "per_class_accuracy_by_evaluated_round",
    }
    missing = []
    for index, result in enumerate(report["results"]):
        result_missing = sorted(required_fields - set(result))
        if result_missing:
            missing.append({"result_index": index, "missing": result_missing})
    profiles_by_seed = {}
    hashes_by_seed_rho = {}
    for result in report["results"]:
        profiles_by_seed.setdefault((result["kind"], result["seed"], result.get("sensitivity_parameter"), result.get("sensitivity_value")), set()).add(
            _sha256_json(result["profile_map"])
        )
        hashes_by_seed_rho.setdefault((
            result["kind"],
            result["seed"],
            result["rho"],
            result.get("sensitivity_parameter"),
            result.get("sensitivity_value"),
        ), set()).add(result["availability_trace_sha256"])
    invalid_profile_pairs = [key for key, values in profiles_by_seed.items() if len(values) > 1]
    invalid_trace_pairs = [key for key, values in hashes_by_seed_rho.items() if len(values) > 1]
    valid = (
        len(report["results"]) == expected_runs
        and not missing
        and not invalid_profile_pairs
        and not invalid_trace_pairs
    )
    return {
        "passed": valid,
        "expected_runs": expected_runs,
        "observed_runs": len(report["results"]),
        "missing_required_fields": missing,
        "profile_map_mismatches": invalid_profile_pairs,
        "trace_hash_mismatches": invalid_trace_pairs,
        "required_per_run_fields": sorted(required_fields),
    }


def run_final_workflow(
    config: dict,
    config_path: str = "config.yaml",
    mode: str = "full",
    workers: Optional[int] = None,
) -> dict:
    config_bytes = Path(config_path).resolve().read_bytes()
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    git_info = _git_metadata(config_path)
    if mode == "full" and config["study"]["require_committed_config_for_full_run"]:
        if not git_info["commit"] or not git_info["config_is_committed_and_matches_head"]:
            raise RuntimeError(
                "GO/NO-GO: full run blocked because config.yaml is not verifiably committed. "
                f"Git status: {git_info['error'] or 'working config differs from HEAD'}"
            )
        pilot_report_path = Path(config["benchmarks"]["output_dir"]).resolve() / "pilot" / "availability_study_final.json"
        if not pilot_report_path.exists():
            raise RuntimeError("GO/NO-GO: passing pilot report is missing; run --mode pilot first")
        pilot_report = json.loads(pilot_report_path.read_text(encoding="utf-8"))
        pilot_gates_passed = (
            pilot_report.get("config_sha256") == config_hash
            and pilot_report.get("git_commit") == git_info["commit"]
            and pilot_report.get("git_config_matches_head") is True
            and pilot_report.get("run_schema_validation", {}).get("passed") is True
            and pilot_report.get("determinism_check", {}).get("passed") is True
            and pilot_report.get("profile_map_validation", {}).get("same_within_seed_across_policies_and_rho") is True
            and pilot_report.get("profile_map_validation", {}).get("distinct_profile_maps_across_seeds") is True
            and pilot_report.get("staleness_gate", {}).get("passed") is True
        )
        pilot_figure_dir = pilot_report_path.parent / "figures"
        pilot_figures = list(pilot_figure_dir.glob("*.png")) if pilot_figure_dir.exists() else []
        if not pilot_gates_passed or len(pilot_figures) < 5:
            raise RuntimeError(
                "GO/NO-GO: pilot report/gates/figures do not match the committed config and commit; rerun --mode pilot"
            )

    output_dir = Path(config["benchmarks"]["output_dir"]).resolve()
    if mode == "pilot":
        output_dir = output_dir / "pilot"
    run_root = Path(config["study"]["resumable_run_dir"]).resolve()
    if mode == "pilot":
        run_root = output_dir / "runs"
    jobs = _build_jobs(config, mode, output_dir)
    by_seed = {}
    for job in jobs:
        by_seed.setdefault(job["seed"], []).append(job)

    worker_count = 1 if mode == "pilot" else (workers or config["study"]["workers"])
    worker_arguments = [
        (config, seed_jobs, config_hash, git_info["commit"], str(run_root))
        for _, seed_jobs in sorted(by_seed.items())
    ]
    written_paths = []
    if worker_count == 1:
        for arguments in worker_arguments:
            written_paths.extend(_run_seed_bundle(arguments))
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_run_seed_bundle, arguments) for arguments in worker_arguments]
            for future in as_completed(futures):
                written_paths.extend(future.result())
                print(f"Completed seed bundle ({len(written_paths)}/{len(jobs)} run files)", flush=True)

    runs = []
    run_metadata = []
    for path_text in written_paths:
        envelope = json.loads(Path(path_text).read_text(encoding="utf-8"))
        runs.append(envelope["result"])
        run_metadata.append(envelope["run_metadata"])

    primary_runs = [result for result in runs if result["kind"] in ("primary", "pilot")]
    grouped = {}
    for result in primary_runs:
        key = f"rho={result['rho']:g}/policy={result['policy']}"
        grouped.setdefault(key, []).append(result)
    trace_groups = {}
    for result in primary_runs:
        trace_groups.setdefault((result["seed"], result["rho"]), set()).add(result["availability_trace_sha256"])
    if any(len(values) != 1 for values in trace_groups.values()):
        raise RuntimeError("Paired runs have mismatched availability trace hashes")

    summary = {key: _summarize(group) for key, group in grouped.items()}
    paired = _paired_differences(primary_runs, sorted({result["policy"] for result in primary_runs}))
    sensitivity_runs = [result for result in runs if result["kind"] == "sensitivity"]
    sensitivity_grouped = {}
    for result in sensitivity_runs:
        key = f"{result['sensitivity_parameter']}={result['sensitivity_value']}/policy={result['policy']}"
        sensitivity_grouped.setdefault(key, []).append(result)
    report = {
        "schema_version": 2,
        "run_mode": mode,
        "run_date": __import__("datetime").date.today().isoformat(),
        "config_sha256": config_hash,
        "git_commit": git_info["commit"],
        "git_config_matches_head": git_info["config_is_committed_and_matches_head"],
        "git_metadata_error": git_info["error"],
        "config": config,
        "primary_hypotheses": config["study"]["primary_hypotheses"],
        "exploratory_comparisons": config["study"]["exploratory_comparisons"],
        "summary_mean_std_ci95": summary,
        "paired_differences_vs_A_t_and_bootstrap_ci95": paired,
        "holm_corrected_primary_hypotheses": _registered_hypotheses(primary_runs) if mode == "full" else None,
        "sensitivity_summary_mean_std_ci95": {
            key: _summarize(group) for key, group in sensitivity_grouped.items()
        },
        "staleness_gate": _staleness_gate(config),
        "run_schema_validation": None,
        "run_metadata": run_metadata,
        "results": runs,
        "per_run_files": written_paths,
    }
    report["run_schema_validation"] = _validate_final_report(report, len(jobs))
    primary_profile_maps = {
        (result["seed"], _sha256_json(result["profile_map"]))
        for result in primary_runs
        if result["kind"] in ("primary", "pilot")
    }
    profiles_per_seed = {}
    for seed, profile_hash in primary_profile_maps:
        profiles_per_seed.setdefault(seed, set()).add(profile_hash)
    report["profile_map_validation"] = {
        "same_within_seed_across_policies_and_rho": all(len(hashes) == 1 for hashes in profiles_per_seed.values()),
        "distinct_profile_maps_across_seeds": len({next(iter(hashes)) for hashes in profiles_per_seed.values() if hashes}) > 1,
        "unique_maps": len({next(iter(hashes)) for hashes in profiles_per_seed.values() if hashes}),
        "seed_count": len(profiles_per_seed),
    }
    if mode == "pilot":
        report["determinism_check"] = _pilot_determinism_check(config, config_path, jobs[0], run_root)
    results_path = output_dir / "availability_study_final.json"
    _write_json_atomic(results_path, report)
    if not report["staleness_gate"]["passed"]:
        raise RuntimeError("GO/NO-GO: large-bonus staleness sanity gate failed")
    if not report["run_schema_validation"]["passed"]:
        raise RuntimeError("GO/NO-GO: output schema/profile pairing/trace validation failed")
    if not report["profile_map_validation"]["same_within_seed_across_policies_and_rho"]:
        raise RuntimeError("GO/NO-GO: profile map changed within a seed")
    if len(profiles_per_seed) > 1 and not report["profile_map_validation"]["distinct_profile_maps_across_seeds"]:
        raise RuntimeError("GO/NO-GO: vehicle profiles were not randomized across seeds")
    if mode == "pilot" and not report["determinism_check"]["passed"]:
        raise RuntimeError("GO/NO-GO: deterministic rerun check failed")
    print(f"Saved {mode} report: {results_path}", flush=True)
    return report


def _pilot_determinism_check(config: dict, config_path: str, job: dict, run_root: Path) -> dict:
    assumptions = job["config"]["assumptions"]
    sim_cfg = job["config"]["simulation"]
    seed = job["seed"]
    rho = job["rho"]
    fleet = FleetTelemetrySimulator(
        num_clients=sim_cfg["fleet_size"],
        seed=seed,
        profile_share=assumptions["profile_share"],
        initial_state_probabilities=assumptions["initial_state_probabilities"],
        transition_matrices=assumptions["availability_transition_matrices"],
        dropout_assumptions=assumptions["midround_dropout"],
    )
    loaders, test_loader = partition_dataset(
        num_clients=sim_cfg["fleet_size"],
        batch_size=sim_cfg["batch_size"],
        data_dir=sim_cfg["data_dir"],
        seed=seed,
        usage_profiles=[fleet.usage_profiles[f"ev_client_{index:02d}"] for index in range(sim_cfg["fleet_size"])],
        rho=rho,
        max_client_examples=sim_cfg["max_client_examples"],
        test_sample_size=sim_cfg["test_sample_size"],
        dirichlet_alpha=assumptions["non_iid_dirichlet_alpha"],
        favored_class_weight=assumptions["profile_class_allocation_weight"]["favored"],
        non_favored_class_weight=assumptions["profile_class_allocation_weight"]["non_favored"],
    )
    first = run_one(job["config"], job["policy"], rho, seed, loaders, test_loader)
    second = run_one(job["config"], job["policy"], rho, seed, loaders, test_loader)
    return {
        "passed": _sha256_json(first) == _sha256_json(second),
        "seed": seed,
        "policy": job["policy"],
        "result_sha256_first": _sha256_json(first),
        "result_sha256_second": _sha256_json(second),
    }