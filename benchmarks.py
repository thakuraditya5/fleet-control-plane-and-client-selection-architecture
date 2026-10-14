"""
Benchmarking Engine for EV Federated Learning Fleet.
Executes Baseline (Random) vs Constraint-Aware Client Selection experiments.
Tracks convergence, battery power preservation, metered data offloading, and mid-round dropouts.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dataclasses import asdict, dataclass
import json
import logging
from typing import Any, Dict, List, Tuple
import yaml
import torch
from flwr.common import ndarrays_to_parameters
import flwr.server

from src.client import EVFlowerClient
from src.model import EdgeCNN, get_parameters, partition_dataset
from src.orchestrator import Orchestrator
from src.server import EVClientManager, EVFedAvg, LocalEVClientProxy
from src.telemetry import FleetTelemetrySimulator, NetworkType

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BenchmarkEngine")


def run_fl_experiment(
    strategy_type: str,
    config: dict,
    client_loaders: list,
    test_loader: Any,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """
    Executes an end-to-end FL training run for a given strategy.
    """
    sim_cfg = config.get("simulation", {})
    tel_cfg = config.get("telemetry", {})
    guard_cfg = config.get("guardrails", {})

    num_clients = sim_cfg.get("fleet_size", 20)
    num_rounds = sim_cfg.get("num_rounds", 15)
    clients_per_round = sim_cfg.get("clients_per_round", 5)
    seed = sim_cfg.get("seed", 42)
    local_epochs = sim_cfg.get("local_epochs", 1)
    learning_rate = sim_cfg.get("learning_rate", 0.01)

    # Initialize Telemetry Simulator with identical seed for reproducible vehicle traces
    telemetry_simulator = FleetTelemetrySimulator(
        num_clients=num_clients,
        seed=seed,
        battery_capacity_kwh=tel_cfg.get("battery_capacity_kwh", 75.0),
        training_energy_per_round_kwh=tel_cfg.get("training_energy_per_round_kwh", 0.025),
        transmission_power_w=tel_cfg.get("transmission_power_w", 5.0),
    )

    # Initialize Orchestrator with rule constraints
    power_rules = guard_cfg.get("power", {})
    net_rules = guard_cfg.get("network", {})

    required_net = NetworkType.UNMETERED_WIFI
    if net_rules.get("required_network_type") != "UNMETERED_WIFI":
        required_net = NetworkType(net_rules.get("required_network_type", "UNMETERED_WIFI"))

    orchestrator = Orchestrator(
        min_battery_general=power_rules.get("min_battery_general", 50.0),
        min_battery_dc_fast=power_rules.get("min_battery_dc_fast", 20.0),
        required_network=required_net,
        min_bandwidth_mbps=net_rules.get("min_bandwidth_mbps", 10.0),
    )

    # Instantiate local clients
    is_baseline = strategy_type == "baseline_random"
    clients = {}
    client_manager = EVClientManager()

    for i in range(num_clients):
        cid = f"ev_client_{i:02d}"
        client = EVFlowerClient(
            client_id=cid,
            train_loader=client_loaders[i],
            val_loader=test_loader,
            telemetry_simulator=telemetry_simulator,
            is_baseline_run=is_baseline,
            device=device,
        )
        clients[cid] = client
        client_proxy = LocalEVClientProxy(cid=cid, client=client)
        client_manager.register(client_proxy)

    # Initial global model
    initial_model = EdgeCNN().to(device)
    initial_parameters = ndarrays_to_parameters(get_parameters(initial_model))

    # Initialize custom Flower Strategy
    strategy = EVFedAvg(
        orchestrator=orchestrator,
        telemetry_simulator=telemetry_simulator,
        strategy_type=strategy_type,
        target_clients_per_round=clients_per_round,
        test_loader=test_loader,
        device=device,
        epochs_per_round=local_epochs,
        learning_rate=learning_rate,
        initial_parameters=initial_parameters,
    )

    # Flower Server
    server = flwr.server.Server(client_manager=client_manager, strategy=strategy)

    logger.info(f"--- Starting FL Experiment: {strategy_type.upper()} ({num_rounds} rounds, {num_clients} clients) ---")
    
    # Run FL rounds
    server.fit(num_rounds=num_rounds, timeout=None)

    # Gather cumulative metrics
    summary_metrics = telemetry_simulator.get_summary_metrics()

    return {
        "strategy": strategy_type,
        "round_history": strategy.round_history,
        "summary": summary_metrics,
        "num_rounds": num_rounds,
        "fleet_size": num_clients,
    }


def compare_benchmarks(baseline_res: dict, constraint_res: dict) -> dict:
    """
    Computes comparative metrics between baseline and constraint-aware runs:
    - Model Convergence Rate (Accuracy per round)
    - Resource Preservation (Energy saved kWh, Metered cellular MB saved)
    - Reliability / Dropout reduction
    """
    b_summary = baseline_res["summary"]
    c_summary = constraint_res["summary"]

    b_history = baseline_res["round_history"]
    c_history = constraint_res["round_history"]

    b_final_acc = b_history[-1]["test_accuracy"] if b_history else 0.0
    c_final_acc = c_history[-1]["test_accuracy"] if c_history else 0.0

    b_final_loss = b_history[-1]["test_loss"] if b_history else 0.0
    c_final_loss = c_history[-1]["test_loss"] if c_history else 0.0

    # Resource preservation
    energy_saved_kwh = max(0.0, b_summary["total_energy_kwh"] - c_summary["total_energy_kwh"])
    metered_saved_mb = max(0.0, b_summary["total_metered_mb"] - c_summary["total_metered_mb"])
    metered_reduction_pct = (
        (metered_saved_mb / max(b_summary["total_metered_mb"], 1e-5)) * 100.0
        if b_summary["total_metered_mb"] > 0
        else 100.0
    )

    # Dropout reduction
    b_dropouts = b_summary["total_dropouts"]
    c_dropouts = c_summary["total_dropouts"]
    dropout_reduction = b_dropouts - c_dropouts

    # Total selections attempted vs completed
    b_total_selected = sum(r["selected_count"] for r in b_history)
    b_total_success = sum(r["successful_count"] for r in b_history)
    b_completion_rate = (b_total_success / max(b_total_selected, 1)) * 100.0

    c_total_selected = sum(r["selected_count"] for r in c_history)
    c_total_success = sum(r["successful_count"] for r in c_history)
    c_completion_rate = (c_total_success / max(c_total_selected, 1)) * 100.0

    comparison = {
        "baseline": {
            "final_accuracy": b_final_acc,
            "final_loss": b_final_loss,
            "total_energy_kwh": b_summary["total_energy_kwh"],
            "total_metered_mb": b_summary["total_metered_mb"],
            "total_unmetered_mb": b_summary["total_unmetered_mb"],
            "total_dropouts": b_dropouts,
            "completion_rate_pct": round(b_completion_rate, 2),
        },
        "constraint_aware": {
            "final_accuracy": c_final_acc,
            "final_loss": c_final_loss,
            "total_energy_kwh": c_summary["total_energy_kwh"],
            "total_metered_mb": c_summary["total_metered_mb"],
            "total_unmetered_mb": c_summary["total_unmetered_mb"],
            "total_dropouts": c_dropouts,
            "completion_rate_pct": round(c_completion_rate, 2),
        },
        "improvements": {
            "accuracy_delta": round(c_final_acc - b_final_acc, 2),
            "energy_saved_kwh": round(energy_saved_kwh, 4),
            "metered_data_saved_mb": round(metered_saved_mb, 2),
            "metered_reduction_pct": round(metered_reduction_pct, 2),
            "dropouts_prevented": int(dropout_reduction),
            "completion_rate_improvement_pct": round(c_completion_rate - b_completion_rate, 2),
        },
    }
    return comparison
