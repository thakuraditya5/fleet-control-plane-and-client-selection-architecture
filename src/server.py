"""
Custom Flower Server, Strategy, and Client Manager for EV Fleet FL.
Integrates Orchestrator guardrails, dynamic telemetry tracking, and resource logging.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader
import flwr as flwr
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    NDArrays,
    Parameters,
    ReconnectIns,
    DisconnectRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate

from src.client import EVFlowerClient
from src.model import EdgeCNN, get_parameters, set_parameters, test
from src.orchestrator import Orchestrator
from src.telemetry import FleetTelemetrySimulator, VehicleTelemetry

logger = logging.getLogger("EVServer")


class LocalEVClientProxy(ClientProxy):
    """
    Flower ClientProxy wrapper for local on-device EV clients.
    Facilitates lightweight in-process simulation without heavyweight RPC/Ray overhead.
    """

    def __init__(self, cid: str, client: EVFlowerClient):
        super().__init__(cid)
        self.client = client

    def get_properties(self, ins: GetPropertiesIns, timeout: Optional[float], group_id: Optional[int]) -> GetPropertiesRes:
        return GetPropertiesRes(status=Status(code=Code.OK, message="Success"), properties={})

    def get_parameters(self, ins: GetParametersIns, timeout: Optional[float], group_id: Optional[int]) -> GetParametersRes:
        params = self.client.get_parameters(ins.config)
        return GetParametersRes(status=Status(code=Code.OK, message="Success"), parameters=ndarrays_to_parameters(params))

    def fit(self, ins: FitIns, timeout: Optional[float], group_id: Optional[int]) -> FitRes:
        ndarrays = parameters_to_ndarrays(ins.parameters)
        updated_params, num_examples, metrics = self.client.fit(ndarrays, ins.config)
        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=ndarrays_to_parameters(updated_params),
            num_examples=num_examples,
            metrics=metrics,
        )

    def evaluate(self, ins: EvaluateIns, timeout: Optional[float], group_id: Optional[int]) -> EvaluateRes:
        ndarrays = parameters_to_ndarrays(ins.parameters)
        loss, num_examples, metrics = self.client.evaluate(ndarrays, ins.config)
        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=loss,
            num_examples=num_examples,
            metrics=metrics,
        )

    def reconnect(self, ins: ReconnectIns, timeout: Optional[float], group_id: Optional[int]) -> DisconnectRes:
        return DisconnectRes(reason="Reconnected")


class EVClientManager(SimpleClientManager):
    """
    Custom Client Manager for the EV Fleet Control Plane.
    Maintains client registry and interfaces with Orchestrator guardrails.
    """

    def __init__(self):
        super().__init__()

    def register(self, client: ClientProxy) -> bool:
        return super().register(client)

    def get_all_clients(self) -> Dict[str, ClientProxy]:
        return self.clients


class EVFedAvg(FedAvg):
    """
    Custom Flower Strategy implementing:
    1. Baseline Random Selection vs. Constraint-Aware Guardrail Selection.
    2. Telemetry tracking per round (Energy, Metered vs. Unmetered Data).
    3. Dropout logging and convergence tracking.
    """

    def __init__(
        self,
        orchestrator: Orchestrator,
        telemetry_simulator: FleetTelemetrySimulator,
        strategy_type: str = "constraint_aware",
        target_clients_per_round: int = 5,
        test_loader: Optional[DataLoader] = None,
        device: torch.device = torch.device("cpu"),
        epochs_per_round: int = 1,
        learning_rate: float = 0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.orchestrator = orchestrator
        self.telemetry_simulator = telemetry_simulator
        self.strategy_type = strategy_type
        self.target_clients_per_round = target_clients_per_round
        self.test_loader = test_loader
        self.device = device
        self.epochs_per_round = epochs_per_round
        self.learning_rate = learning_rate

        # Evaluation model
        self.eval_model = EdgeCNN().to(device)

        # Metrics history
        self.round_history: List[Dict] = []

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: EVClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Configures the next round of training.
        Applies either Baseline Random Selection or Constraint-Aware Orchestration.
        """
        # 1. Advance telemetry state
        self.telemetry_simulator.advance_round(server_round)
        fleet_telemetry = self.telemetry_simulator.get_fleet_telemetry()

        # 2. Query Orchestrator
        selected_cids, evaluations = self.orchestrator.select_clients(
            fleet_telemetry=fleet_telemetry,
            target_count=self.target_clients_per_round,
            strategy_type=self.strategy_type,
            rng=self.telemetry_simulator.rng,
        )

        eligible_count = sum(1 for e in evaluations.values() if e.is_eligible)
        rejected_count = len(fleet_telemetry) - eligible_count

        logger.info(
            f"[Round {server_round}] Strategy: {self.strategy_type.upper()} | "
            f"Eligible: {eligible_count}/{len(fleet_telemetry)} | "
            f"Selected: {len(selected_cids)} clients: {selected_cids}"
        )

        # 3. Build FitIns
        all_clients = client_manager.get_all_clients()
        fit_configurations = []

        config = {
            "round": server_round,
            "epochs": self.epochs_per_round,
            "lr": self.learning_rate,
        }

        for cid in selected_cids:
            if cid in all_clients:
                fit_ins = FitIns(parameters=parameters, config=config)
                fit_configurations.append((all_clients[cid], fit_ins))

        # Store round metadata for later logging
        self._current_round_meta = {
            "round": server_round,
            "eligible_count": eligible_count,
            "rejected_count": rejected_count,
            "selected_clients": selected_cids,
        }

        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Union[bool, bytes, float, int, str]]]:
        """
        Aggregates model updates from clients.
        Filters out dropouts and logs resource usage.
        """
        if not results:
            return None, {}

        successful_results = []
        dropouts_this_round = 0
        successful_cids = []
        dropped_cids = []

        for client_proxy, fit_res in results:
            if fit_res.num_examples > 0 and fit_res.metrics.get("status") == "success":
                # Convert back to NDArrays for FedAvg aggregate
                ndarrays = parameters_to_ndarrays(fit_res.parameters)
                successful_results.append((ndarrays, fit_res.num_examples))
                successful_cids.append(client_proxy.cid)
            else:
                dropouts_this_round += 1
                dropped_cids.append(client_proxy.cid)

        # Aggregate parameters using FedAvg if any successful clients
        aggregated_parameters: Optional[Parameters] = None
        if successful_results:
            aggregated_ndarrays = aggregate(successful_results)
            aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)
            # Update eval model weights
            set_parameters(self.eval_model, aggregated_ndarrays)

        # Fetch telemetry summary
        telem_summary = self.telemetry_simulator.get_summary_metrics()

        # Centralized evaluation on global test set
        test_loss, test_acc = 0.0, 0.0
        if self.test_loader is not None and aggregated_parameters is not None:
            test_loss, test_acc = test(self.eval_model, self.test_loader, device=self.device)

        round_data = {
            "round": server_round,
            "strategy": self.strategy_type,
            "test_accuracy": round(float(test_acc) * 100.0, 2),
            "test_loss": round(float(test_loss), 4),
            "selected_count": len(self._current_round_meta["selected_clients"]),
            "successful_count": len(successful_cids),
            "dropouts_this_round": dropouts_this_round,
            "total_dropouts": telem_summary["total_dropouts"],
            "eligible_in_fleet": self._current_round_meta["eligible_count"],
            "total_energy_kwh": telem_summary["total_energy_kwh"],
            "metered_mb": telem_summary["total_metered_mb"],
            "unmetered_mb": telem_summary["total_unmetered_mb"],
        }
        self.round_history.append(round_data)

        logger.info(
            f"[Round {server_round} Results] Acc: {test_acc*100.0:.2f}% | Loss: {test_loss:.4f} | "
            f"Success: {len(successful_cids)} | Dropouts: {dropouts_this_round} | "
            f"Metered: {telem_summary['total_metered_mb']}MB | Energy: {telem_summary['total_energy_kwh']}kWh"
        )

        metrics = {
            "accuracy": float(test_acc),
            "loss": float(test_loss),
            "dropouts": dropouts_this_round,
        }
        return aggregated_parameters, metrics

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Union[bool, bytes, float, int, str]]]]:
        """Flower strategy global evaluation hook."""
        if self.test_loader is None:
            return None
        ndarrays = parameters_to_ndarrays(parameters)
        set_parameters(self.eval_model, ndarrays)
        loss, accuracy = test(self.eval_model, self.test_loader, device=self.device)
        return float(loss), {"accuracy": float(accuracy)}
