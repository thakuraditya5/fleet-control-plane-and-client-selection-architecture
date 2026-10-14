"""
Flower Client implementation with dynamic EV Telemetry Hooks.
Simulates edge model updates, local training, telemetry logging, and mid-round dropouts.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
import flwr as flwr
from flwr.client import NumPyClient

from src.model import EdgeCNN, get_parameters, set_parameters, train, test, get_model_size_bytes
from src.telemetry import FleetTelemetrySimulator, VehicleTelemetry

logger = logging.getLogger("EVClient")


class EVFlowerClient(NumPyClient):
    """
    Simulated On-Device EV Flower Client.
    Integrates telemetry monitoring, dynamic dropouts, and resource accounting.
    """

    def __init__(
        self,
        client_id: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        telemetry_simulator: FleetTelemetrySimulator,
        is_baseline_run: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        self.client_id = client_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.telemetry_simulator = telemetry_simulator
        self.is_baseline_run = is_baseline_run
        self.device = device
        self.model = EdgeCNN().to(device)

    def get_telemetry(self) -> VehicleTelemetry:
        """Retrieves real-time telemetry profile for this vehicle."""
        return self.telemetry_simulator.clients[self.client_id]

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        return get_parameters(self.model)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Executes on-device local training round.
        Simulates telemetry resource usage and mid-round state dropouts.
        """
        # 1. Update local weights
        set_parameters(self.model, parameters)

        epochs = int(config.get("epochs", 1))
        lr = float(config.get("lr", 0.01))
        model_size_bytes = get_model_size_bytes(self.model)

        # 2. Check for mid-round dynamic state changes & dropouts
        success, dropout_reason = self.telemetry_simulator.simulate_mid_round_dynamics(
            self.client_id, is_baseline=self.is_baseline_run
        )

        if not success:
            # Client dropped out mid-round (e.g. driving disconnection, battery limit)
            self.telemetry_simulator.record_resource_usage(
                client_id=self.client_id,
                num_epochs=epochs,
                model_size_bytes=model_size_bytes,
                completed=False,
            )
            telem = self.get_telemetry()
            logger.warning(
                f"Client {self.client_id} DROPPED OUT mid-round: {dropout_reason} "
                f"[{telem.charging_state.value}, {telem.battery_percentage:.1f}%, {telem.network_type.value}]"
            )
            # Return empty update to signal failure to strategy
            return (
                parameters,  # keep existing or return 0 samples
                0,
                {
                    "status": "dropped_out",
                    "reason": dropout_reason,
                    "client_id": self.client_id,
                    "metered_mb": telem.metered_bytes_transferred / (1024 * 1024),
                    "unmetered_mb": telem.unmetered_bytes_transferred / (1024 * 1024),
                },
            )

        # 3. Train model locally on device
        loss, accuracy = train(
            net=self.model,
            trainloader=self.train_loader,
            epochs=epochs,
            learning_rate=lr,
            device=self.device,
        )

        # 4. Record successful resource usage (compute + transmission)
        self.telemetry_simulator.record_resource_usage(
            client_id=self.client_id,
            num_epochs=epochs,
            model_size_bytes=model_size_bytes,
            completed=True,
        )

        telem = self.get_telemetry()
        metrics = {
            "status": "success",
            "loss": float(loss),
            "accuracy": float(accuracy),
            "client_id": self.client_id,
            "battery_percentage": float(telem.battery_percentage),
            "charging_state": telem.charging_state.value,
            "network_type": telem.network_type.value,
            "bandwidth_mbps": float(telem.bandwidth_mbps),
            "compute_energy_kwh": float(telem.compute_energy_kwh),
            "transfer_energy_kwh": float(telem.transfer_energy_kwh),
        }

        updated_parameters = get_parameters(self.model)
        num_samples = len(self.train_loader.dataset)
        return updated_parameters, num_samples, metrics

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        set_parameters(self.model, parameters)
        loss, accuracy = test(self.model, self.val_loader, device=self.device)
        return float(loss), len(self.val_loader.dataset), {"accuracy": float(accuracy)}
