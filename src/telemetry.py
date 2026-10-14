"""
Vehicle Telemetry Module for EV Federated Learning Fleet.
Simulates real-world dynamic battery, charging state, compute, and connectivity telemetry.
"""

from dataclasses import dataclass, field
from enum import Enum
import random
from typing import Dict, List, Optional, Tuple


class ChargingState(str, Enum):
    DISCHARGING = "DISCHARGING"
    LEVEL_2_CHARGING = "LEVEL_2_CHARGING"
    DC_FAST_CHARGING = "DC_FAST_CHARGING"


class NetworkType(str, Enum):
    CELLULAR_5G = "CELLULAR_5G"
    METERED_4G = "METERED_4G"
    UNMETERED_WIFI = "UNMETERED_WIFI"


@dataclass
class VehicleTelemetry:
    client_id: str
    charging_state: ChargingState
    battery_percentage: float  # 0.0 to 100.0%
    network_type: NetworkType
    bandwidth_mbps: float
    battery_capacity_kwh: float = 75.0  # standard EV pack capacity
    
    # State tracking per round
    compute_energy_kwh: float = 0.0
    transfer_energy_kwh: float = 0.0
    metered_bytes_transferred: int = 0
    unmetered_bytes_transferred: int = 0
    dropped_out: bool = False
    dropout_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "client_id": self.client_id,
            "charging_state": self.charging_state.value,
            "battery_percentage": round(self.battery_percentage, 2),
            "network_type": self.network_type.value,
            "bandwidth_mbps": round(self.bandwidth_mbps, 2),
            "battery_capacity_kwh": self.battery_capacity_kwh,
            "compute_energy_kwh": round(self.compute_energy_kwh, 6),
            "transfer_energy_kwh": round(self.transfer_energy_kwh, 6),
            "metered_mb": round(self.metered_bytes_transferred / (1024 * 1024), 3),
            "unmetered_mb": round(self.unmetered_bytes_transferred / (1024 * 1024), 3),
            "dropped_out": self.dropped_out,
            "dropout_reason": self.dropout_reason,
        }


class FleetTelemetrySimulator:
    """
    Dynamically generates and updates reproducible trace profiles for an EV fleet.
    Uses Markov-based transitions to simulate realistic driving, charging, and connectivity shifts.
    """

    def __init__(
        self,
        num_clients: int = 20,
        seed: int = 42,
        battery_capacity_kwh: float = 75.0,
        training_energy_per_round_kwh: float = 0.025,
        transmission_power_w: float = 5.0,
    ):
        self.num_clients = num_clients
        self.seed = seed
        self.rng = random.Random(seed)
        self.battery_capacity_kwh = battery_capacity_kwh
        self.training_energy_per_round_kwh = training_energy_per_round_kwh
        self.transmission_power_w = transmission_power_w
        
        # Cumulative fleet metrics
        self.total_compute_energy_kwh = 0.0
        self.total_transfer_energy_kwh = 0.0
        self.total_metered_bytes = 0
        self.total_unmetered_bytes = 0
        self.total_dropouts = 0

        # Initial client states
        self.clients: Dict[str, VehicleTelemetry] = {}
        self._initialize_fleet()

    def reset(self, seed: Optional[int] = None):
        """Resets the fleet telemetry simulator to initial state with specified seed."""
        if seed is not None:
            self.seed = seed
        self.rng = random.Random(self.seed)
        self.total_compute_energy_kwh = 0.0
        self.total_transfer_energy_kwh = 0.0
        self.total_metered_bytes = 0
        self.total_unmetered_bytes = 0
        self.total_dropouts = 0
        self.clients = {}
        self._initialize_fleet()

    def _initialize_fleet(self):
        """Populates initial fleet states with realistic diversity."""
        for i in range(self.num_clients):
            cid = f"ev_client_{i:02d}"
            
            # Diverse initial distribution across fleet archetypes:
            # ~40% home/depot charging, ~20% DC fast charging, ~40% driving
            rand_val = self.rng.random()
            if rand_val < 0.40:
                # Depot / Home charging
                state = ChargingState.LEVEL_2_CHARGING
                battery = self.rng.uniform(45.0, 95.0)
                net = NetworkType.UNMETERED_WIFI
                bw = self.rng.uniform(15.0, 75.0)
            elif rand_val < 0.60:
                # Fast charging hub
                state = ChargingState.DC_FAST_CHARGING
                battery = self.rng.uniform(15.0, 80.0)
                net = self.rng.choice([NetworkType.UNMETERED_WIFI, NetworkType.CELLULAR_5G])
                bw = self.rng.uniform(25.0, 120.0)
            else:
                # Discharging (in-transit or parked off-grid)
                state = ChargingState.DISCHARGING
                battery = self.rng.uniform(20.0, 85.0)
                net = self.rng.choice([NetworkType.METERED_4G, NetworkType.CELLULAR_5G, NetworkType.UNMETERED_WIFI])
                bw = self.rng.uniform(2.0, 25.0)

            self.clients[cid] = VehicleTelemetry(
                client_id=cid,
                charging_state=state,
                battery_percentage=battery,
                network_type=net,
                bandwidth_mbps=bw,
                battery_capacity_kwh=self.battery_capacity_kwh,
            )

    def advance_round(self, round_num: int):
        """
        Advances the fleet state to the next federated round.
        Simulates natural vehicle state transitions over time.
        """
        for cid, t in self.clients.items():
            # Reset round-specific flags
            t.dropped_out = False
            t.dropout_reason = None

            # Markov State Transitions:
            if t.charging_state == ChargingState.DISCHARGING:
                # While driving: battery naturally drops slightly
                t.battery_percentage = max(5.0, t.battery_percentage - self.rng.uniform(1.0, 3.5))
                # 25% chance vehicle parks and plugs into charger
                if self.rng.random() < 0.25:
                    if self.rng.random() < 0.35:
                        t.charging_state = ChargingState.DC_FAST_CHARGING
                        t.network_type = self.rng.choice([NetworkType.UNMETERED_WIFI, NetworkType.CELLULAR_5G])
                        t.bandwidth_mbps = self.rng.uniform(20.0, 100.0)
                    else:
                        t.charging_state = ChargingState.LEVEL_2_CHARGING
                        t.network_type = NetworkType.UNMETERED_WIFI
                        t.bandwidth_mbps = self.rng.uniform(12.0, 60.0)
                else:
                    # Still driving: network changes with mobility
                    t.network_type = self.rng.choices(
                        [NetworkType.METERED_4G, NetworkType.CELLULAR_5G, NetworkType.UNMETERED_WIFI],
                        weights=[0.55, 0.35, 0.10],
                    )[0]
                    t.bandwidth_mbps = self.rng.uniform(2.0, 30.0)

            elif t.charging_state == ChargingState.LEVEL_2_CHARGING:
                # Level 2 charging increases battery
                t.battery_percentage = min(100.0, t.battery_percentage + self.rng.uniform(3.0, 8.0))
                # 20% chance unplugged and driven
                if self.rng.random() < 0.20 or t.battery_percentage >= 98.0:
                    t.charging_state = ChargingState.DISCHARGING
                    t.network_type = self.rng.choice([NetworkType.METERED_4G, NetworkType.CELLULAR_5G])
                    t.bandwidth_mbps = self.rng.uniform(5.0, 25.0)
                else:
                    # Stays on WiFi at depot/home with minor jitter
                    t.network_type = NetworkType.UNMETERED_WIFI
                    t.bandwidth_mbps = max(5.0, t.bandwidth_mbps + self.rng.uniform(-3.0, 3.0))

            elif t.charging_state == ChargingState.DC_FAST_CHARGING:
                # DC Fast charging rapidly increases battery
                t.battery_percentage = min(100.0, t.battery_percentage + self.rng.uniform(10.0, 20.0))
                # Fast charge sessions end quickly (~40% chance per round or when high)
                if self.rng.random() < 0.40 or t.battery_percentage >= 90.0:
                    t.charging_state = ChargingState.DISCHARGING
                    t.network_type = self.rng.choice([NetworkType.METERED_4G, NetworkType.CELLULAR_5G])
                    t.bandwidth_mbps = self.rng.uniform(5.0, 35.0)
                else:
                    t.bandwidth_mbps = max(10.0, t.bandwidth_mbps + self.rng.uniform(-5.0, 5.0))

    def simulate_mid_round_dynamics(self, client_id: str, is_baseline: bool) -> Tuple[bool, Optional[str]]:
        """
        Simulates real-world mid-round execution dynamics.
        If a client was assigned while non-eligible (e.g. DISCHARGING or METERED_4G in baseline),
        it has a high probability of mid-round dropout (packet loss, vehicle start-up, low-battery cutoff).
        """
        t = self.clients[client_id]

        # Inherent stability check:
        if t.charging_state == ChargingState.DISCHARGING:
            # Driving vehicles frequently drop connection or terminate background ML jobs for ECU safety
            if self.rng.random() < 0.50:
                t.dropped_out = True
                t.dropout_reason = "ECU Preempted: Vehicle active driving mode & safety override"
                self.total_dropouts += 1
                return False, t.dropout_reason

        if t.battery_percentage < 25.0:
            # Critical low battery drop
            if self.rng.random() < 0.60:
                t.dropped_out = True
                t.dropout_reason = f"Low battery shutdown ({t.battery_percentage:.1f}% < 25%)"
                self.total_dropouts += 1
                return False, t.dropout_reason

        if t.network_type == NetworkType.METERED_4G or t.bandwidth_mbps < 8.0:
            # Cellular congestion / handover timeout
            if self.rng.random() < 0.45:
                t.dropped_out = True
                t.dropout_reason = f"Network timeout on {t.network_type.value} ({t.bandwidth_mbps:.1f} Mbps)"
                self.total_dropouts += 1
                return False, t.dropout_reason

        # Otherwise client succeeds
        return True, None

    def record_resource_usage(
        self,
        client_id: str,
        num_epochs: int,
        model_size_bytes: int,
        completed: bool,
    ):
        """
        Records the compute energy and wireless network data consumed by a client.
        Note: Even if a client drops out mid-round, it still wasted initial download and partial compute!
        """
        t = self.clients[client_id]
        
        # Compute energy
        compute_kwh = self.training_energy_per_round_kwh * num_epochs if completed else (self.training_energy_per_round_kwh * 0.4)
        t.compute_energy_kwh += compute_kwh
        self.total_compute_energy_kwh += compute_kwh

        # Battery percentage impact:
        battery_pct_drain = (compute_kwh / t.battery_capacity_kwh) * 100.0
        t.battery_percentage = max(0.0, t.battery_percentage - battery_pct_drain)

        # Network data accounting:
        # Downlink (model receive) + Uplink (model send if completed)
        total_transfer_bytes = model_size_bytes * 2 if completed else model_size_bytes
        
        # Transmission time (seconds)
        bw_bytes_sec = (t.bandwidth_mbps * 1e6) / 8.0
        transfer_time_sec = total_transfer_bytes / max(bw_bytes_sec, 1e5)
        transfer_kwh = (self.transmission_power_w * transfer_time_sec) / (3600.0 * 1000.0)
        
        t.transfer_energy_kwh += transfer_kwh
        self.total_transfer_energy_kwh += transfer_kwh

        if t.network_type == NetworkType.UNMETERED_WIFI:
            t.unmetered_bytes_transferred += total_transfer_bytes
            self.total_unmetered_bytes += total_transfer_bytes
        else:
            t.metered_bytes_transferred += total_transfer_bytes
            self.total_metered_bytes += total_transfer_bytes

    def get_fleet_telemetry(self) -> Dict[str, VehicleTelemetry]:
        """Returns snapshot of current fleet telemetry."""
        return self.clients

    def get_summary_metrics(self) -> dict:
        """Returns cumulative resource and dropout metrics for the fleet."""
        return {
            "total_compute_energy_kwh": round(self.total_compute_energy_kwh, 5),
            "total_transfer_energy_kwh": round(self.total_transfer_energy_kwh, 5),
            "total_energy_kwh": round(self.total_compute_energy_kwh + self.total_transfer_energy_kwh, 5),
            "total_metered_mb": round(self.total_metered_bytes / (1024 * 1024), 2),
            "total_unmetered_mb": round(self.total_unmetered_bytes / (1024 * 1024), 2),
            "total_dropouts": self.total_dropouts,
        }
