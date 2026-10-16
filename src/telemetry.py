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


class AvailabilityState(str, Enum):
    DRIVING = "DRIVING"
    CHARGING_WIFI = "CHARGING_WIFI"
    CHARGING_METERED = "CHARGING_METERED"
    PARKED_WIFI = "PARKED_WIFI"


AVAILABILITY_TRANSITIONS = {
    "heavy-use": {
        AvailabilityState.DRIVING: (0.75, 0.10, 0.05, 0.10),
        AvailabilityState.CHARGING_WIFI: (0.35, 0.50, 0.05, 0.10),
        AvailabilityState.CHARGING_METERED: (0.50, 0.10, 0.30, 0.10),
        AvailabilityState.PARKED_WIFI: (0.45, 0.10, 0.05, 0.40),
    },
    "light-use": {
        AvailabilityState.DRIVING: (0.45, 0.30, 0.05, 0.20),
        AvailabilityState.CHARGING_WIFI: (0.08, 0.68, 0.04, 0.20),
        AvailabilityState.CHARGING_METERED: (0.20, 0.25, 0.40, 0.15),
        AvailabilityState.PARKED_WIFI: (0.08, 0.38, 0.04, 0.50),
    },
}
AVAILABILITY_STATES = tuple(AvailabilityState)


@dataclass
class VehicleTelemetry:
    client_id: str
    charging_state: ChargingState
    battery_percentage: float  # 0.0 to 100.0%
    network_type: NetworkType
    bandwidth_mbps: float
    battery_capacity_kwh: float = 75.0  # standard EV pack capacity
    usage_profile: str = "light-use"
    availability_state: AvailabilityState = AvailabilityState.PARKED_WIFI
    
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
        profile_share: float = 0.5,
        initial_state_probabilities: Optional[Dict[str, List[float]]] = None,
        transition_matrices: Optional[Dict[str, List[List[float]]]] = None,
        dropout_assumptions: Optional[Dict[str, float]] = None,
    ):
        self.num_clients = num_clients
        self.seed = seed
        self.rng = random.Random(seed)
        self.state_rng = random.Random(seed + 1)
        self.battery_capacity_kwh = battery_capacity_kwh
        self.training_energy_per_round_kwh = training_energy_per_round_kwh
        self.transmission_power_w = transmission_power_w
        self.profile_share = profile_share
        self.state_order = AVAILABILITY_STATES
        self.initial_state_probabilities = {
            profile: tuple(
                initial_state_probabilities[profile]
                if initial_state_probabilities
                else ([0.70, 0.10, 0.05, 0.15] if profile == "heavy-use" else [0.15, 0.40, 0.10, 0.35])
            )
            for profile in ("heavy-use", "light-use")
        }
        self.transitions = {}
        for profile in ("heavy-use", "light-use"):
            rows = transition_matrices.get(profile) if transition_matrices else None
            self.transitions[profile] = {
                state: tuple(rows[index]) if rows else AVAILABILITY_TRANSITIONS[profile][state]
                for index, state in enumerate(self.state_order)
            }
        self.dropout_assumptions = {
            "driving_probability": 0.50,
            "critical_battery_pct": 25.0,
            "low_battery_probability": 0.60,
            "low_bandwidth_mbps": 8.0,
            "poor_network_probability": 0.45,
        }
        if dropout_assumptions:
            self.dropout_assumptions.update(dropout_assumptions)
        
        # Cumulative fleet metrics
        self.total_compute_energy_kwh = 0.0
        self.total_transfer_energy_kwh = 0.0
        self.total_metered_bytes = 0
        self.total_unmetered_bytes = 0
        self.total_dropouts = 0
        self.usage_profiles: Dict[str, str] = {}

        # Initial client states
        self.clients: Dict[str, VehicleTelemetry] = {}
        self._initialize_fleet()
        self.selection_rng = random.Random(seed + 2)
        self.dropout_rng = random.Random(seed + 3)
        self.current_round = 0
        self.dropout_draws: Dict[str, Tuple[float, float, float]] = {}

    def reset(self, seed: Optional[int] = None):
        """Resets the fleet telemetry simulator to initial state with specified seed."""
        if seed is not None:
            self.seed = seed
        self.rng = random.Random(self.seed)
        self.state_rng = random.Random(self.seed + 1)
        self.total_compute_energy_kwh = 0.0
        self.total_transfer_energy_kwh = 0.0
        self.total_metered_bytes = 0
        self.total_unmetered_bytes = 0
        self.total_dropouts = 0
        self.usage_profiles = {}
        self.clients = {}
        self._initialize_fleet()
        self.selection_rng = random.Random(self.seed + 2)
        self.dropout_rng = random.Random(self.seed + 3)
        self.current_round = 0
        self.dropout_draws = {}

    def _initialize_fleet(self):
        """Creates a balanced fleet and samples initial profile-conditioned states."""
        heavy_count = round(self.num_clients * self.profile_share)
        profiles = ["heavy-use"] * heavy_count
        profiles.extend(["light-use"] * (self.num_clients - len(profiles)))
        self.rng.shuffle(profiles)

        for i in range(self.num_clients):
            cid = f"ev_client_{i:02d}"
            profile = profiles[i]
            availability_state = self.rng.choices(
                self.state_order,
                weights=self.initial_state_probabilities[profile],
            )[0]
            state, net, bw = self._telemetry_for_state(availability_state)
            battery = self.rng.uniform(20.0, 85.0)
            self.usage_profiles[cid] = profile

            self.clients[cid] = VehicleTelemetry(
                client_id=cid,
                charging_state=state,
                battery_percentage=battery,
                network_type=net,
                bandwidth_mbps=bw,
                battery_capacity_kwh=self.battery_capacity_kwh,
                usage_profile=profile,
                availability_state=availability_state,
            )

    def _telemetry_for_state(
        self, state: AvailabilityState, rng: Optional[random.Random] = None
    ) -> Tuple[ChargingState, NetworkType, float]:
        if rng is None:
            rng = self.rng
        if state == AvailabilityState.DRIVING:
            return (
                ChargingState.DISCHARGING,
                rng.choice([NetworkType.METERED_4G, NetworkType.CELLULAR_5G]),
                rng.uniform(2.0, 30.0),
            )
        if state == AvailabilityState.CHARGING_WIFI:
            return (
                ChargingState.LEVEL_2_CHARGING,
                NetworkType.UNMETERED_WIFI,
                rng.uniform(12.0, 60.0),
            )
        if state == AvailabilityState.CHARGING_METERED:
            return (
                ChargingState.DC_FAST_CHARGING,
                NetworkType.CELLULAR_5G,
                rng.uniform(20.0, 100.0),
            )
        return (
            ChargingState.DISCHARGING,
            NetworkType.UNMETERED_WIFI,
            rng.uniform(12.0, 60.0),
        )

    def advance_round(self, round_num: int):
        """
        Advances the fleet state to the next federated round.
        Simulates natural vehicle state transitions over time.
        """
        self.current_round = round_num
        for cid, t in self.clients.items():
            self.dropout_draws[cid] = (
                self.dropout_rng.random(),
                self.dropout_rng.random(),
                self.dropout_rng.random(),
            )
            # Reset round-specific flags
            t.dropped_out = False
            t.dropout_reason = None

            prior_state = t.availability_state
            transition = self.transitions[t.usage_profile][prior_state]
            next_state = self.state_rng.choices(self.state_order, weights=transition)[0]
            t.availability_state = next_state

            if next_state == AvailabilityState.DRIVING:
                t.battery_percentage = max(5.0, t.battery_percentage - self.state_rng.uniform(1.0, 3.5))
            elif next_state == AvailabilityState.CHARGING_WIFI:
                t.battery_percentage = min(100.0, t.battery_percentage + self.state_rng.uniform(3.0, 8.0))
            elif next_state == AvailabilityState.CHARGING_METERED:
                t.battery_percentage = min(100.0, t.battery_percentage + self.state_rng.uniform(10.0, 20.0))

            t.charging_state, t.network_type, t.bandwidth_mbps = self._telemetry_for_state(next_state, self.state_rng)

    def simulate_mid_round_dynamics(self, client_id: str, is_baseline: bool) -> Tuple[bool, Optional[str]]:
        """
        Simulates real-world mid-round execution dynamics.
        If a client was assigned while non-eligible (e.g. DISCHARGING or METERED_4G in baseline),
        it has a high probability of mid-round dropout (packet loss, vehicle start-up, low-battery cutoff).
        """
        t = self.clients[client_id]
        draws = self.dropout_draws.get(client_id)
        if draws is None:
            draws = (
                self.dropout_rng.random(),
                self.dropout_rng.random(),
                self.dropout_rng.random(),
            )
        driving_draw, battery_draw, network_draw = draws

        # Inherent stability check:
        if t.availability_state == AvailabilityState.DRIVING:
            # Driving vehicles frequently drop connection or terminate background ML jobs for ECU safety
            if driving_draw < self.dropout_assumptions["driving_probability"]:
                t.dropped_out = True
                t.dropout_reason = "ECU Preempted: Vehicle active driving mode & safety override"
                self.total_dropouts += 1
                return False, t.dropout_reason

        if t.battery_percentage < self.dropout_assumptions["critical_battery_pct"]:
            # Critical low battery drop
            if battery_draw < self.dropout_assumptions["low_battery_probability"]:
                t.dropped_out = True
                t.dropout_reason = f"Low battery shutdown ({t.battery_percentage:.1f}% < 25%)"
                self.total_dropouts += 1
                return False, t.dropout_reason

        if (
            t.network_type == NetworkType.METERED_4G
            or t.bandwidth_mbps < self.dropout_assumptions["low_bandwidth_mbps"]
        ):
            # Cellular congestion / handover timeout
            if network_draw < self.dropout_assumptions["poor_network_probability"]:
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
