"""
Intelligent Orchestration & Client Selection Engine for EV Federated Learning.
Implements Rule-Based Guardrails (Power, Connectivity, Bandwidth) and Selection Controllers.
"""

from dataclasses import dataclass
import logging
import random
from typing import Dict, List, Optional, Tuple

from src.telemetry import ChargingState, NetworkType, VehicleTelemetry

logger = logging.getLogger("EVOrchestrator")


@dataclass
class EligibilityResult:
    client_id: str
    is_eligible: bool
    passed_power: bool
    passed_network: bool
    reason: str


class Orchestrator:
    """
    Control Plane Orchestration Engine.
    Enforces Power Guardrails and Bandwidth/Network Guardrails based on real-time EV telemetry.
    """

    def __init__(
        self,
        min_battery_general: float = 50.0,
        min_battery_dc_fast: float = 20.0,
        required_network: NetworkType = NetworkType.UNMETERED_WIFI,
        min_bandwidth_mbps: float = 10.0,
    ):
        self.min_battery_general = min_battery_general
        self.min_battery_dc_fast = min_battery_dc_fast
        self.required_network = required_network
        self.min_bandwidth_mbps = min_bandwidth_mbps
        self.rejection_history: List[EligibilityResult] = []

    def evaluate_client(self, telemetry: VehicleTelemetry) -> EligibilityResult:
        """
        Evaluates a client against guardrails:
        1. Power Guardrail:
           - charging_state != DISCHARGING AND battery_percentage >= 50.0
           OR
           - charging_state == DC_FAST_CHARGING AND battery_percentage >= 20.0
        2. Bandwidth/Network Guardrail:
           - network_type == UNMETERED_WIFI AND bandwidth_mbps >= 10.0
        """
        cid = telemetry.client_id
        cs = telemetry.charging_state
        bp = telemetry.battery_percentage
        nt = telemetry.network_type
        bw = telemetry.bandwidth_mbps

        # 1. Power Guardrail check
        passed_power = False
        power_reason = ""
        if cs == ChargingState.DC_FAST_CHARGING and bp >= self.min_battery_dc_fast:
            passed_power = True
        elif cs != ChargingState.DISCHARGING and bp >= self.min_battery_general:
            passed_power = True
        else:
            if cs == ChargingState.DISCHARGING:
                power_reason = "Vehicle is DISCHARGING (Driving/Parked off-grid)"
            elif cs == ChargingState.DC_FAST_CHARGING and bp < self.min_battery_dc_fast:
                power_reason = f"DC Fast Charging battery ({bp:.1f}%) < threshold ({self.min_battery_dc_fast:.1f}%)"
            else:
                power_reason = f"Battery ({bp:.1f}%) < required general threshold ({self.min_battery_general:.1f}%)"

        # 2. Bandwidth/Network Guardrail check
        passed_network = False
        net_reason = ""
        if nt == self.required_network and bw >= self.min_bandwidth_mbps:
            passed_network = True
        else:
            if nt != self.required_network:
                net_reason = f"Network is {nt.value} (requires {self.required_network.value})"
            if bw < self.min_bandwidth_mbps:
                net_bw_msg = f"Bandwidth ({bw:.1f} Mbps) < min required ({self.min_bandwidth_mbps:.1f} Mbps)"
                net_reason = f"{net_reason}; {net_bw_msg}" if net_reason else net_bw_msg

        is_eligible = passed_power and passed_network
        reasons = []
        if not passed_power:
            reasons.append(f"[Power Violation: {power_reason}]")
        if not passed_network:
            reasons.append(f"[Network Violation: {net_reason}]")

        summary_reason = "ACCEPTED" if is_eligible else " REJECTED: " + " ".join(reasons)

        result = EligibilityResult(
            client_id=cid,
            is_eligible=is_eligible,
            passed_power=passed_power,
            passed_network=passed_network,
            reason=summary_reason,
        )
        self.rejection_history.append(result)
        return result

    def filter_eligible_clients(
        self, fleet_telemetry: Dict[str, VehicleTelemetry]
    ) -> Tuple[List[str], Dict[str, EligibilityResult]]:
        """Filters fleet telemetry and returns eligible client IDs alongside audit results."""
        eligible_cids = []
        evaluations = {}

        for cid, telemetry in fleet_telemetry.items():
            res = self.evaluate_client(telemetry)
            evaluations[cid] = res
            if res.is_eligible:
                eligible_cids.append(cid)

        return eligible_cids, evaluations

    def select_clients(
        self,
        fleet_telemetry: Dict[str, VehicleTelemetry],
        target_count: int,
        strategy_type: str = "constraint_aware",
        rng: Optional[random.Random] = None,
    ) -> Tuple[List[str], Dict[str, EligibilityResult]]:
        """
        Orchestration selection controller:
        - 'baseline_random': Uniform random selection ignoring guardrails
        - 'constraint_aware': Filters through guardrails first, then selects target_count
        """
        if rng is None:
            rng = random.Random()

        all_cids = sorted(list(fleet_telemetry.keys()))
        evaluations = {cid: self.evaluate_client(fleet_telemetry[cid]) for cid in all_cids}

        if strategy_type == "baseline_random":
            # Baseline samples randomly from entire fleet
            k = min(target_count, len(all_cids))
            selected = rng.sample(all_cids, k)
            return selected, evaluations

        elif strategy_type == "constraint_aware":
            # Filter first
            eligible = [cid for cid in all_cids if evaluations[cid].is_eligible]
            if len(eligible) == 0:
                # If zero meet criteria, fallback gracefully or report 0
                return [], evaluations
            k = min(target_count, len(eligible))
            selected = rng.sample(eligible, k)
            return selected, evaluations

        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
