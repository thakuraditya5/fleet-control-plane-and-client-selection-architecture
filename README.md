# Intelligent Fleet Control Plane & Client Selection Architecture for EV Federated Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Framework: Flower](https://img.shields.io/badge/FL_Framework-Flower-yellow.svg)](https://flower.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A lightweight, reproducible simulation framework based on the paper **"An On-Device Federated Learning Approach for Cooperative Model Update Between Edge Devices"**, implementing an **Intelligent Fleet Control Plane & Client Selection Architecture** for Electric Vehicle (EV) fleets.

The system simulates dynamic vehicle telemetry (battery charge, charging mode, compute energy, cellular vs. Wi-Fi connectivity) and enforces rule-based guardrails to prevent battery drain, eliminate costly cellular data transfer, and eliminate mid-round client dropouts.

---

## 🏛️ Architecture Overview

In an EV federated learning deployment, vehicle clients are highly dynamic edge nodes. Randomly selecting vehicles (the un-guarded baseline) causes critical issues:
1. **Drive Power Degradation:** Training while driving or with low state-of-charge (SoC) drains range and triggers ECU safety overrides.
2. **Expensive Cellular Costs:** Transmitting model weight updates over metered 4G/5G connections incurs high cellular network bills.
3. **Mid-Round Dropouts:** Dynamic vehicle state transitions (driving off, handoffs, thermal throttling) cause failed updates and wasted server aggregation rounds.

The **Intelligent Fleet Control Plane** queries vehicle telemetry prior to training assignment and executes strict **Power & Network Guardrails**.

```mermaid
flowchart TD
    subgraph Fleet ["Dynamic EV Fleet (20+ Edge Nodes)"]
        EV1["EV Client 01<br/>(LEVEL_2_CHARGING, 78% SoC, UNMETERED_WIFI)"]
        EV2["EV Client 02<br/>(DISCHARGING, 45% SoC, METERED_4G)"]
        EV3["EV Client 03<br/>(DC_FAST_CHARGING, 35% SoC, UNMETERED_WIFI)"]
        EV4["EV Client 04<br/>(DISCHARGING, 18% SoC, CELLULAR_5G)"]
        EVN["... EV Client N ..."]
    end

    subgraph ControlPlane ["Intelligent Fleet Control Plane (Flower Server)"]
        Telem["Telemetry Engine<br/>(src/telemetry.py)"]
        Orch["Orchestrator Rule Guardrails<br/>(src/orchestrator.py)"]
        Manager["EVClientManager<br/>(src/server.py)"]
        Strategy["EVFedAvg Strategy<br/>(src/server.py)"]
    end

    subgraph Guardrails ["Rule-Based Guardrail Verification"]
        PowerRule{"Power Guardrail<br/>(charging_state != DISCHARGING & SoC >= 50%)<br/>OR (DC_FAST & SoC >= 20%)"}
        NetRule{"Network Guardrail<br/>(UNMETERED_WIFI & Bandwidth >= 10 Mbps)"}
    end

    Fleet -->|Telemetry Ingestion| Telem
    Telem --> Orch
    Orch --> PowerRule
    PowerRule -->|Pass| NetRule
    PowerRule -->|Fail| Rejected["Rejected / Preserved (0 MB Cellular, 0% Battery Drain)"]
    NetRule -->|Pass| EligiblePool["Eligible Vehicle Pool"]
    NetRule -->|Fail| Rejected
    EligiblePool --> Manager
    Manager --> Strategy
    Strategy -->|Dispatch Model Weights (FitIns)| Fleet
```

---

## 📋 Rule Guardrail Specification

The `Orchestrator` (`src/orchestrator.py`) enforces two key guardrails before assigning a client to an FL training round:

### 1. Power Guardrail
A vehicle is eligible **ONLY IF**:
$$\Big(\text{charging\_state} \neq \text{DISCHARGING} \;\land\; \text{battery\_percentage} \ge 50.0\%\Big)$$
$$\mathbf{OR}$$
$$\Big(\text{charging\_state} == \text{DC\_FAST\_CHARGING} \;\land\; \text{battery\_percentage} \ge 20.0\%\Big)$$

* **Rationale:** Disallowing `DISCHARGING` ensures vehicles in active transit do not deplete drive range. Level 2 charging requires at least 50% SoC buffer, whereas high-rate DC Fast Chargers allow safe participation from 20% SoC upwards.

### 2. Bandwidth & Network Guardrail
A vehicle is eligible **ONLY IF**:
$$\text{network\_type} == \text{UNMETERED\_WIFI} \;\land\; \text{bandwidth\_mbps} \ge 10.0\text{ Mbps}$$

* **Rationale:** Completely offloads model weight transmission from expensive `METERED_4G` and `CELLULAR_5G` connections to stationary depot/home unmetered Wi-Fi, preventing carrier quota exhaustion and high latency timeouts.

---

## 📁 Repository Structure

```text
├── src/
│   ├── client.py        # Flower client implementation with telemetry hooks
│   ├── server.py        # Custom Flower strategy with custom client manager
│   ├── telemetry.py     # Synthetic EV telemetry stream simulator
│   ├── orchestrator.py  # Rule-based guardrails & selection engine
│   └── model.py         # PyTorch lightweight EdgeCNN model & non-IID partitioning
├── benchmarks.py        # FL experiment engine and metrics comparator
├── config.yaml          # Hyperparameters, fleet size, and rule thresholds
├── run_benchmark.py     # Main script executing Baseline vs. Intelligent runs
├── plot_results.py      # Publication-grade chart generation (Accuracy & Energy Saved)
├── requirements.txt     # Python package dependencies
└── README.md            # Architecture docs and execution guide
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/thakuraditya5/fleet-control-plane-and-client-selection-architecture.git
cd fleet-control-plane-and-client-selection-architecture
```

### 2. Environment Setup
Create and activate a virtual environment (Python 3.10+):

```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run Automated FL Benchmarks
Execute the complete parallel benchmark (Baseline vs. Constraint-Aware):
```bash
python run_benchmark.py
```
This script will:
- Partition the MNIST vision task non-IID across 20 dynamic EV clients.
- Run 15 federated learning rounds under the **Baseline (Random Selection)** strategy.
- Run 15 federated learning rounds under the **Constraint-Aware (Intelligent)** strategy with identical vehicle traces.
- Output a comprehensive comparative table to the console.
- Save detailed round-by-round metrics to `results/benchmark_results.json`.
- Automatically generate 4 comparison plots in the `results/` directory.

### 4. Generate Visualizations Independently
```bash
python plot_results.py results/benchmark_results.json results
```

Generated plots in `results/`:
- `accuracy_vs_rounds.png`: Test Accuracy & Loss convergence curves.
- `resource_preservation.png`: Fleet Energy (kWh) and Metered vs. Unmetered Data (MB).
- `client_dropouts.png`: Mid-round dropouts prevented per round.
- `comprehensive_dashboard.png`: 4-panel dashboard summarizing the entire benchmark.

---

## ⚙️ Configuration (`config.yaml`)

All simulation parameters can be configured in `config.yaml`:
```yaml
simulation:
  fleet_size: 20                   # Dynamic EV nodes
  num_rounds: 15                   # FL training rounds
  clients_per_round: 5             # Target clients sampled per round
  seed: 42                         # Random seed for reproducibility
  dataset: "MNIST"                 # Edge classification task
  batch_size: 32
  local_epochs: 1
  learning_rate: 0.01

telemetry:
  battery_capacity_kwh: 75.0       # Standard EV pack size
  training_energy_per_round_kwh: 0.025 # Compute energy per round
  transmission_power_w: 5.0        # RF transmission power (Watts)

guardrails:
  power:
    disallowed_states: ["DISCHARGING"]
    min_battery_general: 50.0
    min_battery_dc_fast: 20.0
  network:
    required_network_type: "UNMETERED_WIFI"
    min_bandwidth_mbps: 10.0
```

---

## 📊 Evaluation Metrics

The framework evaluates:
1. **Model Convergence Rate:** Global Test Accuracy (%) and Cross-Entropy Loss over rounds.
2. **Resource Preservation:**
   - Total battery power saved across the fleet (kWh).
   - Total data transferred over metered connections (`METERED_4G`, `CELLULAR_5G`) vs. unmetered (`UNMETERED_WIFI`).
   - Metered data avoidance rate (typically 100% avoided under constraint-aware selection).
3. **Client Reliability & Dropouts:**
   - Pre-round rejected client candidates.
   - Mid-round dropped clients due to dynamic driving handoffs, critical battery cutoffs, or cellular congestion.

---

## 🔬 Citation & Reference

Based on research principles from:
> *An On-Device Federated Learning Approach for Cooperative Model Update Between Edge Devices.*
