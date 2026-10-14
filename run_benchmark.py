"""
Main Execution Script for EV Federated Learning Benchmarks.
Executes Baseline Random vs. Constraint-Aware Guardrail Selection for 15-20 rounds.
Saves metrics to results/benchmark_results.json and displays a comparative summary.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import sys
import time
import yaml
import torch

from benchmarks import run_fl_experiment, compare_benchmarks
from src.model import partition_dataset


def print_comparison_table(comparison: dict):
    """Prints a structured ASCII table highlighting key metrics and resource preservation."""
    b = comparison["baseline"]
    c = comparison["constraint_aware"]
    imp = comparison["improvements"]

    print("\n" + "=" * 84)
    print("      INTELLIGENT FLEET CONTROL PLANE & CLIENT SELECTION BENCHMARK REPORT")
    print("=" * 84)
    print(f"{'Performance Metric':<35} | {'Baseline (Random)':<18} | {'Constraint-Aware':<18} | {'Impact / Gain'}")
    print("-" * 84)
    print(f"{'Model Test Accuracy (%)':<35} | {b['final_accuracy']:<18.2f} | {c['final_accuracy']:<18.2f} | {imp['accuracy_delta']:+.2f}%")
    print(f"{'Model Test Loss':<35} | {b['final_loss']:<18.4f} | {c['final_loss']:<18.4f} | {b['final_loss'] - c['final_loss']:+.4f} (lower is better)")
    print(f"{'Round Completion Rate (%)':<35} | {b['completion_rate_pct']:<18.2f} | {c['completion_rate_pct']:<18.2f} | {imp['completion_rate_improvement_pct']:+.2f}%")
    print(f"{'Mid-Round Client Dropouts':<35} | {b['total_dropouts']:<18} | {c['total_dropouts']:<18} | {-imp['dropouts_prevented']} prevented")
    print("-" * 84)
    print(f"{'Total Fleet Energy Used (kWh)':<35} | {b['total_energy_kwh']:<18.4f} | {c['total_energy_kwh']:<18.4f} | {imp['energy_saved_kwh']:+.4f} kWh saved")
    print(f"{'Metered Cellular Data (MB)':<35} | {b['total_metered_mb']:<18.2f} | {c['total_metered_mb']:<18.2f} | -{imp['metered_reduction_pct']:.1f}% avoided")
    print(f"{'Unmetered Wi-Fi Data (MB)':<35} | {b['total_unmetered_mb']:<18.2f} | {c['total_unmetered_mb']:<18.2f} | {c['total_unmetered_mb'] - b['total_unmetered_mb']:+.2f} MB")
    print("=" * 84 + "\n")


def main():
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    out_dir = config.get("benchmarks", {}).get("output_dir", "./results")
    os.makedirs(out_dir, exist_ok=True)

    sim_cfg = config.get("simulation", {})
    num_clients = sim_cfg.get("fleet_size", 20)
    batch_size = sim_cfg.get("batch_size", 32)
    seed = sim_cfg.get("seed", 42)
    num_rounds = sim_cfg.get("num_rounds", 15)

    print(f"\n=======================================================")
    print(f" EV FEDERATED LEARNING SIMULATION BENCHMARK")
    print(f" Fleet Size: {num_clients} nodes | Rounds: {num_rounds} | Dataset: MNIST")
    print(f"=======================================================\n")

    # 1. Partition Dataset
    print(f"[*] Partitioning dataset across {num_clients} EV clients...")
    client_loaders, test_loader = partition_dataset(
        num_clients=num_clients,
        batch_size=batch_size,
        data_dir="./data",
        seed=seed,
    )

    # 2. Run Baseline Experiment (Random Selection)
    print("\n>>> RUNNING EXPERIMENT 1/2: BASELINE STRATEGY (RANDOM CLIENT SELECTION)")
    start_time = time.time()
    baseline_results = run_fl_experiment(
        strategy_type="baseline_random",
        config=config,
        client_loaders=client_loaders,
        test_loader=test_loader,
    )
    baseline_time = time.time() - start_time
    print(f"[✓] Baseline Run Finished in {baseline_time:.1f}s")

    # 3. Run Constraint-Aware Experiment (Intelligent Guardrails)
    print("\n>>> RUNNING EXPERIMENT 2/2: INTELLIGENT STRATEGY (CONSTRAINT-AWARE SELECTION)")
    start_time = time.time()
    constraint_results = run_fl_experiment(
        strategy_type="constraint_aware",
        config=config,
        client_loaders=client_loaders,
        test_loader=test_loader,
    )
    constraint_time = time.time() - start_time
    print(f"[✓] Constraint-Aware Run Finished in {constraint_time:.1f}s")

    # 4. Compare Benchmarks
    comparison = compare_benchmarks(baseline_results, constraint_results)

    # 5. Save Output
    results_payload = {
        "config": config,
        "baseline": baseline_results,
        "constraint_aware": constraint_results,
        "comparison": comparison,
    }
    json_path = os.path.join(out_dir, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\n[+] Detailed benchmark results saved to: {json_path}")

    # 6. Display Summary Table
    print_comparison_table(comparison)

    # 7. Generate Plots
    print("[*] Generating comparison plots...")
    from plot_results import generate_all_plots
    generate_all_plots(json_path, out_dir)
    print(f"[✓] Plots generated successfully in {out_dir}/")


if __name__ == "__main__":
    main()
