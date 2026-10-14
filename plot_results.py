"""
Visualization and Plotting Module for EV Federated Learning Benchmarks.
Generates publication-quality charts for convergence, resource preservation, and dropouts.
"""

import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Set aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.titlesize"] = 14


def plot_accuracy_convergence(data: dict, out_dir: str):
    """Generates Accuracy and Loss vs Round convergence plots."""
    b_history = data["baseline"]["round_history"]
    c_history = data["constraint_aware"]["round_history"]

    rounds = [r["round"] for r in b_history]
    b_acc = [r["test_accuracy"] for r in b_history]
    c_acc = [r["test_accuracy"] for r in c_history]

    b_loss = [r["test_loss"] for r in b_history]
    c_loss = [r["test_loss"] for r in c_history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Accuracy Plot
    ax1.plot(rounds, b_acc, marker="o", color="#d9534f", linestyle="--", linewidth=2, label="Baseline (Random)")
    ax1.plot(rounds, c_acc, marker="s", color="#2ca02c", linestyle="-", linewidth=2.5, label="Constraint-Aware (Intelligent)")
    ax1.set_title("Model Convergence: Test Accuracy vs. Round", fontweight="bold")
    ax1.set_xlabel("Federated Round")
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_ylim(0, 100)
    ax1.set_xticks(rounds)
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Loss Plot
    ax2.plot(rounds, b_loss, marker="o", color="#d9534f", linestyle="--", linewidth=2, label="Baseline (Random)")
    ax2.plot(rounds, c_loss, marker="s", color="#1f77b4", linestyle="-", linewidth=2.5, label="Constraint-Aware (Intelligent)")
    ax2.set_title("Model Convergence: Test Loss vs. Round", fontweight="bold")
    ax2.set_xlabel("Federated Round")
    ax2.set_ylabel("Cross-Entropy Loss")
    ax2.set_xticks(rounds)
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "accuracy_vs_rounds.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_resource_preservation(data: dict, out_dir: str):
    """Generates Resource Preservation comparison (Energy & Cellular Data Offloading)."""
    b_summary = data["baseline"]["summary"]
    c_summary = data["constraint_aware"]["summary"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # 1. Total Energy Consumed
    categories = ["Baseline (Random)", "Constraint-Aware"]
    compute_energy = [b_summary["total_compute_energy_kwh"], c_summary["total_compute_energy_kwh"]]
    transfer_energy = [b_summary["total_transfer_energy_kwh"], c_summary["total_transfer_energy_kwh"]]

    bars1 = ax1.bar(categories, compute_energy, label="Compute Energy (ECU)", color="#4575b4", width=0.5)
    bars2 = ax1.bar(categories, transfer_energy, bottom=compute_energy, label="Transmission Energy (RF)", color="#fdae61", width=0.5)

    ax1.set_title("Fleet Energy Consumption (kWh)", fontweight="bold")
    ax1.set_ylabel("Energy (kWh)")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)

    # Add value labels
    for i in range(len(categories)):
        tot = compute_energy[i] + transfer_energy[i]
        ax1.text(i, tot + 0.005, f"{tot:.4f} kWh", ha="center", fontweight="bold")

    # 2. Data Transferred: Metered vs Unmetered
    x = np.arange(len(categories))
    width = 0.35

    metered = [b_summary["total_metered_mb"], c_summary["total_metered_mb"]]
    unmetered = [b_summary["total_unmetered_mb"], c_summary["total_unmetered_mb"]]

    rects1 = ax2.bar(x - width/2, metered, width, label="Metered Cellular (4G/5G)", color="#d73027")
    rects2 = ax2.bar(x + width/2, unmetered, width, label="Unmetered Wi-Fi", color="#1a9850")

    ax2.set_title("Network Data Allocation (MB)", fontweight="bold")
    ax2.set_ylabel("Data Transferred (MB)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend(loc="upper left", frameon=True)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)

    for r in rects1:
        h = r.get_height()
        ax2.annotate(f"{h:.1f} MB", xy=(r.get_x() + r.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    for r in rects2:
        h = r.get_height()
        ax2.annotate(f"{h:.1f} MB", xy=(r.get_x() + r.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "resource_preservation.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_client_dropouts(data: dict, out_dir: str):
    """Generates Client Dropouts and Round Completion Rate plots."""
    b_history = data["baseline"]["round_history"]
    c_history = data["constraint_aware"]["round_history"]

    rounds = [r["round"] for r in b_history]
    b_dropouts = [r["dropouts_this_round"] for r in b_history]
    c_dropouts = [r["dropouts_this_round"] for r in c_history]

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.38
    x = np.array(rounds)

    ax.bar(x - bar_width/2, b_dropouts, width=bar_width, color="#d9534f", label="Baseline (Random) Mid-Round Dropouts")
    ax.bar(x + bar_width/2, c_dropouts, width=bar_width, color="#5cb85c", label="Constraint-Aware Mid-Round Dropouts")

    ax.set_title("Client Dropouts Due to Dynamic Vehicle State Changes Mid-Round", fontweight="bold")
    ax.set_xlabel("Federated Round")
    ax.set_ylabel("Number of Dropped Clients")
    ax.set_xticks(rounds)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "client_dropouts.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_comprehensive_dashboard(data: dict, out_dir: str):
    """Generates an all-in-one 4-panel dashboard summarizing the entire benchmark."""
    b_history = data["baseline"]["round_history"]
    c_history = data["constraint_aware"]["round_history"]
    b_summary = data["baseline"]["summary"]
    c_summary = data["constraint_aware"]["summary"]

    rounds = [r["round"] for r in b_history]
    b_acc = [r["test_accuracy"] for r in b_history]
    c_acc = [r["test_accuracy"] for r in c_history]

    b_cum_dropouts = np.cumsum([r["dropouts_this_round"] for r in b_history])
    c_cum_dropouts = np.cumsum([r["dropouts_this_round"] for r in c_history])

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("EV Federated Learning Fleet Control Plane Benchmark Dashboard", fontsize=16, fontweight="bold", y=0.98)

    # 1. Accuracy
    axs[0, 0].plot(rounds, b_acc, marker="o", color="#d9534f", linestyle="--", label="Baseline (Random)")
    axs[0, 0].plot(rounds, c_acc, marker="s", color="#2ca02c", linestyle="-", linewidth=2, label="Constraint-Aware")
    axs[0, 0].set_title("(a) Test Accuracy vs. Training Round", fontweight="bold")
    axs[0, 0].set_xlabel("Round")
    axs[0, 0].set_ylabel("Accuracy (%)")
    axs[0, 0].set_xticks(rounds)
    axs[0, 0].legend()
    axs[0, 0].grid(True, linestyle=":", alpha=0.6)

    # 2. Cumulative Dropouts
    axs[0, 1].plot(rounds, b_cum_dropouts, marker="o", color="#d9534f", linestyle="--", label="Baseline (Un-guarded)")
    axs[0, 1].plot(rounds, c_cum_dropouts, marker="s", color="#2ca02c", linestyle="-", linewidth=2, label="Constraint-Aware (Guarded)")
    axs[0, 1].set_title("(b) Cumulative Mid-Round Client Dropouts", fontweight="bold")
    axs[0, 1].set_xlabel("Round")
    axs[0, 1].set_ylabel("Cumulative Dropouts")
    axs[0, 1].set_xticks(rounds)
    axs[0, 1].legend()
    axs[0, 1].grid(True, linestyle=":", alpha=0.6)

    # 3. Data Transfer Allocation
    categories = ["Baseline", "Constraint-Aware"]
    metered = [b_summary["total_metered_mb"], c_summary["total_metered_mb"]]
    unmetered = [b_summary["total_unmetered_mb"], c_summary["total_unmetered_mb"]]
    x = np.arange(len(categories))
    w = 0.35
    axs[1, 0].bar(x - w/2, metered, w, label="Metered Cellular (4G/5G)", color="#d73027")
    axs[1, 0].bar(x + w/2, unmetered, w, label="Unmetered Wi-Fi", color="#1a9850")
    axs[1, 0].set_title("(c) Network Data Offloading (MB)", fontweight="bold")
    axs[1, 0].set_ylabel("MB Transferred")
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(categories)
    axs[1, 0].legend()
    axs[1, 0].grid(True, axis="y", linestyle=":", alpha=0.6)

    # 4. Energy Breakdown
    compute_energy = [b_summary["total_compute_energy_kwh"], c_summary["total_compute_energy_kwh"]]
    transfer_energy = [b_summary["total_transfer_energy_kwh"], c_summary["total_transfer_energy_kwh"]]
    axs[1, 1].bar(categories, compute_energy, label="Compute Energy", color="#4575b4", width=0.45)
    axs[1, 1].bar(categories, transfer_energy, bottom=compute_energy, label="Transmission Energy", color="#fdae61", width=0.45)
    axs[1, 1].set_title("(d) Total Fleet Energy Consumption (kWh)", fontweight="bold")
    axs[1, 1].set_ylabel("Energy (kWh)")
    axs[1, 1].legend()
    axs[1, 1].grid(True, axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "comprehensive_dashboard.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def generate_all_plots(results_file: str, out_dir: str):
    """Loads benchmark JSON results and outputs all visual figures."""
    with open(results_file, "r") as f:
        data = json.load(f)

    os.makedirs(out_dir, exist_ok=True)
    p1 = plot_accuracy_convergence(data, out_dir)
    p2 = plot_resource_preservation(data, out_dir)
    p3 = plot_client_dropouts(data, out_dir)
    p4 = plot_comprehensive_dashboard(data, out_dir)
    print(f"  -> Generated: {p1}")
    print(f"  -> Generated: {p2}")
    print(f"  -> Generated: {p3}")
    print(f"  -> Generated: {p4}")


if __name__ == "__main__":
    res_path = sys.argv[1] if len(sys.argv) > 1 else "./results/benchmark_results.json"
    target_dir = sys.argv[2] if len(sys.argv) > 2 else "./results"
    generate_all_plots(res_path, target_dir)
