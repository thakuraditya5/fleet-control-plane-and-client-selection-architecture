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


def generate_study_plots(data: dict, out_dir: str, rho: float = 1.0):
    """Generate mechanism and tradeoff plots from availability-study results."""
    os.makedirs(out_dir, exist_ok=True)
    selected = [row for row in data["results"] if row["rho"] == rho]
    policies = list(dict.fromkeys(row["policy"] for row in selected))
    colors = {
        "A": "#d95f02",
        "B": "#1b9e77",
        "C": "#7570b3",
        "D": "#e7298a",
        "P2P": "#666666",
        "E1": "#009e73",
        "E2": "#cc79a7",
        "F": "#e69f00",
        "P2P-2": "#56b4e9",
    }

    fig, axes = plt.subplots(len(policies), 1, figsize=(14, max(3.0 * len(policies), 6)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, policy in zip(axes, policies):
        policy_runs = [row for row in selected if row["policy"] == policy]
        client_ids = sorted(policy_runs[0]["participation_by_client"])
        heavy_values = []
        light_values = []
        for cid in client_ids:
            heavy_values.append(np.mean([
                row["participation_by_client"][cid]
                for row in policy_runs
                if row["fleet_profiles"][cid] == "heavy-use"
            ]) if any(row["fleet_profiles"][cid] == "heavy-use" for row in policy_runs) else np.nan)
            light_values.append(np.mean([
                row["participation_by_client"][cid]
                for row in policy_runs
                if row["fleet_profiles"][cid] == "light-use"
            ]) if any(row["fleet_profiles"][cid] == "light-use" for row in policy_runs) else np.nan)
        x = np.arange(len(client_ids))
        ax.bar(x - 0.2, heavy_values, color="#0072b2", width=0.4, label="Heavy-use")
        ax.bar(x + 0.2, light_values, color="#d55e00", width=0.4, label="Light-use")
        ax.set_xticks(x)
        ax.set_xticklabels(client_ids)
        ax.set_ylabel("Rounds")
        ax.set_title(f"{policy}: {policy_runs[0]['policy_name']}")
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)
        ax.tick_params(axis="x", labelrotation=70, labelsize=7)
        ax.legend(ncol=2, fontsize=8)
    axes[-1].set_xlabel("Vehicle ID; bars show runs where that ID held the profile")
    fig.suptitle(f"Participation per vehicle at $\\rho={rho:g}$")
    fig.tight_layout()
    participation_path = os.path.join(out_dir, "participation_by_vehicle.png")
    fig.savefig(participation_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for policy in policies:
        policy_runs = [row for row in selected if row["policy"] == policy]
        rounds = policy_runs[0]["round_history"]
        coverage = [
            np.mean([
                sum(value > 0 for value in run["round_history"][round_index]["class_coverage"])
                for run in policy_runs
            ])
            for round_index, _ in enumerate(rounds)
        ]
        ax.plot([row["round"] for row in rounds], coverage, label=policy_runs[0]["policy_name"], color=colors[policy])
    ax.set(title=f"Selected-client class coverage at $\\rho={rho:g}$", xlabel="Round", ylabel="Mean represented classes (of 10)")
    ax.set_ylim(0, 10)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)
    coverage_path = os.path.join(out_dir, "class_coverage_by_round.png")
    fig.tight_layout()
    fig.savefig(coverage_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for policy in policies:
        points_x = []
        points_y = []
        for run in (row for row in selected if row["policy"] == policy):
            for round_row in run["round_history"]:
                if round_row["evaluated"]:
                    points_x.extend(round_row["class_coverage_window"])
                    points_y.extend(round_row["per_class_accuracy"])
        ax.scatter(points_x, points_y, s=10, alpha=0.22, label=policy, color=colors[policy])
    ax.set(title=f"Per-class accuracy versus selected-client coverage ($\\rho={rho:g}$)", xlabel="Selected clients holding class", ylabel="Test accuracy by class (%)")
    ax.legend(title="Policy")
    ax.grid(True, linestyle=":", alpha=0.5)
    relation_path = os.path.join(out_dir, "coverage_vs_class_accuracy.png")
    fig.tight_layout()
    fig.savefig(relation_path, dpi=180)
    plt.close(fig)

    fig, (energy_ax, data_ax) = plt.subplots(1, 2, figsize=(13, 6))
    for policy in policies:
        policy_runs = [row for row in selected if row["policy"] == policy]
        energy_ax.scatter(
            [row.get("energy_kwh_per_selected_client_round", row["total_energy_kwh"]) for row in policy_runs],
            [row["worst_class_accuracy_pct"] for row in policy_runs],
            label=policy,
            color=colors[policy],
            alpha=0.7,
        )
        data_ax.scatter(
            [row.get("metered_mb_per_selected_client_round", row["metered_mb"]) for row in policy_runs],
            [row["mean_class_accuracy_pct"] for row in policy_runs],
            label=policy,
            color=colors[policy],
            alpha=0.7,
        )
    energy_ax.set(xlabel="Energy per selected client-round (kWh)", ylabel="Worst-class accuracy (%)", title=f"Energy vs. worst-class accuracy ($\\rho={rho:g}$)")
    data_ax.set(xlabel="Metered MB per selected client-round", ylabel="Mean per-class accuracy (%)", title=f"Metered data vs. mean class accuracy ($\\rho={rho:g}$)")
    for ax in (energy_ax, data_ax):
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(title="Policy")
    tradeoff_path = os.path.join(out_dir, "resource_class_accuracy_tradeoff.png")
    fig.tight_layout()
    fig.savefig(tradeoff_path, dpi=180)
    plt.close(fig)

    fig, (final_ax, worst_ax) = plt.subplots(1, 2, figsize=(13, 6))
    difference_data = data.get("paired_policy_differences_vs_A", {})
    for ax, metric, title in (
        (final_ax, "final_accuracy_pct", "Final-window accuracy difference"),
        (worst_ax, "worst_class_accuracy_pct", "Worst-class accuracy difference"),
    ):
        labels = []
        means = []
        lower_errors = []
        upper_errors = []
        for policy in policies:
            if policy == "A":
                continue
            stats = difference_data.get(f"rho={rho:g}/policy={policy}-minus-A", {}).get(metric)
            if not stats or stats["mean"] is None or stats.get("ci95_lower") is None:
                continue
            labels.append(policy)
            means.append(stats["mean"])
            lower_errors.append(stats["mean"] - stats["ci95_lower"])
            upper_errors.append(stats["ci95_upper"] - stats["mean"])
        if labels:
            y = np.arange(len(labels))
            ax.errorbar(
                means,
                y,
                xerr=[lower_errors, upper_errors],
                fmt="o",
                capsize=4,
                color="#0072b2",
            )
            ax.axvline(0, color="#555555", linewidth=1, linestyle="--")
            ax.set_yticks(y)
            ax.set_yticklabels(labels)
        ax.set(title=f"{title} vs A ($\\rho={rho:g}$)", xlabel="Paired difference (percentage points)")
        ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    paired_path = os.path.join(out_dir, "paired_policy_differences_ci.png")
    fig.tight_layout()
    fig.savefig(paired_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for policy in policies:
        policy_runs = [row for row in selected if row["policy"] == policy]
        mins = [min(row["participation_by_client"].values()) for row in policy_runs]
        jitter = np.linspace(-0.12, 0.12, len(mins)) if len(mins) > 1 else [0.0]
        ax.scatter(np.full(len(mins), policy, dtype=object), np.asarray(mins) + jitter, color=colors[policy], label=policy)
        under_five = sum(value < 5 for value in mins) / max(len(mins), 1) * 100
        if mins:
            ax.text(policy, max(mins) + 0.7, f"<5: {under_five:.0f}%", ha="center", fontsize=8)
    ax.axhline(5, color="#555555", linewidth=1, linestyle="--", label="5-round threshold")
    ax.set(title=f"Per-run minimum vehicle participation ($\\rho={rho:g}$)", xlabel="Policy", ylabel="Minimum rounds for any vehicle")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    starvation_path = os.path.join(out_dir, "participation_min_by_seed.png")
    fig.tight_layout()
    fig.savefig(starvation_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    labels = list(range(10))
    x = np.arange(10)
    width = 0.8 / max(len(policies), 1)
    for index, policy in enumerate(policies):
        policy_runs = [row for row in selected if row["policy"] == policy]
        if not policy_runs:
            continue
        mean_coverage = [
            np.mean([
                np.mean([round_row["class_coverage"][label] for round_row in run["round_history"]])
                for run in policy_runs
            ])
            for label in labels
        ]
        ax.bar(x - 0.4 + width / 2 + index * width, mean_coverage, width, color=colors[policy], label=policy)
    ax.set(title=f"Selected-client coverage by class ($\\rho={rho:g}$)", xlabel="MNIST class", ylabel="Mean selected clients holding class per round")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(title="Policy")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    class_coverage_path = os.path.join(out_dir, "coverage_by_class.png")
    fig.tight_layout()
    fig.savefig(class_coverage_path, dpi=180)
    plt.close(fig)

    if all("participation_weighted_class_mix_pct" in row for row in selected):
        fig, ax = plt.subplots(figsize=(12, 5))
        width = 0.8 / max(len(policies), 1)
        for index, policy in enumerate(policies):
            policy_runs = [row for row in selected if row["policy"] == policy]
            if not policy_runs:
                continue
            class_mix = [
                np.mean([run["participation_weighted_class_mix_pct"][label] for run in policy_runs])
                for label in range(10)
            ]
            ax.bar(x - 0.4 + width / 2 + index * width, class_mix, width, color=colors[policy], label=policy)
        ax.axhline(10.0, color="#555555", linestyle="--", linewidth=1, label="10% balanced reference")
        ax.set(title=f"Participation-weighted training-label mix ($\\rho={rho:g}$)", xlabel="MNIST class", ylabel="Share of selected local training examples (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(title="Policy")
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)
        class_mix_path = os.path.join(out_dir, "participation_class_mix.png")
        fig.tight_layout()
        fig.savefig(class_mix_path, dpi=180)
        plt.close(fig)
    else:
        class_mix_path = None

    paths = [participation_path, coverage_path, relation_path, tradeoff_path, paired_path, starvation_path, class_coverage_path]
    if class_mix_path:
        paths.append(class_mix_path)
    diagnostic = data.get("staleness_bonus_diagnostic")
    if diagnostic:
        diagnostic_runs = diagnostic["runs"]
        keys = list(diagnostic["summary_mean_std_ci95"])
        fig, (min_ax, spread_ax) = plt.subplots(1, 2, figsize=(12, 5))
        x = np.arange(len(keys))
        min_stats = [diagnostic["summary_mean_std_ci95"][key]["participation_min_rounds"] for key in keys]
        spread_stats = [diagnostic["summary_mean_std_ci95"][key]["participation_spread_rounds"] for key in keys]
        for ax, stats, title, ylabel in (
            (min_ax, min_stats, "Minimum participation", "Min rounds per vehicle"),
            (spread_ax, spread_stats, "Participation spread", "Max minus min rounds"),
        ):
            means = [value["mean"] for value in stats]
            lower = [value["mean"] - value["ci95_lower"] if value["ci95_lower"] is not None else 0 for value in stats]
            upper = [value["ci95_upper"] - value["mean"] if value["ci95_upper"] is not None else 0 for value in stats]
            ax.errorbar(x, means, yerr=[lower, upper], fmt="o", capsize=4, color="#0072b2")
            ax.set_xticks(x)
            ax.set_xticklabels(keys, rotation=35, ha="right")
            ax.set(title=f"{title} ($\\rho={rho:g}$)", ylabel=ylabel)
            ax.grid(True, axis="y", linestyle=":", alpha=0.5)
        bonus_path = os.path.join(out_dir, "staleness_bonus_participation.png")
        fig.tight_layout()
        fig.savefig(bonus_path, dpi=180)
        plt.close(fig)
        paths.append(bonus_path)
    print("Study plots generated:")
    for path in paths:
        print(f"  -> {path}")
    return paths


def generate_final_plots(report: dict, out_dir: str, rho: float = 1.0):
    """Generate pre-registered final-run figures from seed-level output records."""
    os.makedirs(out_dir, exist_ok=True)
    selected = [
        row for row in report["results"]
        if row["rho"] == rho and row.get("kind") == "primary"
    ]
    policies = list(dict.fromkeys(row["policy"] for row in selected))
    colors = {
        "A": "#d95f02", "B": "#1b9e77", "C": "#7570b3", "D": "#e7298a",
        "E1": "#009e73", "E2": "#cc79a7", "F": "#e69f00",
        "P2P": "#666666", "P2P-2": "#56b4e9",
    }

    fig, ax = plt.subplots(figsize=(13, 6))
    x_positions = np.arange(len(policies))
    for profile, color, offset in (("heavy-use", "#0072b2", -0.13), ("light-use", "#d55e00", 0.13)):
        for policy_index, policy in enumerate(policies):
            runs = [row for row in selected if row["policy"] == policy]
            values = [
                run["participation_profile_summary"][profile]["mean"]
                for run in runs
            ]
            jitter = np.linspace(-0.06, 0.06, len(values)) if len(values) > 1 else [0.0]
            ax.scatter(
                np.full(len(values), x_positions[policy_index] + offset) + jitter,
                values,
                color=color,
                alpha=0.7,
                label=profile if policy_index == 0 else None,
            )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(policies)
    ax.set(title=f"Per-seed participation by usage profile ($\\rho={rho:g}$)", xlabel="Policy", ylabel="Mean selected rounds per vehicle in profile")
    ax.legend(title="Vehicle profile")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    participation_path = os.path.join(out_dir, "final_participation_by_profile_seed.png")
    fig.tight_layout()
    fig.savefig(participation_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, profile in zip(axes, ("heavy-use", "light-use")):
        for policy in policies:
            policy_runs = [row for row in selected if row["policy"] == policy]
            if not policy_runs:
                continue
            round_count = len(policy_runs[0]["eligible_pool_by_profile_by_round"])
            means = [
                np.mean([
                    run["eligible_pool_by_profile_by_round"][round_index][profile]
                    for run in policy_runs
                ])
                for round_index in range(round_count)
            ]
            ax.plot(np.arange(1, round_count + 1), means, label=policy, color=colors[policy])
        ax.set(title=f"{profile} eligible pool ($\\rho={rho:g}$)", xlabel="Round", ylabel="Eligible vehicles")
        ax.grid(True, linestyle=":", alpha=0.5)
    axes[-1].legend(title="Policy", ncol=2)
    eligibility_path = os.path.join(out_dir, "eligible_pool_by_profile.png")
    fig.tight_layout()
    fig.savefig(eligibility_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    means = []
    lows = []
    highs = []
    for policy in policies:
        stats = report["summary_mean_std_ci95"].get(f"rho={rho:g}/policy={policy}", {}).get("participation_weighted_heavy_class_share_pct")
        if not stats:
            continue
        means.append(stats["mean"])
        lows.append(stats["ci95_lower"] if stats["ci95_lower"] is not None else stats["mean"])
        highs.append(stats["ci95_upper"] if stats["ci95_upper"] is not None else stats["mean"])
    class_mass_policies = [policy for policy in policies if f"rho={rho:g}/policy={policy}" in report["summary_mean_std_ci95"]]
    if means:
        x = np.arange(len(means))
        ax.bar(x, means, color=[colors[policy] for policy in class_mass_policies])
        ax.errorbar(x, means, yerr=[np.asarray(means) - lows, np.asarray(highs) - means], fmt="none", ecolor="#333333", capsize=4)
        ax.axhline(50.0, color="#555555", linestyle="--", label="Balanced 50% reference")
        ax.set_xticks(x)
        ax.set_xticklabels(class_mass_policies)
    ax.set(title=f"Participation-weighted classes 5-9 training mass ($\\rho={rho:g}$)", xlabel="Policy", ylabel="Classes 5-9 share of selected examples (%)")
    ax.legend()
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    class_mass_path = os.path.join(out_dir, "final_class_mass_share.png")
    fig.tight_layout()
    fig.savefig(class_mass_path, dpi=180)
    plt.close(fig)

    difference_stats = report.get("paired_differences_vs_A_t_and_bootstrap_ci95", {})
    fig, axes = plt.subplots(1, 2, figsize=(15, max(6, 0.55 * len(policies))))
    forest_specs = (
        ("worst_class_accuracy_pct", "Worst-class accuracy difference (pp)"),
        ("heavy_class_accuracy_pct", "Classes 5-9 accuracy difference (pp)"),
    )
    for ax, (metric, label) in zip(axes, forest_specs):
        plotted = []
        for policy in policies:
            if policy == "A":
                continue
            stats = difference_stats.get(f"rho={rho:g}/policy={policy}-minus-A", {}).get(metric)
            if stats and stats["mean"] is not None:
                plotted.append((policy, stats))
        y = np.arange(len(plotted))
        for y_pos, (policy, stats) in enumerate(plotted):
            t_lower = stats.get("ci95_lower")
            t_upper = stats.get("ci95_upper")
            boot_lower = stats.get("bootstrap_ci95_lower")
            boot_upper = stats.get("bootstrap_ci95_upper")
            if t_lower is not None:
                ax.errorbar(stats["mean"], y_pos - 0.07, xerr=[[stats["mean"] - t_lower], [t_upper - stats["mean"]]], fmt="o", capsize=3, color=colors[policy], label="t interval" if y_pos == 0 else None)
            if boot_lower is not None:
                ax.errorbar(stats["mean"], y_pos + 0.07, xerr=[[stats["mean"] - boot_lower], [boot_upper - stats["mean"]]], fmt="|", capsize=5, color="#222222", label="bootstrap percentile" if y_pos == 0 else None)
        ax.axvline(0, color="#555555", linestyle="--", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels([policy for policy, _ in plotted])
        ax.set(title=f"{label} vs A ($\\rho={rho:g}$)", xlabel="Paired difference (percentage points)")
        ax.grid(True, axis="x", linestyle=":", alpha=0.5)
        ax.legend(fontsize=8)
    forest_path = os.path.join(out_dir, "final_paired_forest_ci.png")
    fig.tight_layout()
    fig.savefig(forest_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    resource_metrics = (
        ("total_energy_kwh", "Modeled energy (kWh)"),
        ("metered_mb", "Metered traffic (MB)"),
    )
    for ax, (resource_metric, xlabel) in zip(axes, resource_metrics):
        for policy_index, policy in enumerate(policies):
            stats = report["summary_mean_std_ci95"].get(f"rho={rho:g}/policy={policy}", {})
            resource = stats.get(resource_metric)
            accuracy = stats.get("worst_class_accuracy_pct")
            if not resource or not accuracy:
                continue
            xerr = None if resource["ci95_lower"] is None else [[resource["mean"] - resource["ci95_lower"]], [resource["ci95_upper"] - resource["mean"]]]
            yerr = None if accuracy["ci95_lower"] is None else [[accuracy["mean"] - accuracy["ci95_lower"]], [accuracy["ci95_upper"] - accuracy["mean"]]]
            ax.errorbar(resource["mean"], accuracy["mean"], xerr=xerr, yerr=yerr, fmt="o", capsize=3, color=colors[policy], label=policy)
            ax.annotate(policy, (resource["mean"], accuracy["mean"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set(title=f"{xlabel} vs worst-class accuracy ($\\rho={rho:g}$)", xlabel=xlabel, ylabel="Worst-class accuracy (%)")
        ax.grid(True, linestyle=":", alpha=0.5)
    axes[0].legend(title="Policy")
    resource_path = os.path.join(out_dir, "final_resource_vs_worst_class.png")
    fig.tight_layout()
    fig.savefig(resource_path, dpi=180)
    plt.close(fig)

    paths = [participation_path, eligibility_path, class_mass_path, forest_path, resource_path]
    return paths


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
    with open(res_path, "r", encoding="utf-8") as result_file:
        result_data = json.load(result_file)
    if "results" in result_data:
        generate_study_plots(result_data, target_dir)
    else:
        generate_all_plots(res_path, target_dir)
