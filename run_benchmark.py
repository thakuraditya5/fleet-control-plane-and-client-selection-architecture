"""Run the availability, data-coupling, and client-selection study."""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import ctypes
import json
import yaml

from study import run_final_workflow


def main():
    parser = argparse.ArgumentParser(description="Run gated federated-learning experiments")
    parser.add_argument("--mode", choices=("pilot", "full"), default="full")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"{config_path} not found")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("Registered primary hypotheses at rho=1:")
    for hypothesis, details in config["study"]["primary_hypotheses"].items():
        print(f"{hypothesis}: {details['statement']}")
    print("Other comparisons are exploratory; Holm correction is applied across H1-H4.")
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    try:
        report = run_final_workflow(config, config_path=config_path, mode=args.mode, workers=args.workers)
        from plot_results import generate_final_plots

        output_dir = os.path.join(config["benchmarks"]["output_dir"], "pilot") if args.mode == "pilot" else config["benchmarks"]["output_dir"]
        plot_paths = generate_final_plots(report, os.path.join(output_dir, "figures"))
        missing_plots = [path for path in plot_paths if not os.path.exists(path) or os.path.getsize(path) == 0]
        if missing_plots:
            raise RuntimeError(f"GO/NO-GO: expected plots missing or empty: {missing_plots}")
        if args.mode == "pilot":
            print("Pilot gates:")
            print(f"  output schema/pairing: {report['run_schema_validation']['passed']}")
            print(f"  deterministic rerun: {report['determinism_check']['passed']}")
            print(f"  profile maps randomized across seeds: {report['profile_map_validation']['distinct_profile_maps_across_seeds']}")
            print(f"  staleness bonus gate: {report['staleness_gate']['passed']}")
            print(f"  figures verified: {len(plot_paths)}")
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()
