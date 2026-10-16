# Availability-Aware Federated Learning Simulation

This repository contains a synthetic experiment on EV client availability and selection bias. It compares random selection, random over-selection, guardrails, participation-aware aggregation, heavy-use fallback, and connectable peer-pair baselines while varying the association between vehicle usage profile and MNIST labels. Results describe this simulation under the assumptions below; they are not evidence about deployed vehicles or real-world federated learning systems.

## Research Question

When vehicle availability is correlated with label distribution, do guardrails underweight classes 5-9 because heavy-use vehicles are less often eligible? Can staleness-aware selection or inverse-participation aggregation reduce that class-mass gap, and what resource cost does it impose?

The base A-D/P2P experiment changes data-state coupling with $\rho \in \{0, 0.25, 0.5, 0.75, 1\}$. E1/E2/F/P2P-2 run at $\rho \in \{0, 0.5, 1\}$:

- $\rho=0$: class allocation is independent of heavy-use/light-use profile.
- $\rho=0.5$: half of each class allocation follows the profile tilt.
- $\rho=1$: class allocation follows the profile tilt, favoring classes 5-9 for heavy-use vehicles and classes 0-4 for light-use vehicles.

For each class, profile weights are 4.5 for a favored profile and 0.5 for the other profile, then normalized across clients. At $\rho=0$, an independent Dirichlet allocation with $\alpha=0.8$ is used.

## Registered Hypotheses

The primary family is frozen at $\rho=1$ and paired by seed against the stated baseline:

- **H1:** C has lower worst-class and classes 5-9 accuracy than A.
- **H2:** E1 and F each improve both accuracy metrics versus C. E1 and F are realistic remedies; E2 is an oracle comparison.
- **H3:** B uses more metered MB and modeled kWh than C.
- **H4:** C and D have more vehicles below five participation rounds than A.

One-sided paired t-tests are combined as intersection-union tests within each composite hypothesis. Holm step-down correction is applied across H1-H4. Paired bootstrap percentile intervals use 10,000 seed resamples. All other policy, rho, subgroup, sensitivity, and P2P comparisons are exploratory and are not included in this correction family.

## Policies

- **A, Random:** Sample 5 of 20 vehicles. Aggregate successful updates. If no client survives, retain the prior global model and accuracy.
- **B, Random over-selection:** Estimate survival probability from current telemetry and select enough random vehicles to target 5 expected survivors. This is a simplified dropout-tolerant baseline, not an implementation or validation of a P2P aggregation protocol.
- **C, Guardrail:** Select up to 5 vehicles that meet all power, network, and bandwidth guardrails. There is no fallback to ineligible clients.
- **D, Guardrail + staleness:** Keep C's eligibility rules, then sample without replacement using weight $1 + 2s$, where $s$ is rounds since last selected. This is a participation fairness heuristic, not a guarantee that every vehicle participates.
- **E1, Inverse-participation weighted:** Use C's selected clients, aggregating with weight $n_i/p_i$, where $n_i$ is local sample count and $p_i$ is cumulative selections for that client. This is the realistic participation-aware aggregation variant.
- **E2, Oracle group-balanced:** Use E1 weighting, then rescale successful heavy/light group weights to exactly 50/50. It requires known profile membership and is an oracle comparison, not a deployable policy.
- **F, Heavy-use battery fallback:** Use C's eligibility rules. If fewer than two heavy-use clients are eligible, also allow heavy-use vehicles on parked or charging unmetered Wi-Fi at or above the configured fallback battery threshold (30% by default).
- **P2P, Connectable peer pairs:** Select one pair of vehicles currently parked with unmetered Wi-Fi. Each trains from the current global model; the pair sample-weight merges updates, and the resulting pair update is used for the next simulated round. This is a minimal pair-connectivity baseline, not a decentralized protocol implementation.
- **P2P-2:** The same pair merge using two connectable pairs (four clients) per round.

Each successful centralized round uses sample-count-weighted FedAvg. P2P merges each pair before applying the resulting update. Failed selected clients are counted as wasted client-rounds and incur partial modeled compute and download costs. Each seed/policy run uses the same profile assignment, deterministic data partition and test subset, exogenous Markov-state trace, and pre-generated per-vehicle/per-round dropout uniforms; a trace hash check fails the study if paired availability paths diverge. Policy selection and dropout draws use separate random streams.

## Assumptions

These values are simulation assumptions, not measurements of a vehicle fleet. The executable values are in `config.yaml`; the implementation is in `src/telemetry.py`, `src/orchestrator.py`, `src/model.py`, and `study.py`.

| Quantity | Assumption |
|---|---|
| Fleet and training | 20 vehicles; 10 heavy-use and 10 light-use; 75 rounds; 5 requested clients; 1 local epoch; at most 256 training examples per client; 2,000 MNIST test examples |
| Heavy-use initial state order | Driving, charging on Wi-Fi, charging on metered cellular, parked on Wi-Fi: `[0.70, 0.10, 0.05, 0.15]` |
| Light-use initial state order | `[0.15, 0.40, 0.10, 0.35]` |
| Heavy-use transition rows | Driving `[0.75, 0.10, 0.05, 0.10]`; charging Wi-Fi `[0.35, 0.50, 0.05, 0.10]`; charging metered `[0.50, 0.10, 0.30, 0.10]`; parked Wi-Fi `[0.45, 0.10, 0.05, 0.40]` |
| Light-use transition rows | Driving `[0.45, 0.30, 0.05, 0.20]`; charging Wi-Fi `[0.08, 0.68, 0.04, 0.20]`; charging metered `[0.20, 0.25, 0.40, 0.15]`; parked Wi-Fi `[0.08, 0.38, 0.04, 0.50]` |
| Power guardrails | Non-driving charging requires battery $\ge 50\%$; DC fast charging requires $\ge 20\%$. Driving is ineligible. |
| Network guardrail | Unmetered Wi-Fi only, bandwidth $\ge 10$ Mbps |
| Mid-round dropout | Driving: 50%; battery below 25%: 60%; metered 4G or bandwidth below 8 Mbps: 45%. Checks are sequential; a client drops at the first failed check. |
| Energy and transfer | Battery capacity 75 kWh; compute 0.025 kWh per local epoch (40% of this for failed clients); 5 W transmission power; model-size downlink for failures and downlink plus uplink for successes |
| Over-selection sizing | Target 5 expected survivors; per-client survival probability is the product of the relevant pass probabilities; 0.05 floor for the mean survival estimate |
| Learning and partitioning | EdgeCNN, SGD learning rate 0.02, Dirichlet $\alpha=0.8$, batch size 128 |
| Evaluation | Checkpoint every 5 rounds; always evaluate each of the final 10 rounds. Report final accuracy as the average over those 10 rounds. |
| Primary seeds | 42-61 (20 paired seeds); sensitivity seeds 42-51 (10 seeds) |
| Inference | Paired t intervals and paired bootstrap percentile intervals (10,000 resamples); Holm correction across H1-H4 only |
| Policy D staleness bonus | 2.0 per round since last participation |
| P2P capacity | One pair per round; only parked + unmetered Wi-Fi states are connectable |

The Markov states are simplified: parked Wi-Fi is represented as `DISCHARGING` for charging-state telemetry, and charging on metered cellular is represented as DC fast charging. The dropout simulator uses the explicit availability state to avoid treating parked vehicles as driving. These mappings and uniform battery/throughput ranges are synthetic. The energy metric includes only modeled training and transmission energy, not vehicle traction, charger energy, or infrastructure overhead. P2P models a pairwise rendezvous and merge but does not simulate transport latency, topology, multi-hop forwarding, or concurrent consensus.

## Metrics and Sensitivity

Headline metrics are worst-class accuracy, classes 5-9 accuracy, and the heavy-minus-light accuracy gap, averaged over the final 10 evaluated rounds. Secondary metrics include mean per-class accuracy. Reports include rounds-to-90 reach fraction and median among successful seeds, Jain participation index, minimum participation, percentage of runs with any vehicle below five rounds, worst-class identity/group frequency, and classes 5-9 share of successful training-example mass.

Every per-run JSON records the randomized profile map and trace hash, eligible-pool counts by profile per round, selected client IDs, selected and successful training-example mass by class, per-vehicle participation counts, and all 10 class accuracies at each evaluated round. Runs are atomic, independently saved, and skipped on resume only when their fingerprint matches the current design.

At $\rho=1$, ten-seed sensitivity runs compare A/C/E1/F while changing heavy-use share (0.25, 0.5, 0.75), driving-dropout probability (0.25, 0.5, 0.75), requested clients (3, 5, 8), and F's fallback battery threshold (20%, 30%, 40%). DC-fast battery and Wi-Fi bandwidth threshold sweeps are omitted because those thresholds are inactive under the current telemetry-state mapping.

The final workflow creates per-seed participation by profile, eligible-pool size by profile over rounds, classes 5-9 mass share by policy, paired-difference forest plots with t and bootstrap intervals, and policy-mean resource versus worst-class plots with error bars. Every plot title includes $\rho$. The earlier per-vehicle-ID plots mixed profile identities across seeds and have been replaced by within-seed profile summaries.

### Final Run Status

The prior A-D/P2P aggregate is historical, not the final preregistered experiment. On 2026-10-16, the required two-seed, ten-round pilot completed for all nine policies at $\rho=1$. It passed the output-schema, profile/trace pairing, deterministic rerun, plotting, and 1,000-bonus staleness gates. The pilot report is `results/pilot/availability_study_final.json`; its five figures are under `results/pilot/figures/`. Pilot accuracy is not final evidence because it has only two seeds and ten rounds.

The full run is currently **NO-GO** until the frozen `config.yaml` is committed. Git was found at `C:\Users\1000684\AppData\Local\Programs\Git\cmd\git.exe`, but no config commit has been created yet. Full mode fails closed until Git confirms that the exact config bytes match committed `HEAD`. The pilot output includes the config SHA-256 and reports that it predates this commit. Do not interpret the old aggregate as results for H1-H4.

## Run

Install dependencies with `pip install -r requirements.txt`, then run:

```bash
python run_benchmark.py --mode pilot
```

The pilot is the required pre-run gate. Full execution is intentionally refused until `config.yaml` is committed and matches Git HEAD. Commit the frozen config, rerun and pass the pilot against that commit, then run `python run_benchmark.py --mode full`. The full workflow writes resumable, atomic one-file-per-run JSON under `results/runs/`, skips exact completed fingerprints, parallelizes seed bundles, and records config SHA-256 plus Git commit in every output.

The planned primary design is A-D plus P2P at all five $\rho$ values and E1/E2/F/P2P-2 at $\rho \in \{0, 0.5, 1\}$, across 20 seeds. Ten-seed sensitivity runs use A/C/E1/F. MNIST downloads on first use. Use `--workers N` to set process parallelism; Windows runs disable system sleep while executing.

## Scope and Follow-up

This is a synthetic CPU simulation with 20 vehicles, capped MNIST training data, assumed transition/dropout probabilities, and simplified charging/connectivity mappings. It does not model differential privacy, heterogeneous hardware, real vehicle traces, or deployment costs. E2 is an oracle because it uses profile-group knowledge to force equal group contribution. Interpret results as outcomes under the listed assumptions, not as validation of a general selection or aggregation approach.

A later extension could connect availability-driven participation to client-level cumulative differential privacy accounting. That accounting is not implemented here, and standard random-sampling assumptions should not be assumed to hold for availability-filtered participation.
