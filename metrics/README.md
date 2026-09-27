# A-MFO-Net rollout metrics

These CSVs were copied unchanged from the completed matched A-MFO-Net runs used
in the submitted manuscript. The files correspond to seeds 42, 43, and 44,
respectively:

- `amfo_seed42_rollout_metrics_per_step.csv`
- `amfo_seed43_rollout_metrics_per_step.csv`
- `amfo_seed44_rollout_metrics_per_step.csv`

Each CSV has one row for forecast steps 1–400 on the held-out extended-room
data-center case (90% ACU cooling, 100% server workload; 0.05 seconds per step).
The training protocol was 100 epochs, 200 selected three-step windows per
epoch, a 14/2/1 train/validation/test case split, and the complete 22,000-cell
mesh at evaluation. These files belong to the earlier completed matched
campaign, not the later full-window campaign.

`metric_scope=all` means all 22,000 mesh cells contribute to each reported
metric. Recorded ground-truth values replace predictions outside the flow mask
before each next-step input and before scoring. Relative-L2 columns are
fractions; multiply by 100 for percentages. `vel_vec_rel_l2` combines the
three velocity components; `pressure_rel_l2` and
`temperature_rel_l2` are scalar-field metrics. RMSE columns retain the units
of their corresponding physical fields. The two timing columns are measured
per forecast step on the original evaluation host, not training times.

At step 400, the source CSVs give the following percentages:

| Seed | Velocity | Pressure | Temperature |
| --- | ---: | ---: | ---: |
| 42 | 6.92910 | 27.57669 | 5.22368 |
| 43 | 7.16658 | 36.81353 | 4.93932 |
| 44 | 6.81027 | 30.27976 | 4.99270 |

Their three-seed mean and sample standard deviation, rounded to two decimal
places, are 6.97 ± 0.18% velocity, 31.56 ± 4.75% pressure, and
5.05 ± 0.15% temperature, as in the manuscript. These derived files contain
aggregate error and timing measurements only; they contain no CFD state arrays,
mesh, prediction trajectories, or model weights.
