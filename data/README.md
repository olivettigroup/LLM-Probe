# Model-level results

One CSV per task. Each row is one model in one condition (`zero-shot` or `fine-tuned`). These are the values reported in the SI tables.

| File | Test set | Main metric |
|---|---|---|
| `bandgap_models.csv` | 3,415 compounds | RMSE (eV) |
| `dielectric_models.csv` | 775 formula-disjoint compounds | RMSE |
| `crystal_system_models.csv` | 6,730 compounds | accuracy (%) |
| `matkg_models.csv` | 2,355 subject–relation queries | Top-1 / Top-5 (%) |

## Columns

- `runs`: number of repeated runs per item.
- `valid_pct`: share of all run × item answers that are valid. `below1_pct` / `missing_pct` (regression): answers below 1 / missing.
- `n_scored`: items scored (an item is scored if ≥ 60% of its runs are valid). `pct_cpd_all_runs` / `coverage_pct`: share of items scored.
- `rmse_mode`, `acc`, `top1`: metric on the most common answer per item. `rmse_ci`, `acc_lo`–`acc_hi`, `top1_lo`–`top1_hi`: 95% CI from 20,000 bootstrap resamples of test items.
- `rmse_mean`, `rmse_median`: RMSE using the mean or median answer; `*_minus_mode_ci`: 95% CI of the difference from the mode.
- `*_common`, `n_common`, `n_common_covered`: metric on the items scored by every retained model of that task.
- `tie_rate_pct`: items whose most common answer is tied.
- `H_exact`, `H_bin01` (bandgap, 0.1 eV bins), `H_bin05` (dielectric, bins of 0.5), `H_mean` (crystal, MatKG): response entropy in bits, averaged over items.
- `median_sd` / `mean_sd`: median / mean over items of the per-item standard deviation across runs. `pct_identical`: items with the same answer in every run.
- `balanced_acc`, `macro_f1`, `beats_majority` (crystal): additional classification metrics.
- `note`: reason a model is not scored. `status`: Qwen2 rows are listed for completeness and excluded from the analysis.

Blank cells: the metric was not computed for that model.
