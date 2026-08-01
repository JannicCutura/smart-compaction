# Referee Report

**Conference:** The 4th International Conference on Intelligent Computing, Communication, Networking and Services (ICCNS 2026)

**Paper:** "Predicting Compaction Utility from Lakehouse Table Metadata"

**Recommendation:** Minor Revision (Borderline Accept)

---

## Summary

The paper studies whether bin-packing compaction utility in Apache Iceberg tables can be predicted from metadata features alone, without reading data files. The authors construct a simulation framework generating 2,376 synthetic Iceberg tables across a 7-axis parameter grid, extract 17 metadata features, and train XGBoost models for both binary classification (does the table need compaction?) and regression (what file-reduction ratio will compaction achieve?). The central finding is a task separation: the binary decision is trivially solved by a single threshold (`max_files_per_partition > 4`, 100% accuracy), while predicting the continuous reduction ratio requires nonlinear modelling (XGBoost R² = 0.998 vs. OLS R² = 0.361). Cross-schema validation on 96 TPC-H tables shows transfer without retraining (97.9% accuracy, R² = 0.976).

---

## Strengths

1. **Well-defined and practical problem.** The paper addresses a genuine gap: no prior work has systematically studied which metadata features predict compaction outcomes. The motivation from AutoComp (SIGMOD 2025) and the connection to production systems (Databricks, Snowflake, AWS) is well-articulated.

2. **Intellectual honesty about the "tautology trap."** The authors candidly report that an earlier version of their grid produced trivially perfect results because all files were small. This kind of negative-result transparency is rare and valuable.

3. **Reproducibility.** Public code and data, a clearly documented parameter grid, and standard benchmarks (TPC-H) make full replication feasible. This is commendable.

4. **Clean experimental design.** The separation into classification and regression tasks, the threshold sweep analysis, and the per-schema breakdown in TPC-H validation are methodologically sound.

5. **The central insight is genuinely useful.** The finding that partition-level file counts dominate compaction decisions — and that ML adds no value over a trivial threshold for the binary task — is an honest, practically important result that could save practitioners from over-engineering.

---

## Weaknesses

### Major

**M1. The core ML contribution is underwhelming.** The paper's own results demonstrate that the binary classification task is trivially solved by a hand-crafted threshold rule (`max_files_per_partition > 4`). The XGBoost classifier provides zero additional value. This means the ML contribution rests entirely on the regression task, where the practical payoff (cost-aware scheduling) is discussed speculatively but never demonstrated. The paper essentially proves that ML is unnecessary for the primary decision it set out to automate.

**M2. The simulation is too narrow to support the paper's claims.** The entire experimental apparatus uses a single compaction configuration (target 128 MB, min 96 MB, max 230 MB) and a single compaction strategy (bin-packing). The authors acknowledge this in Section III-E4 ("Compaction Target Sensitivity") but do not test any alternatives. The `k = 4` threshold is an artifact of these specific parameters. A paper titled "Predicting Compaction Utility" that only validates one compaction configuration significantly overstates its generality.

**M3. No workload-aware evaluation.** The paper predicts *structural* compaction outcomes (file count reduction) but makes no connection to actual query performance improvements. A table may benefit from compaction structurally but see no query speedup. Without at least one experiment linking predicted compaction utility to downstream query latency (even on TPC-H queries), the practical value of this prediction remains unsubstantiated. The paper acknowledges this gap but treats it entirely as future work.

**M4. 89.3% positive class prevalence.** The synthetic dataset has 89.3% positive labels. This extreme imbalance means that a majority-class baseline already achieves 89.3% accuracy. While the authors report this baseline (good practice), the experimental design raises the question of whether the grid was inadvertently biased toward compaction-positive tables. A more balanced design — or at least stratified evaluation at multiple prevalence levels — would strengthen the findings.

**M5. Cross-schema validation is limited.** The TPC-H validation, while a step in the right direction, uses the same simulation framework with the same compaction parameters, just different schemas. This validates schema transfer but not distribution transfer. The claim of "generalization" should be tempered: the model has never been tested on production tables with schema evolution, delete files, mixed CoW/MoR strategies, or real-world write patterns. The two false negatives on TPC-H (customer, partsupp) are borderline cases that the paper explains away rather than investigates.

### Minor

**m1. Related work is thorough but disproportionate.** The LSM-tree compaction literature review (Section I-A) is extensive for a paper that explicitly argues lakehouse compaction is fundamentally different from LSM compaction. Nearly a full column is spent on work that the authors then dismiss as not directly applicable. This space would be better used for additional experiments or deeper analysis.

**m2. Feature engineering is under-analyzed.** 17 features are extracted but only 2 matter. While the paper reports this correctly, it does not explore *why* the other 15 features are irrelevant. Are they redundant (correlated with the top 2)? Would the model degrade if only the top 2 were used? A feature ablation study is missing.

**m3. No hyperparameter sensitivity analysis.** XGBoost defaults are presumably used (the paper does not specify hyperparameters). Given that the model achieves near-perfect scores, it would be informative to know whether a simpler tree (depth-2, 10 trees) achieves comparable results, which would further support the "this problem is easy" narrative.

**m4. Statistical rigor.** Results are reported as single train/test split metrics. No confidence intervals, no repeated random splits, no bootstrap estimates. With n = 476 test samples, variance in accuracy and F1 is non-trivial. The R² = 0.998 figure looks impressive but could benefit from a confidence interval.

**m5. Writing quality.** Generally clear and well-organized, but the paper would benefit from tightening. Some redundancy between the abstract, introduction contributions, and conclusion. The phrase "tautology trap" is used without formal definition in the abstract.

**m6. Missing comparison with AutoComp's heuristics.** The paper repeatedly positions itself against AutoComp but never implements or benchmarks against AutoComp's actual compaction trait computation. Even an approximate reimplementation would strengthen the comparative analysis.

**m7. The Zipfian skew parameter is fixed at s=1.** Only one skew profile is tested. Real-world partition skew varies widely; a sweep over skew parameters would strengthen the generalization claim.

---

## Questions for the Authors

1. Have you tested what `k` value achieves perfect classification under a different compaction target (e.g., 512 MB or 1 GB)? If `k` shifts with the target, the practical utility of the threshold rule collapses.

2. What is the correlation matrix among the top-5 features? If `max_files_per_partition` and `avg_files_per_partition` are highly correlated, the 95.2% combined importance may be misleading.

3. Can you provide wall-clock times for the feature extraction step? The claim that metadata extraction is "fast and non-intrusive" is unsupported by measurements.

4. The paper states 5-fold cross-validation was used but only reports held-out test metrics. Can you report the cross-validation variance?

---

## Detailed Comments

- **Eq. (1):** The file reduction ratio r_fc is a reasonable metric but ignores file *size* reduction. A compaction that merges 10 files of 1 KB each into one file has r_fc = 0.9 but negligible practical impact. Total bytes rewritten would be a more informative metric.

- **Table I (Parameter Grid):** The `file_size_target_kb` axis has 11 values but only a few span the compaction threshold range. How many of the 2,376 tables actually produce files in the interesting 50–200 MB range?

- **Section III-D (Threshold Baseline):** The sharp drop at k=5 is interesting but the explanation is tied to the specific compaction configuration. This should be stated more prominently as a limitation rather than a finding.

- **Table III (TPC-H results):** Reporting "---" for RMSE and R² in classification rows and vice versa for regression rows is standard, but consider adding classification metrics for the regression model (using a binarized prediction at r_fc > 0) to show consistency.

- **The paper does not discuss computational cost of training vs. the threshold rule.** If the threshold rule achieves identical binary accuracy and the regression model is only needed for cost-aware scheduling (which is not demonstrated), what is the practical argument for deploying a trained model?

---

## Minor Technical Issues

- Section II-B2 mentions "even-indexed batches" use 8x smaller file targets, but the indexing convention (0-based or 1-based) is unspecified.
- The Spearman ρ = 0.965 between avg_files_per_partition and r_fc is high but should be compared against ρ for max_files_per_partition to justify the feature importance ranking.
- Reference [6] (AutoComp) is a SIGMOD companion paper, not a full research paper. This distinction matters when positioning it as the primary related work.

---

## Overall Assessment

This is a competent empirical study that asks the right question and provides an honest answer: the binary compaction decision in Iceberg is trivially predictable from a single metadata feature, while the continuous prediction task genuinely benefits from ML. The reproducibility and the "tautology trap" insight are genuine contributions.

However, the paper suffers from a tension between its ambition and its results. It frames itself as an ML-for-systems contribution but then demonstrates that ML is unnecessary for the primary task. The regression contribution, where ML *does* add value, is not connected to any downstream application. The single compaction configuration, absence of workload-aware evaluation, and lack of production validation leave the generalizability claims weakly supported.

The paper is publishable with revisions that (a) test at least one additional compaction target size to show whether the threshold shifts, (b) add a feature ablation study, (c) provide confidence intervals for key metrics, and (d) either demonstrate a downstream use of the regression predictions or reframe the contribution more modestly.

**Score: 5/10 — Borderline. Interesting problem, honest results, but the contribution is thinner than the framing suggests.**
