# Referee Report

**Venue:** The 4th International Conference on Intelligent Computing, Communication, Networking and Services (ICCNS 2026)

**Paper:** *Predicting Compaction Utility from Lakehouse Table Metadata*

**Recommendation:** Major Revision

---

## Summary

The paper presents an ML-based approach to predict whether Apache Iceberg tables benefit from bin-packing compaction, using 17 metadata features extracted from manifest files. A simulation framework generates 2,376 Iceberg tables across a parameter grid, runs compaction, and trains XGBoost classifiers/regressors. The central finding is that the binary compaction decision reduces to a single threshold rule (`max_files_per_partition > 4`), while predicting the continuous file-reduction ratio requires nonlinear modelling (XGBoost R^2 = 0.998). Cross-schema validation on 96 TPC-H tables and a query benchmark complement the evaluation.

---

## Strengths

1. **Well-defined, practical problem.** The small-file problem in lakehouses is real and costly. Framing compaction scheduling as a prediction task is sensible and timely, particularly given AutoComp's explicit identification of ML-based scheduling as future work.

2. **Reproducibility.** The open-source framework, parameter grid, and dataset are commendable. The end-to-end pipeline---from generation through compaction through evaluation---is well-documented and appears fully reproducible. This is above the standard for the venue.

3. **Honest and thorough self-criticism.** The paper identifies the tautology trap, discusses the reduced-parallelism phenomenon, acknowledges synthetic-data limitations, and enumerates specific threats to validity (MoR, sort/z-order, temporal evolution, compaction target sensitivity). This intellectual honesty is appreciated but does not eliminate the concerns raised below.

4. **Interesting negative result on query performance.** The finding that bin-packing compaction degrades full-scan aggregations by 45--55% due to reduced Spark task parallelism is genuinely useful and counter-intuitive. This is arguably the paper's most practically relevant contribution.

5. **Strong ablation and robustness analysis.** The progressive top-k, leave-one-out, hyperparameter sweep, model family comparison, prevalence downsampling, and bootstrap CIs are thorough. Few papers at this level test robustness this carefully.

---

## Weaknesses

### Major Issues

**W1. The classification task is trivially solvable and not a meaningful ML problem.**

The classifier achieves perfect accuracy (F1 = 1.000, AUC = 1.000). The paper acknowledges this and shows the k=4 threshold matches XGBoost exactly. But the implications are not confronted head-on: the binary compaction decision is a *deterministic function* of whether the bin-packing filter finds files below the min-file-size threshold (96 MB). Training XGBoost, logistic regression, and performing feature importance analysis on a task that is solvable by a single SQL predicate (`SELECT MAX(cnt) FROM (SELECT COUNT(*) cnt FROM files GROUP BY partition) > 4`) does not constitute a machine learning contribution. Approximately half the paper (Sections III-B, III-D, III-F, parts of III-E) is devoted to analysing a trivially separable classification task. This space would be better allocated to the regression task, which is where ML genuinely adds value---as the authors themselves concede in Section III-D.

**W2. Circularity in the label definition undermines the research question.**

`needs_compaction` is defined as `rewritten_data_files_count > 0`, i.e., "did Iceberg's compactor rewrite anything?" This is a mechanical outcome, not a measure of *utility*. The paper's title promises "Predicting Compaction *Utility*," but the label captures compaction *activity*. A table where one 50 KB file is merged into an existing 95 MB file receives `needs_compaction = 1` despite negligible practical benefit. A more meaningful formulation would incorporate a benefit threshold (e.g., r_fc > epsilon for some non-trivial epsilon, or a composite metric including downstream query speedup).

Additionally, the feature `small_file_ratio` (fraction of files below 96 MB) directly encodes knowledge of how the compaction filter decides which files to rewrite, since both use the identical 96 MB threshold. This constitutes a form of target leakage: the feature is a near-deterministic indicator of the label. Although the feature turns out not to be the top predictor (that role goes to `max_files_per_partition`), its presence in the feature set is methodologically problematic and should be discussed explicitly.

**W3. The simulation is too distant from production to support the paper's generalisability claims.**

- **All-UUID payloads.** Synthetic tables use only `uuid()` string columns. Compression behaviour, on-disk file sizes, and per-row byte counts differ substantially from production tables with mixed types, nulls, dictionary-encoded low-cardinality columns, and realistic value distributions. The TPC-H validation partially addresses this, but the primary model is trained exclusively on UUID data.
- **No deletes, updates, or schema evolution.** Production Iceberg tables accumulate positional and equality delete files. Merge-on-Read compaction dynamics differ fundamentally from the Copy-on-Write bin-packing evaluated here. The paper acknowledges this (Section III-H) but the limitation is severe: the model's feature set is structurally incomplete for MoR tables, which are increasingly common in production.
- **Single-node local-mode Spark.** All experiments run in `local[*]` mode. File layout, parallelism, and I/O behaviour differ from distributed clusters with S3/HDFS. The query benchmark results (Section III-G) are particularly suspect given local-mode execution, where there are no network I/O costs, no object-store LIST latency, and no distributed task scheduling overhead---precisely the factors that make the small-file problem costly in production.
- **Identical write mechanics across "validation" sets.** The TPC-H tables use the same batch-heterogeneity pattern (50% first batch, alternating file-size targets). Cross-schema "generalisation" therefore tests robustness to column types, not to genuinely different write patterns or operational workflows. This weakens the generalisation claim.

**W4. The regression task, while more interesting, is under-developed.**

The paper's strongest ML contribution---predicting the continuous reduction ratio---receives disproportionately less attention than the trivial classification task. Key gaps:

- No analysis of *when* the regression prediction would change an operational decision. A production scheduler needs a cost-benefit threshold, not a raw r_fc prediction. What reduction ratio justifies the compaction I/O cost? Without this, the regression model has no actionable output.
- The regression target r_fc is distribution-shifted: mean 0.818, median 0.950, heavy left tail. The reported RMSE of 0.013 may mask significant relative errors at the low end of the distribution (e.g., predicting 0.15 vs. 0.05 matters more for scheduling than predicting 0.95 vs. 0.97). Per-quantile error analysis is absent.
- Given that `max_files_per_partition` + `avg_files_per_partition` account for 95.2% of regressor gain, the paper should formally quantify the marginal value of the remaining 15 features with confidence intervals on the R^2 difference. The current top-k ablation (Figure 5) is suggestive but lacks statistical testing.

---

### Minor Issues

**W5.** The 89.3% positive class prevalence makes accuracy a misleading metric. While the prevalence sensitivity experiment (Section III-F) partially addresses this, the main results tables (Tables II, III) should report balanced accuracy or Matthews Correlation Coefficient in addition to accuracy/F1.

**W6.** The related work on LSM-tree compaction (Section I-A) is disproportionately long given that the paper explicitly argues LSM-trees are *not* directly applicable to flat lakehouse files. This section could be condensed to a single paragraph, freeing space for deeper analysis of the regression task or cost modelling.

**W7.** Figure 1 (ridge plot of features) is difficult to read at conference column width. Many distributions are bimodal or heavy-tailed but the kernel density estimates obscure structure. Consider box plots or ECDF curves, or provide descriptive statistics in a supplementary table.

**W8.** The query benchmark (Section III-G) uses 1 warmup and 3 timed iterations (the text says 3; the source code appears to use 5---please clarify). For local-mode Spark with JVM warm-up, page cache effects, and garbage collection, this is insufficient for stable latency measurements. Standard practice is >= 5 warmup runs and >= 20 timed iterations, or at minimum reporting confidence intervals on speedup ratios.

**W9.** The k=4 threshold is mechanically determined by the compaction configuration (target 128 MB, min 96 MB). The paper acknowledges this dependency (Section III-H, "Compaction Target Sensitivity") but does not test it. Running even one alternative configuration (e.g., Iceberg's production default of 512 MB target) would substantially strengthen the methodology.

**W10.** `num_snapshots` as a feature is confounded with a generation parameter: in the simulation, it equals `num_write_batches + 1` (accounting for the compaction snapshot). In production, snapshots accumulate heterogeneously. This feature is effectively a proxy for a generation parameter, not a genuine, independently observable metadata signal.

---

## Questions for the Authors

1. Given that the classification task reduces to a deterministic threshold, what is the ML contribution beyond confirming that XGBoost can learn a step function? Would the paper be stronger if the classification analysis were condensed to a single paragraph and the focus shifted entirely to regression and cost-aware scheduling?

2. Have you tested whether the k=4 threshold shifts when compaction target size changes (e.g., to Iceberg's 512 MB production default)? If so, does the XGBoost model generalise where the fixed threshold does not?

3. The query benchmark shows compaction *hurts* full-scan aggregations. Have you considered incorporating query-type information into the compaction decision model? This seems like a more impactful and novel research direction than refining binary classification of a trivially separable problem.

4. Can you provide per-quantile regression error analysis, particularly for tables with r_fc in [0.0, 0.5] where scheduling decisions are most ambiguous and operationally consequential?

5. The abstract states "100% accuracy on both synthetic and TPC-H data." TPC-H accuracy is 97.9%. This appears to conflate the threshold rule's performance on synthetic data with overall cross-schema performance. Please correct.

---

## Detailed Comments

- **Abstract:** "100% accuracy on both synthetic and TPC-H data" is imprecise---TPC-H accuracy is 97.9%. The 100% refers to the threshold rule on synthetic test data only. Rephrase.
- **Section II-A, Eq. 1:** The reduction ratio r_fc penalises tables with few files disproportionately (e.g., 2 files becoming 1 yields r_fc = 0.5, which is practically negligible). Discuss whether this metric appropriately captures "utility."
- **Section III-B:** "Logistic regression falls below the majority-class baseline" --- this is an expected consequence of `class_weight="balanced"` and is well-understood. The explanatory paragraph can be deleted without loss.
- **Table V(b):** The per-schema breakdown shows customer and partsupp at R^2 ~ 0.87. With only n=16 per schema, these point estimates have wide confidence intervals that should be reported or at least acknowledged.
- **Section III-H, "Tautology Trap":** This reads as post-hoc justification. If the initial all-small-file grid was an earlier experimental iteration, say so explicitly and frame it as a design decision rather than a research finding.
- **Conclusion:** "the binary compaction decision is trivially solvable by a single partition-level threshold" --- the paper could be more forthcoming that this finding undermines the ML framing for approximately half of the experimental evaluation.

---

## Summary Assessment

The paper tackles a relevant applied problem with commendable reproducibility and thorough robustness analysis. The query-performance analysis and the tautology-trap discussion are genuinely valuable contributions to the practitioner community. However, approximately half the paper analyses a trivially solvable classification task that does not require machine learning, the label definition conflates compaction *activity* with compaction *utility*, and the simulation's distance from production (no deletes, no MoR, no schema evolution, UUID-only payloads, local-mode Spark) limits the generalisability of results. The more interesting regression contribution is under-developed and lacks the cost-benefit framing needed for operational relevance.

A major revision should: (1) substantially condense the classification analysis, acknowledging its trivial separability upfront rather than building to it; (2) redefine "utility" to incorporate a meaningful cost-benefit threshold rather than a mechanical label; (3) expand the regression analysis with per-quantile error breakdown and operational decision thresholds; (4) test at least one alternative compaction target size to validate the methodology's portability; and (5) either simulate MoR/delete-file scenarios or more precisely scope the title and claims to "Copy-on-Write bin-packing utility."

The engineering quality and reproducibility of the framework are publication-worthy. The research contribution needs sharpening.
