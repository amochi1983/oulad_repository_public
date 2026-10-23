# Numerical verification report — ORIGINAL paper submission copy

Executed2026-10-18 in separate repository. Raw-to-core command exited0 after 317.75seconds. All raw local version1 hashes checked; features/cohorts rebuilt from the CSVs, primary/six nonprimary association specifications refitted, focused exports and unknown-date/module/omission/link-scale checks rerun. The manuscript B/C/D tuning, refit, calibration, scoring and original200-draw bootstrap intervals were freshly reproduced. This is bounded reproduction, not a new experiment or full wider tree/window search.

| table                                                                |   rows |   max_absolute_numeric_difference | status   |
|:---------------------------------------------------------------------|-------:|----------------------------------:|:---------|
| outputs/first_project/tables/S1_group_descriptives.csv               |      4 |                                 0 | PASS     |
| outputs/first_project/tables/S2_observed_group_gaps.csv              |      4 |                                 0 | PASS     |
| outputs/first_project/tables/S3_adjusted_engagement_associations.csv |     12 |                                 0 | PASS     |
| outputs/first_project/tables/S4_engagement_OR_comparisons.csv        |      9 |                                 0 | PASS     |
| outputs/first_project/tables/S5_model_information.csv                |      3 |                                 0 | PASS     |
| outputs/first_project/tables/S6_adjusted_success_probabilities.csv   |      9 |                                 0 | PASS     |
| outputs/first_project/tables/S7_adjusted_success_gaps.csv            |      6 |                                 0 | PASS     |
| outputs/first_project/tables/S8_sensitivity_interactions.csv         |      7 |                                 0 | PASS     |
| outputs/first_project/tables/S9_original_IMD_descriptives.csv        |     11 |                                 0 | PASS     |
| outputs/manuscript_checks/module_OR.csv                              |     21 |                                 0 | PASS     |
| outputs/manuscript_checks/module_interactions.csv                    |      7 |                                 0 | PASS     |
| outputs/manuscript_checks/pooled_module_tests.csv                    |      2 |                                 0 | PASS     |
| outputs/manuscript_checks/unknown_withdrawal_eligible.csv            |     18 |                                 0 | PASS     |
| outputs/manuscript_checks/unknown_withdrawal_models.csv              |      4 |                                 0 | PASS     |
| outputs/manuscript_checks/unknown_withdrawal_OR.csv                  |     12 |                                 0 | PASS     |
| outputs/manuscript_checks/unknown_withdrawal_gaps.csv                |      6 |                                 0 | PASS     |
| outputs/manuscript_checks/leave_one_module_out_association.csv       |      7 |                                 0 | PASS     |
| outputs/manuscript_checks/no_interaction_probability_gaps.csv        |      3 |                                 0 | PASS     |

All18 manuscript-source exports match the frozen original within1e-7 relative/1e-9 absolute tolerances; the table above records actual maximum differences. The primary51-column design is rank51, with finite covariance and convergence independently checked. Primary known-IMD p=.6759825187, seven-dayOR2.285533/2.236554/2.308700 and day21gap.0758764834 [.0534238344,.0983291325] independently reproduce. No validated manuscript table rows were changed.

Core models selected C=.3 using2013 validation only. Every fresh test probability equals its archived counterpart (maxdifference0); ROC-AUC and Brier match. Each original individual AUC interval and paired C-minus-B/D-minus-C interval matches within1e-10. This resolves manuscript-specific current execution evidence; it does not prove every historical run's chronology. Current primary module checks are now rerun rather than merely read from saved tables.

The initial clean run passed16 tests (11original with explicit absent-input failures plus5 additional variance/artifact/numerical checks), with three known SHAP/matplotlib pending-deprecation warnings. The final verification includes the added PLOS figure contract check; its actual count/status is recorded in final_tests.log. The guarded copies reject nonconvergence, nonfinite covariance and invalid variance; original source/results were not edited. Direct raw-to-core rebuilding is computational reproduction, not independent external data validation.

Separate figures are RGB TIFF at300dpi, within required dimensions/size, with PNG/SVG and sourceCSV. Main PLOS PDF intentionally has captions without embedded figures. Figure-format checks, actual compilation and visual QA are separately logged. Sampling/model limitations and author-controlled declaration/rights items remain; a successful computation does not establish publication acceptance or ethics approval.
