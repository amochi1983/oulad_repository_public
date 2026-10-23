# OULAD: early engagement and area-deprivation analysis

**Software author and maintainer: Aymen Ochi**  
ORCID: [0009-0009-2633-3341](https://orcid.org/0009-0009-2633-3341)  
Contact: amochi@uhb.edu.sa

Reproducibility code for the observational study *Early virtual learning engagement, area deprivation and academic success: an observational OULAD study*. Manuscript coauthors: Abdullah Muzil Alharbi and Aymen Ochi. Their confirmed contributions are in [AUTHORS.md](AUTHORS.md). The software citation names Aymen Ochi; automated tools are not authors.

The software is provided under [MIT](LICENSE). OULAD data and data-derived contents have separate [CC BY 4.0 terms and attribution](DATA_LICENSE.md). See [AI_ASSISTANCE.md](AI_ASSISTANCE.md) for the separate tool-use disclosure.

## Scope and interpretation

- Original OULAD cohorts: **2013-2014**, not newly collected students.
- Main recorded day-28 cohort: **27,528 enrolments from 24,835 students**.
- Primary analysis: active-day engagement, residential-area IMD and final success; student-cluster covariance; model-standardised absolute contrasts.
- Secondary analysis: original 2013 development and student-disjoint 2014 B/C/D logistic prediction of non-success.
- IMD is residential-area deprivation, not personal income. Associations are not causal effects, and historical temporal testing is not contemporary external validation.
- At **21 active days within 28**, the archived adjusted least-minus-most-deprived gap is approximately 7.6 percentage points. This is not a day-21 prediction cutoff.

This software accompanies an unpublished manuscript. Public release does not imply journal acceptance or an institutional ethics determination. The local analysis plan is historical documentation, not external preregistration. Later sensitivity/reporting additions are explicitly post-analysis.

## Environment and installation

Audited scientific environment: Python **3.12.14** on Windows. Exact dependencies are in requirements-exact.txt. The earlier requirements files are retained as provenance; use the exact file for this package.

```bash
python -m venv .venv
```

Activate the environment (`.venv\Scripts\Activate.ps1` in PowerShell, or `source .venv/bin/activate` on Linux/macOS), then:

```bash
python -m pip install -r requirements-exact.txt
```

No fresh cross-platform installation is claimed. See VALIDATION.md for the checks actually performed on the upload package. Model-dependent binaries should only be loaded from a source you trust; the included pipelines are study artefacts.

## Source data

Kuzilek J, Hlosta M, Zdrahal Z (2017), **OULAD version 1**: https://doi.org/10.6084/m9.figshare.5081998.v1 . Descriptor: https://doi.org/10.1038/sdata.2017.171 .

The large raw CSVs/archive are intentionally excluded. The source manifest and metadata remain in data/raw/. Included processed cohorts, split assignments and reference outputs derive from the same public observations. Source checksums are enforced, not silently replaced. Restore the raw source and run the included-data checks:

```bash
python reproduce.py --download
```

This command obtains/checks the raw files and executes tests. It **does not** refit all models. It fails if required data/artefacts or expected hashes are missing or inconsistent.

## Full manuscript-core reproduction

```bash
python reproduce.py --download --rebuild-features --refit-association --retrain-core
```

If raw data already exist, omit `--download`. This rebuilds window features, refits primary/sensitivity/module association analyses, generates figures, reruns the bounded B/C/D logistic tuning/refit/calibration and original 200-draw individual/paired test-student bootstrap, and runs tests. The historical full-core execution is documented separately; this is not the wider tree-model/window search. Seeds and repetitions are in configs/analysis.json and the source (seed 20261005; descriptive bootstrap 1000, predictive bootstrap 200).

Quick checks after source data are restored:

```bash
python reproduce.py
python -m pytest tests -q -p no:cacheprovider
```

Regeneration writes only inside the working checkout. **reference_tables/** is a frozen comparison set: never rewrite it just to force tests to pass. Preserve a clean checkout or extracted copy if comparing before/after results. A package manifest records the upload snapshot; generated runtime outputs will legitimately differ after later execution.

## Repository map

| Path | Contents |
|---|---|
| src/ | Eligibility/features, descriptive and adjusted analyses, prediction, figure generation |
| tools/retrain_core.py | Bounded B/C/D logistic reproduction rather than wider model search |
| configs/ | Frozen analysis configuration |
| data/raw/ | Pinned official metadata and raw SHA256 manifest; no raw bulk |
| data/processed/ | Public-data-derived cohorts needed for the supplied checks |
| outputs/ | Numerical tables, source figures and selected core model/prediction artefacts |
| reference_tables/ | Immutable historical numerical comparison targets |
| reports/ | Scientific methods, frozen split assignments and dated execution evidence |
| provenance_2026-10-18/ | Dated source/derivative lineage evidence, not new student data |
| tests/ | Boundary, split, inference, numerical, figure and acquisition checks |
| AUTHORS.md / CITATION.cff | Human authorship, contributions and software citation |
| VALIDATION.md | Validation scope for this upload package |
| RELEASE_CHANGES.md | Packaging/licensing changes relative to the local scientific copy |

The supplied predictions/pipelines are limited to the manuscript's week-four B/C/D Logistic comparison, not every earlier model/window. Broader scientific source modules remain archival context, not a recommendation to select a better test result. Raw bulk, .git history, private documents, publisher PDFs, compiler binaries and the manuscript/Overleaf project are excluded from this software upload.

## Scientific limits and provenance

Student clustering does not remove confounding or account for all course shocks. Fixed observed contexts and seven modules do not establish population-wide transport. Missing withdrawal timing is not invented; scores lack verified marking/release times and are not early predictors in the B/C/D comparison. A non-detected interaction is not evidence of equivalence. Predictive intervals condition on fitted models and have only 200 bootstrap draws.

The 18 October 2026 audit refitted the manuscript-core analyses and reconciled numerical outputs; later editorial checks do not constitute another full retraining run. Historical source-copy hashes document earlier versions; release edits are described separately. No new novelty extension or causal/intervention claim is introduced.

## Public repository and post-audit revision

Published analysis snapshot (before documentation correction): https://github.com/amochi1983/OULAD_Git (commit ced4606ba39dfff3b228926f68fcdb68e6d209ab).
This snapshot adds exploratory adjustment/reference and English-region checks, descriptive test calibration, and neutral figure terminology after the Claude-assisted internal computational audit of 22 October 2026. The post-audit analyses are public; the 23 October correction updates attribution, provenance and documentation without changing the numerical estimates. No code DOI has been assigned.

Run the new checks with `python reproduce.py --post-audit`, or include them with `python reproduce.py --rebuild-features --refit-association --retrain-core`. The `--download` flag alone verifies acquisition and runs tests; it does not refit models. `reference_tables/` retains immutable original results. New checks write to `outputs/post_audit/`; they do not replace primary estimates or redefine the original multiplicity family.

Original primary odds ratios and contrasts remain observational. Context-only and fully adjusted contrasts answer different conditional questions; education, previous attempts and region are not claimed to identify a sufficient causal adjustment set. Software authorship remains Aymen Ochi; scientific coauthorship and tool disclosure are recorded separately.



23 October 2026 editorial correction: Claude (Anthropic) is disclosed alongside ChatGPT/Codex (OpenAI). Internal AI-assisted audits are distinguished from human or journal peer review. Statistical estimates and the recorded 21-test result are unchanged; the post-audit provenance field and test description were clarified.
