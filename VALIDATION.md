# Validation of the post-audit snapshot - 22 October 2026

New execution: `python reproduce.py --post-audit`, exit 0. The command verified the eight raw-file hashes against the immutable source manifest, reran full/context-only/reduced-adjustment and English-region GLMs, reference-population contrasts, descriptive test calibration, and all **21 tests (21 passed)**. Three third-party pending-deprecation warnings remain. The two new tests check contrasts against separate calculations from the Claude-assisted audit and calibration score equations against unchanged test probabilities.

The same existing Python 3.12.14 environment was used; no fresh installation or cross-platform validation was performed. Raw data were copied from the unchanged local official-v1 files for this run; no new source download is claimed. The entry-point log is supplied in reports/post_audit_reproduction.log. New outputs are in outputs/post_audit/. Primary definitions and immutable reference_tables/ were not changed. Main primary numerical tables were compared against the previous snapshot.

Earlier dated evidence: the independent 22 October audit documented raw-feature reconstruction, original-model refits, B/C/D retraining, 19 original tests and primary calculations in a separate Claude-assisted implementation. That full historical pipeline was not rerun again for these post-audit additions. The wider tree/window search remains outside the manuscript reproduction scope. `--download` alone does not refit models.

The original public analysis baseline was commit ced4606ba39dfff3b228926f68fcdb68e6d209ab (22 October 2026). The post-audit analyses are public, with provenance and documentation corrected in the 23 October update. No permanent code DOI is assigned. Computational validation does not establish ethics approval, final human-author approval or acceptance by a journal.



23 October 2026 editorial correction: Claude (Anthropic) is disclosed alongside ChatGPT/Codex (OpenAI). Internal AI-assisted audits are distinguished from human or journal peer review. Statistical estimates and the recorded 21-test result are unchanged; the post-audit provenance field and test description were clarified.
