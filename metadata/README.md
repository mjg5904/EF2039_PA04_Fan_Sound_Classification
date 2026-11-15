# Additional exact-run metadata

These files can be copied from the original Backend.AI session to complete the reproducibility record:

- `data/real_binary/manifest.csv` -> `metadata/real_source_manifest.csv`
- `data/augmented_binary_1000/original_splits.csv` -> `metadata/original_splits.csv`
- `data/augmented_binary_1000/manifest.csv` -> `metadata/augmentation_manifest.csv`
- `data/augmented_binary_1000/augmentation_info.json` -> `metadata/augmentation_info.json`
- `python -m pip freeze` output -> `installed_versions.txt` in repository root.

Only metadata, not audio WAV files, is intended to be committed here. The `results/augmented_binary/experiment_info.json` is already included in the repository.

## Reproducibility Verification

On November 15, 2026, all three models were retrained and evaluated on Backend.AI using a fresh clone of this GitHub repository and the official dataset archive. All accuracy and Macro-F1 scores exactly matched the original results. The two summary.csv files were identical.
