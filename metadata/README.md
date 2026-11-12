# Additional exact-run metadata

These files can be copied from the original Backend.AI session to complete the reproducibility record:

- `data/real_binary/manifest.csv` -> `metadata/real_source_manifest.csv`
- `data/augmented_binary_1000/original_splits.csv` -> `metadata/original_splits.csv`
- `data/augmented_binary_1000/manifest.csv` -> `metadata/augmentation_manifest.csv`
- `data/augmented_binary_1000/augmentation_info.json` -> `metadata/augmentation_info.json`
- `python -m pip freeze` output -> `installed_versions.txt` in repository root.

Only metadata, not audio WAV files, is intended to be committed here. The `results/augmented_binary/experiment_info.json` is already included in the repository.
