# Data source, limitations, and license

## Real dataset: download required

**DCASE 2026 Challenge Task 2 Development Dataset**, published April 1, 2026, Zenodo record **19336329**, DOI: https://doi.org/10.5281/zenodo.19336329.

Named creators: Tomoya Nishida, Noboru Harada, Daiki Takeuchi, Daisuke Niizumi, Keisuke Imoto, Kota Dohi, Harsh Purohit, Takashi Endo, and Yohei Kawaguchi (see Zenodo record for affiliations and definitive attribution).

Official resource: https://zenodo.org/records/19336329

Original license: **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International**, see https://creativecommons.org/licenses/by-nc-sa/4.0/

Original archive: `dev_fan.zip` (704.2 MB, MD5 `58b79bac4b05bf14a11e9d53e6cdfae9`). The original WAV recordings are NOT INCLUDED in this small starter ZIP: they must be downloaded separately by the included script, because the archive could not be downloaded in the workspace where this starter was prepared. **Do not state that original audio is enclosed in this ZIP.**

The original task is **unsupervised anomalous sound detection**, not fan speed classification. For our new supervised classroom task, we **repurpose the normal/anomaly-labeled recordings from the original DCASE test partition**, then re-split those recordings for educational training and evaluation. This is NOT a valid official DCASE competition result. Channel 0 is near and channel 1 is far. Exact physical microphone distances are not provided.

Data preparation does **not** change audio content; it selects and renames WAV files and records their original paths in `data/real_binary/manifest.csv`. A new 60/20/20 train-validation-test split is generated on the selected recordings, not inherited from DCASE. Original source/target domain differences may confound condition classification and should be discussed as a limitation.

## Synthetic example audio

Files in `data/demo/` are **programmatically generated** by `make_demo.py`. They are *not* microphones or recordings of a real fan. Do not include demo metrics in the submitted research results. The synthetic demo audio can be recreated using `python make_demo.py --n 30`.
