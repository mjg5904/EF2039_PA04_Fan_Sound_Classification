"""Fan operating-condition (normal/anomaly) classification: paired microphone distance experiments.

Run: python run_experiments.py --data data/real
All comparisons use identical held-out recordings, split before audio channels.
"""
from pathlib import Path
import argparse
import csv
import json
import warnings
import numpy as np
import soundfile as sf
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 2026
SR = 16000


def audio_features(wave, sr):
    """Return simple spectral + energy features for a mono WAV recording."""
    x = np.asarray(wave, dtype=np.float32)
    if len(x) == 0:
        raise ValueError('Empty WAV')
    if sr != SR:
        g = np.gcd(int(sr), SR)
        x = signal.resample_poly(x, SR // g, int(sr) // g)
    # Use the middle 4 seconds, avoiding start/stop transients.
    length = min(len(x), SR * 4)
    start = (len(x) - length) // 2
    x = x[start:start + length]
    if len(x) < 1024:
        x = np.pad(x, (0, 1024 - len(x)))
    rms = np.sqrt(np.mean(x**2) + 1e-10)
    zcr = np.mean(np.diff(np.signbit(x)) != 0)
    f, _, stft = signal.stft(x, fs=SR, nperseg=1024, noverlap=768)
    m = np.abs(stft).astype(np.float64) + 1e-9
    power = m ** 2
    # Average energy over 64 linearly spaced frequency bands (0-4000 Hz).
    keep = f <= 4000
    powers = power[keep]
    grouped = np.array([np.mean(band, axis=0) for band in np.array_split(powers, 64)])
    log_energy = np.log1p(10000 * grouped)
    mean = log_energy.mean(axis=1)
    std = log_energy.std(axis=1)
    centroid = (f[:, None] * power).sum(axis=0) / power.sum(axis=0)
    p = power / power.sum(axis=0, keepdims=True)
    flatness = np.exp(np.log(m).mean(axis=0)) / m.mean(axis=0)
    extra = np.array([np.log1p(rms * 1000), zcr, centroid.mean() / 8000,
                      centroid.std() / 8000, flatness.mean(), flatness.std()], dtype=float)
    return np.concatenate([mean, std, extra]).astype(np.float32)


def extract_pair(path):
    wave, sr = sf.read(str(path), always_2d=True, dtype='float32')
    if wave.shape[1] < 2:
        raise ValueError(f'{path} contains only {wave.shape[1]} channel(s); need near+far')
    return audio_features(wave[:, 0], sr), audio_features(wave[:, 1], sr)


def load_data(data_dir, cache_dir):
    manifest = data_dir / 'manifest.csv'
    if not manifest.is_file():
        raise FileNotFoundError(f'No manifest: {manifest}. Run download_real_data.py or make_demo.py first.')
    with manifest.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('Dataset is empty')
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / 'features.npz'
    # Cache is invalidated when number, path, size, or modification time change.
    import hashlib
    fingerprint = hashlib.sha256()
    fingerprint.update(str(data_dir.resolve()).encode())
    for row in rows:
        file = data_dir / row['file']
        if not file.exists():
            raise FileNotFoundError(file)
        st = file.stat()
        fingerprint.update(f'{row["file"]}|{row["label"]}|{st.st_size}|{st.st_mtime_ns}'.encode())
    digest = fingerprint.hexdigest()
    if cache_file.is_file():
        with np.load(cache_file) as cached:
            if str(cached['digest']) == digest:
                print('Loading cached audio features.')
                return rows, cached['near'], cached['far']
    near, far = [], []
    for i, row in enumerate(rows, 1):
        n, f = extract_pair(data_dir / row['file'])
        near.append(n); far.append(f)
        if i % 25 == 0 or i == len(rows):
            print(f'Features extracted: {i}/{len(rows)}', flush=True)
    near, far = np.asarray(near), np.asarray(far)
    np.savez_compressed(cache_file, digest=digest, near=near, far=far)
    return rows, near, far


def make_model():
    return RandomForestClassifier(n_estimators=120, max_depth=16, min_samples_leaf=2,
                                  class_weight='balanced', random_state=SEED, n_jobs=-1)


def create_split(rows, dest):
    ids = np.arange(len(rows))
    y = np.array([r['label'] for r in rows])
    counts = {label: int(np.sum(y == label)) for label in np.unique(y)}
    if len(counts) < 2 or min(counts.values()) < 12:
        raise ValueError(f'Need >=2 operating-condition classes and >=12 files/class, got: {counts}')
    trainval, test = train_test_split(ids, test_size=0.2, random_state=SEED, stratify=y)
    train, val = train_test_split(trainval, test_size=0.25, random_state=SEED,
                                  stratify=y[trainval])
    # Both channels of each recording retain the SAME split (no leakage).
    split_map = {}
    for section, index in [('train', train), ('val', val), ('test', test)]:
        for idx in index:
            split_map[int(idx)] = section
    with (dest / 'split_manifest.csv').open('w', newline='', encoding='utf-8') as f:
        cols = list(rows[0].keys()) + ['split']
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for idx, row in enumerate(rows):
            writer.writerow(dict(row, split=split_map[idx]))
    print('Dataset split (original recordings):', {k: sum(s == k for s in split_map.values())
                                                  for k in ('train', 'val', 'test')})
    return train, val, test, y


def measure(name, model, x, y, classes, dest, subset):
    pred = model.predict(x)
    acc = accuracy_score(y, pred)
    macro = f1_score(y, pred, labels=classes, average='macro', zero_division=0)
    report = classification_report(y, pred, labels=classes, zero_division=0, output_dict=True)
    conf = confusion_matrix(y, pred, labels=classes)
    with (dest / f'{name}_{subset}_class_report.json').open('w') as f:
        json.dump(report, f, indent=2)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(conf, cmap='Blues')
    ax.set_xticks(np.arange(len(classes)), classes, rotation=25)
    ax.set_yticks(np.arange(len(classes)), classes)
    ax.set(xlabel='Predicted condition', ylabel='True condition', title=f'{name} / {subset}')
    for (i, j), num in np.ndenumerate(conf):
        ax.text(j, i, str(num), ha='center', va='center',
                color='white' if num > conf.max() / 2 else 'black')
    fig.colorbar(im, ax=ax, fraction=.045)
    fig.tight_layout()
    fig.savefig(dest / f'{name}_{subset}_confusion.png', dpi=170)
    plt.close(fig)
    return {'model': name, 'test_microphone': subset,
            'accuracy': float(acc), 'macro_f1': float(macro), 'test_samples': int(len(y))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/real')
    p.add_argument('--out', default=None)
    args = p.parse_args()
    data = Path(args.data)
    dest = Path(args.out) if args.out else Path('results') / data.name
    dest.mkdir(parents=True, exist_ok=True)
    rows, near, far = load_data(data, dest)
    train, val, test, y = create_split(rows, dest)
    classes = sorted(set(y))
    rows_result = []
    print('Condition classes:', classes)
    for model_name in ('near_only', 'near_and_far'):
        if model_name == 'near_only':
            x_train, y_train = near[train], y[train]
        else:
            x_train = np.vstack([near[train], far[train]])
            y_train = np.concatenate([y[train], y[train]])
        model = make_model()
        print(f'Training {model_name} with {len(y_train)} channel recordings.')
        model.fit(x_train, y_train)
        joblib.dump(model, dest / f'{model_name}_model.joblib')
        for microphone, x in [('near', near), ('far', far)]:
            # Validation is reported separately, never used for fitting or tuning.
            rows_result.append(measure(model_name, model, x[test], y[test], classes, dest, microphone))
            val_acc = accuracy_score(y[val], model.predict(x[val]))
            print(f'  {microphone} test accuracy={rows_result[-1]["accuracy"]:.3f};'
                  f' macro-F1={rows_result[-1]["macro_f1"]:.3f}; val_accuracy={val_acc:.3f}')
    with (dest / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows_result[0].keys())
        writer.writeheader(); writer.writerows(rows_result)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    keys = ['near_only', 'near_and_far']
    x0 = np.arange(len(keys))
    w = 0.33
    for offset, mic in [(-w/2, 'near'), (w/2, 'far')]:
        values = [next(r['accuracy'] for r in rows_result if r['model']==model and
                       r['test_microphone']==mic) for model in keys]
        ax.bar(x0 + offset, values, width=w, label=mic)
    ax.set_xticks(x0, ['Near-only training', 'Near + far training'])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Held-out accuracy')
    ax.set_title('Fan condition classification by microphone distance')
    ax.legend()
    fig.tight_layout()
    fig.savefig(dest / 'comparison.png', dpi=180)
    plt.close(fig)
    meta = {'dataset': str(data), 'sample_origin': sorted(set(row.get('origin', '') for row in rows)),
            'seed': SEED, 'classes': classes, 'num_original_recordings': len(rows),
            'train_recordings': len(train), 'validation_recordings': len(val),
            'test_recordings': len(test), 'near_channel': 0, 'far_channel': 1,
            'training': 'RandomForestClassifier(120 trees, max_depth 16, min_samples_leaf 2)',
            'split': 'stratified per original stereo WAV, 60%/20%/20%',
            'warning': 'SYNTHETIC DEMO results must NOT be represented as real experiment results' if
                        'demo' in str(data).lower() else 'Real two-microphone DCASE data'}
    with (dest / 'experiment_info.json').open('w') as f:
        json.dump(meta, f, indent=2)
    a = next(r['accuracy'] for r in rows_result if r['model']=='near_only' and r['test_microphone']=='near')
    b = next(r['accuracy'] for r in rows_result if r['model']=='near_only' and r['test_microphone']=='far')
    c = next(r['accuracy'] for r in rows_result if r['model']=='near_and_far' and r['test_microphone']=='far')
    with (dest / 'PAPER_NUMBERS.txt').open('w') as f:
        f.write('Research questions (held-out original recordings)\n')
        f.write('RQ1: How does microphone distance affect normal/anomaly classification?\n')
        f.write(f'Near-only model, near test: {a:.3f}; far test: {b:.3f}; difference: {(a-b):+.3f}\n')
        f.write('RQ2: Does training on both microphones improve distant-sound accuracy?\n')
        f.write(f'Far-test near-only trained: {b:.3f}; both-microphones trained: {c:.3f};'
                f' change: {(c-b):+.3f}\n')
        f.write('Important: mixed training uses twice as many channel examples from the same training recordings.\n')
        if 'demo' in str(data).lower():
            f.write('WARNING: DEMO DATA ARE SYNTHETIC. NOT VALID FOR EMPIRICAL FAN CLAIMS.\n')
    print('Results saved to:', dest)
    print('Read:', dest / 'PAPER_NUMBERS.txt')


if __name__ == '__main__':
    main()
