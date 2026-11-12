"""Create reproducible *training-only* audio augmentation from original stereo fan WAVs.

Output: exactly --count enhanced stereo WAV files plus manifest.csv and
original_splits.csv. Original recordings are split BEFORE augmentation.

IMPORTANT: Augmented files share their source identity. Never test on a
variant of a recording that was used for training.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import re

import numpy as np
import soundfile as sf
from scipy import signal
from sklearn.model_selection import train_test_split

SEED = 2026


def load_manifest(source):
    with (source / 'manifest.csv').open(newline='', encoding='utf-8') as stream:
        originals = list(csv.DictReader(stream))
    if not originals:
        raise ValueError('Empty manifest')
    for row in originals:
        if not (source / row['file']).is_file():
            raise FileNotFoundError(source / row['file'])
        if not row.get('label'):
            raise ValueError(f'Label missing in {row}')
    return originals


def split_originals(rows, seed=SEED):
    """Stratified 60/20/20 split, before augmentation, by original recording."""
    labels = np.array([r['label'] for r in rows])
    counts = Counter(labels)
    if len(counts) < 2 or min(counts.values()) < 12:
        raise ValueError(f'Need at least 2 labels and >=12 originals per class: {dict(counts)}')
    ids = np.arange(len(rows))
    trval, test = train_test_split(ids, test_size=.2, stratify=labels, random_state=seed)
    train, val = train_test_split(trval, test_size=.25,
                                stratify=labels[trval], random_state=seed)
    return {'train': set(map(int, train)), 'val': set(map(int, val)),
            'test': set(map(int, test))}


def convert(wave, sr, target_sr):
    if sr == target_sr:
        return wave.astype(np.float32, copy=False)
    gcd = np.gcd(sr, target_sr)
    return signal.resample_poly(wave, target_sr // gcd, sr // gcd, axis=0).astype(np.float32)


def create_variant(wave, sr, seconds, rng):
    """Moderate gain, crop jitter, EQ, echo and microphone noise; no pitch shifts.

    Pitch shifting would directly change fan rotation harmonic frequencies,
    potentially invalidating the original speed label, so it is excluded.
    """
    n = int(round(seconds * sr))
    if len(wave) < n:
        wave = np.pad(wave, ((0, n-len(wave)), (0, 0)), mode='reflect')
    offset = int(rng.integers(0, max(1, len(wave) - n + 1)))
    y = wave[offset:offset+n].copy()

    gain_db = float(rng.uniform(-3.0, 3.0))
    far_gain_db = float(rng.uniform(-2.0, 2.0))
    gains = 10 ** (np.array([gain_db, gain_db+far_gain_db], dtype=np.float32)/20)
    y *= gains

    cutoff = float(rng.uniform(1800, min(6400, sr * .44)))
    sos = signal.butter(2, cutoff, btype='lowpass', fs=sr, output='sos')
    eq_prob = float(rng.uniform(.12, .58))
    lowpassed = signal.sosfilt(sos, y, axis=0).astype(np.float32)
    y = ((1-eq_prob)*y + eq_prob*lowpassed).astype(np.float32)

    delay_ms = float(rng.uniform(35, 110))
    echo = float(rng.uniform(0, .16))
    delay = int(round(delay_ms/1000 * sr))
    y[delay:] += echo * y[:-delay].copy()

    # Independent microphone noise; relative to the source RMS.
    noise_strength = float(rng.uniform(.005, .055))
    rms = float(np.sqrt(np.mean(np.square(wave)) + 1e-10))
    y += rng.normal(0, max(.0003, noise_strength*rms), size=y.shape).astype(np.float32)

    max_peak = max(.98, float(np.max(np.abs(y))))
    y = np.clip(y * (.98/max_peak), -.98, .98).astype(np.float32)
    return y, {
        'crop_start_sec': round(offset/sr, 4), 'gain_db': round(gain_db, 3),
        'far_gain_db': round(far_gain_db, 3), 'lowpass_hz': round(cutoff, 1),
        'lowpass_mix': round(eq_prob, 3), 'echo_delay_ms': round(delay_ms, 2),
        'echo_mix': round(echo, 4), 'noise_relative_rms': round(noise_strength, 4)
    }


def generate(source, out, count=1000, seed=SEED, seconds=2.0, rate=16000):
    if count < 1:
        raise ValueError('Count must be positive')
    originals = load_manifest(source)
    splits = split_originals(originals, seed)
    if out.resolve() == source.resolve():
        raise ValueError('Output path must differ from source path')
    out.mkdir(parents=True, exist_ok=True)
    # Fail rather than silently combine with a stale dataset.
    if (out / 'manifest.csv').exists():
        raise FileExistsError(f'{out}/manifest.csv exists. Remove the old output folder or choose --out.')

    split_lookup = {idx: name for name, indices in splits.items() for idx in indices}
    with (out / 'original_splits.csv').open('w', newline='', encoding='utf-8') as stream:
        cols = list(originals[0]) + ['split']
        writer = csv.DictWriter(stream, fieldnames=cols)
        writer.writeheader()
        for idx, row in enumerate(originals):
            writer.writerow(dict(row, split=split_lookup[idx]))

    train_by_label = {}
    for i in splits['train']:
        train_by_label.setdefault(originals[i]['label'], []).append(i)
    for members in train_by_label.values():
        members.sort(key=lambda idx: originals[idx]['file'])
    classes = sorted(train_by_label)
    rng = np.random.default_rng(seed)
    samples_per_class = {name: count//len(classes) + (i < count%len(classes))
                         for i, name in enumerate(classes)}

    # Load original sources just once and resample preserving near/far pairing.
    source_audio = {}
    for idx in sorted(splits['train']):
        wave, src_sr = sf.read(str(source / originals[idx]['file']),
                               dtype='float32', always_2d=True)
        if wave.shape[1] != 2:
            raise ValueError(f'Expected exactly two channels (near/far): {originals[idx]["file"]}')
        source_audio[idx] = convert(wave, src_sr, rate)

    manifest_rows = []
    for label in classes:
        for k in range(samples_per_class[label]):
            source_idx = train_by_label[label][k % len(train_by_label[label])]
            # Every source gets different random augmentations, label preserved.
            wave, details = create_variant(source_audio[source_idx], rate, seconds, rng)
            safe_label = re.sub('[^A-Za-z0-9_-]+', '-', label)
            filename = f'aug_{safe_label}_{k:04d}.wav'
            sf.write(str(out / filename), wave, rate, subtype='PCM_16')
            manifest_rows.append({
                'file': filename, 'label': label,
                'source_file': originals[source_idx]['file'],
                'source_row': source_idx, 'split': 'train',
                'origin': originals[source_idx].get('origin', 'unknown'),
                'channels': 'near=0;far=1',
                'distance_m': originals[source_idx].get('distance_m', 'not provided'),
                **details})
        print(f'Generated {samples_per_class[label]} files for {label}', flush=True)

    with (out / 'manifest.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
        writer.writeheader(); writer.writerows(manifest_rows)
    metadata = {
        'source': str(source), 'out': str(out), 'seed': seed,
        'augmentation_files': len(manifest_rows), 'original_files': len(originals),
        'original_train_files': len(splits['train']),
        'original_validation_files': len(splits['val']),
        'original_test_files': len(splits['test']),
        'labels': samples_per_class, 'sample_rate': rate,
        'duration_seconds': seconds,
        'original_type': sorted({r.get('origin', 'unknown') for r in originals}),
        'critical_warning': 'Derivatives are NOT independent recordings. Never evaluate on derivatives of training audio. Synthetic demo source must not be described as a real fan recording.'
    }
    with (out/'augmentation_info.json').open('w', encoding='utf-8') as stream:
        json.dump(metadata, stream, indent=2)
    print(f'Created exactly {len(manifest_rows)} TRAINING-ONLY augmented stereo WAV files at {out}')
    print(f'Original held-out files remain in {source}; they are not augmented.')
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='data/real')
    parser.add_argument('--out', default=None)
    parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('--seconds', type=float, default=2.0)
    parser.add_argument('--rate', type=int, default=16000)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    src = Path(args.source)
    dest = Path(args.out) if args.out else Path('data') / f'augmented_{src.name}_{args.count}'
    generate(src, dest, count=args.count, seed=args.seed,
             seconds=args.seconds, rate=args.rate)


if __name__ == '__main__':
    main()
