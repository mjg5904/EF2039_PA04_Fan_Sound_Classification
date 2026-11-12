"""Compare original-only vs augmented (1000 WAVs) training on untouched originals.

All metrics are evaluated ONLY on the original unseen stereo recordings.
The split file is written at augmentation time, so no source variants can leak.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import joblib
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from run_experiments import audio_features, make_model


def feature_pair(file, seconds=2):
    x, sr = sf.read(str(file), always_2d=True, dtype='float32')
    if x.shape[1] != 2:
        raise ValueError(f'Expected paired microphones in {file}')
    length = int(round(seconds*sr))
    if len(x) > length:
        s = (len(x)-length)//2
        x = x[s:s+length]
    return audio_features(x[:, 0], sr), audio_features(x[:, 1], sr)


def csv_rows(file):
    with file.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def run(source, augmented, out):
    original = csv_rows(augmented/'original_splits.csv')
    augmented_rows = csv_rows(augmented/'manifest.csv')
    assert len(augmented_rows) > 0
    origin_by_file = {row['file']: row for row in original}
    for a in augmented_rows:
        if a['split'] != 'train':
            raise ValueError('Found augmented evaluation sample: '+a['file'])
        if origin_by_file[a['source_file']]['split'] != 'train':
            raise ValueError('DATA LEAKAGE: '+a['file']+' from '+a['source_file'])
        if a['label'] != origin_by_file[a['source_file']]['label']:
            raise ValueError('Mismatched label: '+a['file'])
    out.mkdir(parents=True, exist_ok=True)
    classes = sorted(set(r['label'] for r in original))
    original_features = {}
    for i, r in enumerate(original, 1):
        original_features[r['file']] = feature_pair(source/r['file'])
        if i % 25 == 0: print(f'Original features {i}/{len(original)}', flush=True)
    augmented_features = []
    for i, r in enumerate(augmented_rows, 1):
        augmented_features.append(feature_pair(augmented/r['file']))
        if i % 100 == 0: print(f'Augmented features {i}/{len(augmented_rows)}', flush=True)
    augmentation_by_label = np.array([r['label'] for r in augmented_rows])
    aug_n = np.asarray([v[0] for v in augmented_features], dtype=np.float32)
    aug_f = np.asarray([v[1] for v in augmented_features], dtype=np.float32)
    train = [r for r in original if r['split']=='train']
    val = [r for r in original if r['split']=='val']
    test = [r for r in original if r['split']=='test']
    ytrain = np.asarray([r['label'] for r in train])
    ytest = np.asarray([r['label'] for r in test])
    original_near = np.asarray([original_features[r['file']][0] for r in train])
    original_far = np.asarray([original_features[r['file']][1] for r in train])
    results = []
    designs = {
        'original_near_only': (original_near, ytrain),
        'augmented_near_only': (aug_n, augmentation_by_label),
        'augmented_near_far': (np.concatenate([aug_n, aug_f]),
                               np.concatenate([augmentation_by_label, augmentation_by_label]))
    }
    for name, (x_train, y_train) in designs.items():
        print('Training:', name, 'n_channels=', len(y_train), flush=True)
        clf = make_model()
        clf.fit(x_train, y_train)
        joblib.dump(clf, out / f'{name}_model.joblib')
        for mic in ['near', 'far']:
            mi = 0 if mic=='near' else 1
            x_test = np.asarray([original_features[r['file']][mi] for r in test])
            predictions = clf.predict(x_test)
            record = {'training': name, 'test_mic': mic,
                      'accuracy': float(accuracy_score(ytest, predictions)),
                      'macro_f1': float(f1_score(ytest, predictions, labels=classes,
                                                 average='macro', zero_division=0)),
                      'original_test_count': len(ytest),
                      'training_channel_clips': int(len(y_train))}
            results.append(record)
            matrix = confusion_matrix(ytest, predictions, labels=classes)
            with (out/f'{name}_{mic}_confusion.csv').open('w', newline='') as stream:
                w=csv.writer(stream); w.writerow(['actual/pred']+classes)
                for cl, values in zip(classes, matrix): w.writerow([cl]+list(values))
            print(mic, 'test:', 'accuracy=', round(record['accuracy'], 3),
                  'macro-F1=', round(record['macro_f1'], 3))
    with (out/'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        w=csv.DictWriter(stream, fieldnames=list(results[0])); w.writeheader(); w.writerows(results)
    fig, ax=plt.subplots(figsize=(8.2,4.6))
    names=list(designs)
    offsets=[-.16, .16]
    for off, mic in zip(offsets, ['near','far']):
        ys=[next(r['accuracy'] for r in results if r['training']==name and r['test_mic']==mic)
            for name in names]
        ax.bar(np.arange(3)+off, ys, .32, label=mic)
    ax.set_xticks(np.arange(3), ['Original near', '1000 aug. near', '1000 aug. near+far'])
    ax.set_ylim(0,1.05); ax.set_ylabel('Accuracy on untouched original test recordings')
    ax.legend(title='Test microphone')
    fig.tight_layout(); fig.savefig(out/'comparison.png', dpi=170); plt.close(fig)
    meta={'original_source': str(source), 'augmented_training_source': str(augmented),
          'n_originals': len(original), 'n_original_train': len(train),
          'n_original_val': len(val), 'n_original_test': len(test),
          'n_generated_train': len(augmented_rows), 'seed': 2026,
          'classes': classes, 'original_data_origin': sorted(set(r.get('origin','') for r in original)),
          'warning': ('Demo originals are synthetic and cannot support real fan conclusions.'
                      if any('SYNTHETIC' in r.get('origin','') for r in original) else
                      'Real experimental results; recordings reused only within training split.')}
    with (out/'experiment_info.json').open('w') as f: json.dump(meta,f,indent=2)
    print('Saved:', out/'summary.csv', out/'comparison.png')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source', default='data/real')
    p.add_argument('--augmented', default='data/augmented_real_1000')
    p.add_argument('--out', default='results/augmented_real')
    a=p.parse_args(); run(Path(a.source), Path(a.augmented), Path(a.out))


if __name__ == '__main__': main()
