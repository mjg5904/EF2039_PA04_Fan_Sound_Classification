"""Inspect dataset recordings and report labels, channels, rate, duration."""
from pathlib import Path
import argparse
import csv
from collections import Counter
import soundfile as sf


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/real')
    args = p.parse_args()
    data = Path(args.data)
    with (data / 'manifest.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    print('Number of original recordings:', len(rows))
    print('Labels:', dict(Counter(r['label'] for r in rows)))
    print('Origins:', dict(Counter(r.get('origin', '') for r in rows)))
    for row in rows[:5]:
        file = data / row['file']
        inf = sf.info(file)
        print(f'{file.name}: {inf.samplerate} Hz, {inf.channels} channels, {inf.duration:.1f} seconds')
        if inf.channels < 2:
            raise ValueError('Expected two-channel WAV: ' + str(file))
    print('Preview complete. If origin says SYNTHETIC, do not use as paper data.')


if __name__ == '__main__':
    main()
