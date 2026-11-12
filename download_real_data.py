"""Extract REAL labeled DCASE 2026 fan recordings for a classroom binary classifier.

Classes: normal and anomaly (NOT low / medium / high fan speed).
Original record: https://zenodo.org/records/19336329, dev_fan.zip
Both synchronous channels are preserved: channel 0 near and channel 1 far.

The official DCASE task is unsupervised anomalous sound detection; this script
repurposes *its originally labeled test recordings* as a new, supervised
educational classification dataset. The downstream code creates NEW train/val/
test splits; results must NOT be presented as official DCASE benchmark scores.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import random
import shutil
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

URL = 'https://zenodo.org/records/19336329/files/dev_fan.zip?download=1'
MD5 = '58b79bac4b05bf14a11e9d53e6cdfae9'
SEED = 2026


def download(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print('Using existing ZIP:', path)
        return
    temp = path.with_suffix('.part')
    print('Downloading DCASE 2026 dev_fan.zip (approximately 704 MB)')
    request = urllib.request.Request(URL, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(request, timeout=100) as response, temp.open('wb') as out:
            total = 0
            next_report = 100 * 1024 * 1024
            for block in iter(lambda: response.read(1024 * 1024), b''):
                out.write(block)
                total += len(block)
                if total >= next_report:
                    print(f'Downloaded: {total // (1024*1024)} MiB', flush=True)
                    next_report += 100 * 1024 * 1024
        temp.rename(path)
    finally:
        if temp.exists():
            temp.unlink()


def check_md5(path: Path) -> None:
    h = hashlib.md5()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(8*1024*1024), b''):
            h.update(block)
    actual = h.hexdigest()
    if actual != MD5:
        raise ValueError(f'Wrong ZIP MD5: {actual}; expected {MD5}')
    print('Official ZIP checksum verified.')


def select_entries(zip_file: zipfile.ZipFile, maximum_per_class: int):
    """Take equal counts of normal and anomaly clips, balanced by original domain."""
    groups = defaultdict(list)
    for entry in zip_file.infolist():
        path = entry.filename.replace('\\', '/')
        lower = '/' + path.lower().lstrip('/')
        if not lower.endswith('.wav') or '/fan/test/' not in lower:
            continue
        # Both *_source_test_normal_0001_*.wav and *_target_test_anomaly_0001_*.wav.
        words = Path(lower).name.split('_')
        if 'test' not in words:
            continue
        cls = 'anomaly' if 'anomaly' in words else ('normal' if 'normal' in words else None)
        domain = 'source' if 'source' in words else ('target' if 'target' in words else None)
        if cls and domain:
            groups[(cls, domain)].append(entry)
    print('Official labeled test WAV counts:',
          {f'{cls}/{domain}': len(v) for (cls, domain), v in sorted(groups.items())})
    for cls in ('normal', 'anomaly'):
        if not any(groups[(cls, domain)] for domain in ('source', 'target')):
            raise RuntimeError(f'No {cls} WAVs found in /fan/test/. Print ZIP filenames to inspect format.')

    rng = random.Random(SEED)
    # 100 per class expected (50/source + 50/target), but don't assume exact counts.
    available = {cls: sum(len(groups[(cls,d)]) for d in ('source','target')) for cls in ('normal','anomaly')}
    per_class = min(maximum_per_class, *available.values())
    if per_class < 12:
        raise RuntimeError(f'Need >=12 original recordings of each class: {available}')
    selection = []
    for cls in ('normal','anomaly'):
        source = sorted(groups[(cls,'source')], key=lambda x: x.filename)
        target = sorted(groups[(cls,'target')], key=lambda x: x.filename)
        # Allocate as evenly as possible across domains, without dropping a class.
        ns = min(len(source), per_class // 2)
        nt = min(len(target), per_class - ns)
        while ns + nt < per_class:
            if ns < len(source): ns += 1
            elif nt < len(target): nt += 1
            else: break
        for domain, entries, num in (('source',source,ns),('target',target,nt)):
            for e in rng.sample(entries,num):
                selection.append((cls,domain,e))
    rng.shuffle(selection)
    return selection


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--zip', default='data/dev_fan.zip', help='Locally saved official ZIP')
    p.add_argument('--out', default='data/real_binary')
    p.add_argument('--per-class', type=int, default=100)
    p.add_argument('--per-speed', type=int, default=None,
                   help='DEPRECATED; interpreted as --per-class (speed labels do not exist).')
    p.add_argument('--no-download', action='store_true')
    p.add_argument('--skip-md5', action='store_true')
    args = p.parse_args()
    if args.per_speed is not None:
        print('WARNING: speed classification is unsupported. --per-speed is treated as --per-class.')
        args.per_class = args.per_speed
    if args.per_class < 12:
        p.error('--per-class must be at least 12')
    archive = Path(args.zip)
    if not archive.is_file():
        if args.no_download:
            raise FileNotFoundError(f'{archive} missing. The 704 MB ZIP must already exist.')
        download(archive)
    if not args.skip_md5:
        check_md5(archive)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'manifest.csv').exists():
        raise FileExistsError(f'{out}/manifest.csv already exists; use a new --out for a fresh dataset.')

    with zipfile.ZipFile(archive) as z:
        chosen = select_entries(z, args.per_class)
        rows = []
        for cls in ('normal','anomaly'):
            records = [(domain, entry) for c, domain, entry in chosen if c == cls]
            for i,(domain, entry) in enumerate(records):
                filename = f'{cls}_{i:03d}.wav'
                with z.open(entry) as src, (out/filename).open('wb') as dest:
                    shutil.copyfileobj(src,dest)
                rows.append({
                    'file': filename, 'label': cls,
                    'origin': 'DCASE_2026_ZENODO_19336329',
                    'original_path': entry.filename,
                    'original_domain': domain,
                    'original_partition': 'test',
                    'channels': 'near=0;far=1',
                    'distance_m': 'not provided',
                    'task': 'repurposed supervised normal-vs-anomaly; NOT official DCASE protocol',
                })
    with (out/'manifest.csv').open('w', newline='',encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('SUCCESS. Extracted', len(rows), 'REAL stereo WAV files.')
    print('Classes:',dict(Counter(r['label'] for r in rows)))
    print('Domains:',dict(Counter(r['original_domain'] for r in rows)))
    print('Manifest:', out/'manifest.csv')
    print('Next: python check_data.py --data',out)
    print('IMPORTANT: this is NORMAL vs ANOMALOUS SOUND classification, NOT SPEED.')


if __name__ == '__main__':
    main()
