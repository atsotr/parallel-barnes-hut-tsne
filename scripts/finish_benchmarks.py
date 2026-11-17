#!/usr/bin/env python3
"""Graph t-SNE assignment benchmarks. Python standard library only.

Run from repo root: python3 scripts/finish_benchmarks.py
Generates results/benchmark_summary.csv, raw_runs.csv, bh_vs_exact.csv,
and SVG speedup charts. All timing and raw output are captured for reproducibility.
"""
from __future__ import annotations
import csv
import datetime as dt
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'benchmarks'
LOGS = OUT / 'logs'
DATASETS = ('k49_norm_10NN', 'com-Amazon', 'coPapersDBLP')
THREADS = (1, 2, 4, 8)
REPEATS = 3
PHASES = ('probabilities', 'attraction', 'tree_build', 'center_of_mass',
          'barnes_hut_walk', 'gradient')


def run(cmd, *, env=None, log=None):
    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    output = result.stdout + ('\n' + result.stderr if result.stderr else '')
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text('$ ' + ' '.join(map(str, cmd)) + '\n' + output, encoding='utf-8')
    return result.returncode, output, elapsed


def write_csv(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path):
    if not path.is_file():
        return []
    with path.open(newline='', encoding='utf-8') as file:
        return list(csv.DictReader(file))


def svg_speedup(dataset, medians):
    colors = {'pthreads': '#2563eb', 'omp': '#16a34a', 'cilk': '#d97706'}
    base = medians.get((dataset, 'seq', 1))
    if not base:
        return
    width, height = 670, 400
    left, right, top, bottom = 68, 24, 54, 58
    w, h = width - left - right, height - top - bottom
    peaks = [base / t for (ds, mode, p), t in medians.items()
             if ds == dataset and mode != 'seq' and t > 0]
    ytop = max(2, math.ceil(max(peaks, default=1) * 1.12))
    def X(p): return left + (p - 1) / 7 * w
    def Y(v): return top + h - v / ytop * h
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{left}" y="27" font-family="Arial" font-weight="bold" font-size="17">{dataset}: speedup vs sequential</text>']
    for tick in range(ytop + 1):
        y = Y(tick)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-size="12" font-family="Arial">{tick}×</text>')
    for p in THREADS:
        lines.append(f'<text x="{X(p):.1f}" y="{top+h+24}" text-anchor="middle" font-size="12" font-family="Arial">{p}</text>')
    lines.append(f'<text x="{left+w/2}" y="{height-10}" text-anchor="middle" font-family="Arial" font-size="13">Workers / threads</text>')
    for idx, (mode, color) in enumerate(colors.items()):
        pts = [(X(p), Y(base / medians[(dataset, mode, p)])) for p in THREADS
               if (dataset, mode, p) in medians and medians[(dataset, mode, p)] > 0]
        if pts:
            path = ' '.join(f'{x:.1f},{y:.1f}' for x, y in pts)
            lines.append(f'<polyline points="{path}" stroke="{color}" stroke-width="3" fill="none"/>')
            for x, y in pts:
                lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        lx = left + 150 * idx
        lines += [f'<line x1="{lx}" y1="{height-34}" x2="{lx+20}" y2="{height-34}" stroke="{color}" stroke-width="3"/>',
                  f'<text x="{lx+25}" y="{height-30}" font-size="12" font-family="Arial">{mode}</text>']
    lines.append('</svg>')
    (OUT / f'speedup_{dataset}.svg').write_text('\n'.join(lines), encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    required = [ROOT / 'build' / ('tsne_' + m) for m in ('seq', 'pthreads', 'omp', 'cilk')]
    for path in required:
        if not path.is_file():
            sys.exit(f'Missing {path}. First compile with: make seq pthreads omp cilk')
    (OUT / 'environment.txt').write_text(
        f'UTC start: {dt.datetime.now(dt.timezone.utc).isoformat()}\n'
        f'Platform: {platform.platform()}\nCPU count visible: {os.cpu_count()}\n'
        + subprocess.run(['bash', '-lc', 'lscpu 2>/dev/null | head -25'],
                         capture_output=True, text=True).stdout, encoding='utf-8')

    raw_path = OUT / 'raw_runs.csv'
    fields = ['dataset', 'mode', 'threads', 'repeat', 'seconds', 'wall_seconds'] + list(PHASES)
    existing = load_csv(raw_path)
    records = list(existing)
    completed = {(r['dataset'], r['mode'], int(r['threads']), int(r['repeat'])) for r in existing}
    modes = [('seq', 1)] + [(m, t) for m in ('pthreads', 'omp', 'cilk') for t in THREADS]
    targets = [(ds, mode, t, r) for ds in DATASETS for mode, t in modes
               for r in range(1, REPEATS + 1)]
    print(f'Will collect {len(targets)} runs. Existing completed: {len(completed)}. Results: {OUT}', flush=True)

    for idx, (ds, mode, threads, repeat) in enumerate(targets, 1):
        key = (ds, mode, threads, repeat)
        if key in completed:
            continue
        dataset = ROOT / 'data' / (ds + '.mat')
        if not dataset.is_file():
            print(f'[SKIP] Missing dataset {dataset}', flush=True)
            continue
        print(f'[{idx}/{len(targets)}] {ds}, {mode}, {threads} threads, repeat {repeat}', flush=True)
        cmd = [str(ROOT/'build'/('tsne_'+mode)), 'data/'+ds]
        env = os.environ.copy()
        if mode == 'cilk':
            env['CILK_NWORKERS'] = str(threads)
        else:
            cmd += ['--threads', str(threads)]
        log = LOGS / f'{ds}_{mode}_{threads}_{repeat}.txt'
        with tempfile.TemporaryDirectory(prefix='tsne_', dir=OUT) as tmp:
            cmd += ['--out', tmp]
            code, stdout, wall = run(cmd, env=env, log=log)
            if code != 0:
                print(f'  ERROR (exit {code}); see {log}', flush=True)
                continue
            timefile = Path(tmp) / 'timings.csv'
            if not timefile.exists():
                print(f'  ERROR: timings.csv missing; see {log}', flush=True)
                continue
            timings = {r['phase']: float(r['seconds']) for r in load_csv(timefile)}
            if any(k not in timings for k in PHASES):
                print(f'  ERROR: incomplete timings; see {log}', flush=True)
                continue
            seconds = sum(timings[k] for k in PHASES)
            match = re.search(r'Psum=([0-9.eE+-]+)', stdout)
            if match is None or abs(float(match.group(1)) - 1.0) > 1e-5:
                print(f'  WARNING: unexpected probability sum; see {log}', flush=True)
            row = dict(dataset=ds, mode=mode, threads=threads, repeat=repeat,
                       seconds=seconds, wall_seconds=wall, **{p: timings[p] for p in PHASES})
            records.append(row)
            completed.add(key)
            write_csv(raw_path, fields, records)  # Save progress after every run.
            print(f'  compute={seconds:.3f}s wall={wall:.3f}s', flush=True)

    # Direct O(n^2) vs Barnes-Hut on sampled points from the smallest graph.
    bh_path = OUT / 'bh_vs_exact.csv'
    bh = load_csv(bh_path)
    bh_fields = ['n', 'theta', 'threads', 'bh_walk', 'tree', 'com', 'exact',
                 'speedup_exact_over_bh_walk', 'Frep_rel_error', 'Z_rel_error']
    done = {(int(r['n']), float(r['theta'])) for r in bh}
    sweep = [(n, 0.5) for n in (500, 1000, 2000, 4000, 8000)]
    sweep += [(2000, theta) for theta in (0.1, 0.3, 0.8)]
    for n, theta in sweep:
        if (n, theta) in done:
            continue
        print(f'[EXACT] n={n} theta={theta}', flush=True)
        cmd = [str(ROOT/'build'/'tsne_omp'), 'data/k49_norm_10NN', '--threads', '4',
               '--theta', str(theta), '--bench-rep', str(n)]
        code, output, _ = run(cmd, log=LOGS/f'exact_{n}_{theta}.txt')
        if code:
            print(f'  ERROR (exit {code}); see exact log', flush=True)
            continue
        match = re.search(r'BENCH n=(\d+) tree=([\d.eE+-]+) com=([\d.eE+-]+) '
                          r'BH_walk=([\d.eE+-]+) exact=([\d.eE+-]+) '
                          r'speedup\(exact/BH_walk\)=([\d.eE+-]+) '
                          r'Frep_rel_error=([\d.eE+-]+) Z_rel_error=([\d.eE+-]+)', output)
        if not match:
            print('  WARNING: cannot parse exact benchmark; see log', flush=True)
            continue
        _, tree, com, walk, exact, speed, errF, errZ = match.groups()
        bh.append(dict(n=n, theta=theta, threads=4, tree=tree, com=com,
                       bh_walk=walk, exact=exact, speedup_exact_over_bh_walk=speed,
                       Frep_rel_error=errF, Z_rel_error=errZ))
        done.add((n, theta))
        write_csv(bh_path, bh_fields, bh)

    # Median (robust to scheduling noise) and speedup relative to seq.
    grouped = {}
    for r in records:
        k = r['dataset'], r['mode'], int(r['threads'])
        grouped.setdefault(k, []).append(float(r['seconds']))
    medians = {key: statistics.median(times) for key, times in grouped.items()}
    summary = []
    for (ds, mode, threads), times in sorted(grouped.items()):
        base = medians.get((ds, 'seq', 1))
        t = statistics.median(times)
        summary.append(dict(dataset=ds, mode=mode, threads=threads, runs=len(times),
                            median_seconds=round(t, 9), mean_seconds=round(statistics.mean(times), 9),
                            speedup_vs_seq=round(base/t, 4) if base and t>0 else ''))
    write_csv(OUT/'benchmark_summary.csv',
              ['dataset', 'mode', 'threads', 'runs', 'median_seconds', 'mean_seconds', 'speedup_vs_seq'], summary)
    for ds in DATASETS:
        svg_speedup(ds, medians)
    print(f'\nFINISHED. Summary: {OUT / "benchmark_summary.csv"}')
    print(f'Barnes-Hut vs exact: {bh_path}')
    print('Speedup graphs: results/benchmarks/speedup_*.svg')
    print('Logs: results/benchmarks/logs/*.txt')
    if len(completed) < len(targets):
        print('Some runs skipped/failed. Read logs; re-running the script resumes from successful runs.')


if __name__ == '__main__':
    main()
