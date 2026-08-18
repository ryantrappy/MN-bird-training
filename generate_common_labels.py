#!/usr/bin/env python3
"""
Generate common_labels.json mapping scientificName -> common English name using iNaturalist API.

Usage:
  python generate_common_labels.py --csv combined.csv --out common_labels.json

Options:
  --max N        Limit to first N unique scientific names (0 = unlimited)
  --sleep S      Seconds to wait between requests (default 0.2)
  --fallback     Use title-cased fallback when no common name is found
"""
import argparse
import json
import time
import requests
import pandas as pd
from urllib.parse import quote_plus


def fetch_common(scientific_name, session, timeout=10):
    q = quote_plus(scientific_name)
    url = f"https://api.inaturalist.org/v1/taxa?q={q}&per_page=1"
    try:
        r = session.get(url, timeout=timeout)
        if r.status_code != 200:
            return None
        j = r.json()
        res = j.get('results') or j.get('data') or []
        if not res:
            return None
        first = res[0]
        # iNaturalist uses 'preferred_common_name'
        name = first.get('preferred_common_name') or first.get('english_common_name')
        if name:
            return name
    except Exception:
        return None
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--out', default='common_labels.json')
    p.add_argument('--label-col', default='scientificName')
    p.add_argument('--max', type=int, default=0, help='Limit to first N unique names (0=all)')
    p.add_argument('--sleep', type=float, default=0.2)
    p.add_argument('--fallback', action='store_true', help='Use title-cased fallback when no common name found')
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    col = args.label_col
    if col not in df.columns:
        candidates = [c for c in df.columns if 'scientific' in c.lower() or 'name' in c.lower()]
        if candidates:
            col = candidates[0]
            print(f"Using detected label column: {col}")
        else:
            raise SystemExit(f"Label column '{args.label_col}' not found")

    names = df[col].dropna().astype(str).str.strip()
    unique = []
    seen = set()
    for n in names:
        if n == '':
            continue
        if n in seen:
            continue
        seen.add(n)
        unique.append(n)
        if args.max and len(unique) >= args.max:
            break

    print(f'Found {len(unique)} unique scientific names to query')
    session = requests.Session()
    out = {}
    for i, sname in enumerate(unique, 1):
        common = fetch_common(sname, session)
        if not common and args.fallback:
            common = ' '.join([p.capitalize() for p in sname.replace('_',' ').split()])
        out[sname] = common or ""
        if i % 10 == 0:
            print(f'[{i}/{len(unique)}] {sname} -> {out[sname]}')
        time.sleep(args.sleep)

    with open(args.out, 'w', encoding='utf8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'Wrote {len(out)} entries to {args.out}')

if __name__ == '__main__':
    main()
