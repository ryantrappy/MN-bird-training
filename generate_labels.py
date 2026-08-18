#!/usr/bin/env python3
"""
Generate labels.txt from a CSV file containing label column (scientificName by default).
Writes one label per line, using the same safe_label normalization as training.

Usage:
  python generate_labels.py --csv combined.csv --out labels.txt
"""
import argparse
import pandas as pd


def safe_label(s):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)[:200]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--out', default='labels.txt')
    p.add_argument('--label-col', default='scientificName')
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    label_col = args.label_col
    if label_col not in df.columns:
        candidates = [c for c in df.columns if 'scientific' in c.lower() or 'name' in c.lower()]
        if candidates:
            label_col = candidates[0]
            print(f"Using detected label column: {label_col}")
        else:
            raise SystemExit(f"Label column '{args.label_col}' not found and no candidate detected in CSV")

    vals = df[label_col].dropna().astype(str).tolist()
    norm = sorted({safe_label(v) for v in vals if v.strip() != ''})
    with open(args.out, 'w', encoding='utf8') as f:
        for v in norm:
            f.write(v + '\n')
    print(f'Wrote {len(norm)} labels to {args.out}')

if __name__ == '__main__':
    main()
