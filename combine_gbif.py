#!/usr/bin/env python3
"""
combine_gbif.py

Usage:
  python combine_gbif.py --verbatim verbatim.txt --multimedia multimedia.txt --out combined.csv

Produces a CSV with columns: gbifID,scientificName,identifier
One output row per multimedia identifier. If a multimedia.identifier contains multiple identifiers
separated by ';' or '|', each is written as its own row.
"""

import argparse
import csv
import sys


def load_verbatim(path):
    mapping = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter='\t')
        # Expect header contains gbifID and scientificName
        for row in reader:
            gbif = row.get('gbifID') or row.get('gbifid')
            if not gbif:
                continue
            sci = row.get('scientificName', '')
            mapping[gbif] = sci
    return mapping


def stream_multimedia_and_write(multimedia_path, verbatim_map, out_path):
    with open(multimedia_path, "r", encoding="utf-8", newline="") as mf, \
         open(out_path, "w", encoding="utf-8", newline="") as of:
        reader = csv.DictReader(mf, delimiter='\t')
        writer = csv.writer(of)
        writer.writerow(["gbifID", "scientificName", "identifier"])
        for row in reader:
            gbif = row.get('gbifID') or row.get('gbifid')
            if not gbif:
                continue
            identifier_field = row.get('identifier', '')
            # If identifier contains multiple values separated by ; or |, split them
            if not identifier_field:
                # still write a row with empty identifier to preserve the relation
                sci = verbatim_map.get(gbif, '')
                writer.writerow([gbif, sci, ''])
                continue
            # Split on common separators, but avoid splitting URLs containing | rarely
            parts = []
            if '|' in identifier_field and ';' in identifier_field:
                # both exist -> split on both
                import re
                parts = re.split(r'[|;]', identifier_field)
            elif '|' in identifier_field:
                parts = identifier_field.split('|')
            elif ';' in identifier_field:
                parts = identifier_field.split(';')
            else:
                parts = [identifier_field]
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                sci = verbatim_map.get(gbif, '')
                writer.writerow([gbif, sci, p])


def parse_args():
    p = argparse.ArgumentParser(description="Combine multimedia.txt and verbatim.txt keyed on gbifID")
    p.add_argument('--verbatim', required=True, help='Path to verbatim.txt (tab-separated)')
    p.add_argument('--multimedia', required=True, help='Path to multimedia.txt (tab-separated)')
    p.add_argument('--out', required=True, help='Output CSV file path')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    try:
        print(f"Loading verbatim file: {args.verbatim}", file=sys.stderr)
        verb_map = load_verbatim(args.verbatim)
        print(f"Verbatim rows loaded: {len(verb_map)}", file=sys.stderr)
        print(f"Streaming multimedia and writing output to: {args.out}", file=sys.stderr)
        stream_multimedia_and_write(args.multimedia, verb_map, args.out)
        print("Done.", file=sys.stderr)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise
