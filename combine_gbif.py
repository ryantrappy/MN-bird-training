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
        # include format field from multimedia.txt (some files call it `format`)
        writer.writerow(["gbifID", "scientificName", "identifier", "format"])
        for row in reader:
            gbif = row.get('gbifID') or row.get('gbifid')
            if not gbif:
                continue
            # fetch format/type fields (multimedia exports vary): check 'type', 'format', '`format`'
            media_type = (row.get('type') or row.get('Type') or row.get('format') or row.get('`format`') or '').strip()
            identifier_field = row.get('identifier', '')

            # helper to decide whether this media should be included: require StillImage or image-like
            def is_still_image(media_type_str, identifier_str):
                if media_type_str:
                    t = media_type_str.lower()
                    return 'still' in t or 'image' in t
                # fallback: check identifier extension for common image types
                id_low = (identifier_str or '').lower()
                for ext in ('.jpg', '.jpeg', '.png', '.gif', '.tif', '.tiff', '.bmp', '.webp'):
                    if id_low.endswith(ext):
                        return True
                return False

            if not identifier_field:
                # if no identifier URL, skip unless we still want to preserve relation; skip per user's request
                continue

            # split identifiers into parts if multiple
            parts = []
            if '|' in identifier_field and ';' in identifier_field:
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
                # only include if StillImage or image-like
                if not is_still_image(media_type, p):
                    continue
                sci = verbatim_map.get(gbif, '')
                writer.writerow([gbif, sci, p, media_type])


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
