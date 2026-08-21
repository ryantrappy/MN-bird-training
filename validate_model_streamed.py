#!/usr/bin/env python3
"""Validate the model against 50 random validation image URLs streamed from S3.

This script:
  1) Builds a validation combined CSV using the same logic as combine_gbif.py.
  2) Filters to StillImage URLs in the inaturalist-open-data S3 bucket.
  3) Randomly selects 50 samples.
  4) Streams each image in memory and runs ONNX inference.
  5) Marks a sample correct if the ground-truth scientific name appears in the top-2
     predictions with confidence >= 0.50.
  6) Prints the overall top-2 accuracy percentage.
"""

import argparse
import csv
import io
import os
import random
import sys

import numpy as np
import onnxruntime as ort
import requests
from PIL import Image

import combine_gbif


def safe_label(s):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(s))[:200]


def load_labels(path):
    with open(path, 'r', encoding='utf8') as f:
        return [line.strip() for line in f if line.strip()]


def build_validation_combined(validation_dir, out_path=None):
    verbatim_path = os.path.join(validation_dir, 'verbatim.txt')
    multimedia_path = os.path.join(validation_dir, 'multimedia.txt')
    out_path = out_path or os.path.join(validation_dir, 'combined.csv')
    verb_map = combine_gbif.load_verbatim(verbatim_path)
    combine_gbif.stream_multimedia_and_write(multimedia_path, verb_map, out_path)
    return out_path


def load_known_classes(path='classes.txt'):
    if not os.path.exists(path):
        return set()
    with open(path, 'r', encoding='utf8') as f:
        return {line.strip() for line in f if line.strip()}


def filter_known_class_rows(rows, classes_path='classes.txt'):
    known = load_known_classes(classes_path)
    filtered = []
    for row in rows:
        scientific = (row.get('scientificName') or '').strip()
        if not scientific:
            continue
        if safe_label(scientific) in known:
            filtered.append(row)
    return filtered


def is_still_image_row(row):
    url = (row.get('identifier') or row.get('imageLink') or '').strip()
    if not url:
        return False
    lower_url = url.lower()
    if 'inaturalist-open-data.s3.amazonaws.com' not in lower_url:
        return False
    media_type = (row.get('type') or row.get('format') or '').strip().lower()
    if media_type and 'still' in media_type:
        return True
    return any(lower_url.endswith(ext) for ext in ('.jpg', '.jpeg', '.png', '.gif', '.tif', '.tiff', '.bmp', '.webp'))


def preprocess_image(image):
    img = image.convert('RGB').resize((224, 224))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = x.transpose(2, 0, 1)[None, ...]
    return x.astype(np.float32)


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def model_predict(sess, url, labels):
    # stream image in memory only
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    image = Image.open(io.BytesIO(resp.content))
    tensor = preprocess_image(image)
    input_name = sess.get_inputs()[0].name
    logits = sess.run(None, {input_name: tensor})[0][0]
    probs = softmax(logits)
    top2_idx = np.argsort(probs)[::-1][:2]
    top2 = []
    for idx in top2_idx:
        label_name = labels[int(idx)]
        top2.append((label_name, float(probs[int(idx)])))
    return top2


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--validation-dir', default='validation_data')
    p.add_argument('--model', default='model.onnx')
    p.add_argument('--labels', default='labels.txt')
    p.add_argument('--sample-size', type=int, default=50)
    p.add_argument('--class-file', default='classes.txt', help='Only include scientific names present in this class list')
    p.add_argument('--seed', type=int, default=None, help='Optional random seed; defaults to a different sample each run')
    args = p.parse_args()

    combined_path = build_validation_combined(args.validation_dir)

    rows = []
    with open(combined_path, 'r', encoding='utf8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scientific = (row.get('scientificName') or '').strip()
            url = (row.get('identifier') or '').strip()
            if not scientific or not url:
                continue
            if is_still_image_row(row):
                rows.append({
                    'scientificName': scientific,
                    'identifier': url,
                })

    rows = filter_known_class_rows(rows, args.class_file)
    if len(rows) < args.sample_size:
        raise SystemExit(f'Only {len(rows)} valid S3 StillImage rows found in {args.class_file}; need at least {args.sample_size}.')

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    selected = rows[:args.sample_size]

    labels = load_labels(args.labels)
    sess = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])

    correct = 0
    top2_any = 0
    seen = 0
    for row in selected:
        true_label = safe_label(row['scientificName'])
        try:
            top2 = model_predict(sess, row['identifier'], labels)
        except Exception as exc:
            print(f'Skipping invalid sample {row["identifier"]}: {exc}', file=sys.stderr)
            continue
        top2_over_50 = [(safe_label(name), prob) for name, prob in top2 if prob >= 0.50]
        in_top2 = true_label in {name for name, _ in top2}
        is_correct = true_label in {name for name, _ in top2_over_50}
        if is_correct:
            correct += 1
        if in_top2:
            top2_any += 1
        seen += 1

        guess1 = top2[0][0] if len(top2) > 0 else 'N/A'
        guess2 = top2[1][0] if len(top2) > 1 else 'N/A'
        guess1_prob = top2[0][1] if len(top2) > 0 else 0.0
        guess2_prob = top2[1][1] if len(top2) > 1 else 0.0
        if is_correct:
            marker = '✅'
            result = 'correct'
        elif in_top2:
            marker = '⚠️'
            result = 'top2-but-under-50pct'
        else:
            marker = '❌'
            result = 'incorrect'
        suffix = f" | {row['identifier']}" if result == 'incorrect' else ''
        print(f"{marker} actual={row['scientificName']} | top2={guess1} ({guess1_prob:.2%}) / {guess2} ({guess2_prob:.2%}) | result={result}{suffix}")

    if seen == 0:
        raise SystemExit('No valid validation samples processed.')

    pct_correct = (correct / seen) * 100.0
    pct_top2_any = (top2_any / seen) * 100.0
    print(f'Validated {seen} samples from {combined_path}')
    print(f'Correct in top-2 with confidence >= 0.50: {correct}/{seen} ({pct_correct:.2f}%)')
    print(f'Correct in top-2 with confidence: {top2_any}/{seen} ({pct_top2_any:.2f}%)')


if __name__ == '__main__':
    main()
