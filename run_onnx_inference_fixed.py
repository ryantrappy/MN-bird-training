#!/usr/bin/env python3
"""
Fixed ONNX inference script. Use:
  python run_onnx_inference_fixed.py --model model.onnx --image example.jpg --labels labels.txt --topk 3
"""
import argparse
import io
import json
import os

import numpy as np
import onnxruntime as ort
import requests
from PIL import Image

MEAN = np.array([0.485,0.456,0.406], dtype=np.float32)
STD  = np.array([0.229,0.224,0.225], dtype=np.float32)
INPUT_SIZE = 224


def load_image_from_path(path):
    return Image.open(path).convert('RGB')


def load_image_from_url(url):
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert('RGB')


def preprocess(image, size=INPUT_SIZE):
    img = image.resize((size, size))
    x = np.array(img).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = x.transpose(2,0,1)[None, ...]
    return x


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def load_labels(path):
    if not path:
        return None
    with open(path, 'r', encoding='utf8') as f:
        return [l.strip() for l in f if l.strip()]


def normalize_taxon_name(name):
    s = str(name).strip()
    s = s.replace('×', ' x ')
    s = s.replace('___', ' x ')
    s = s.replace('__', ' ')
    s = s.replace('_', ' ')
    s = ' '.join(s.split())
    return s.lower()


def load_common_labels(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf8') as f:
        raw = json.load(f)
    normalized = {}
    for scientific_name, common_name in raw.items():
        normalized[normalize_taxon_name(scientific_name)] = common_name
    return normalized


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='ONNX model file')
    image_group = p.add_mutually_exclusive_group(required=True)
    image_group.add_argument('--image', help='Local image file to classify')
    image_group.add_argument('--image-url', help='Remote image URL to classify')
    p.add_argument('--labels', default=None, help='labels.txt with one label per line')
    p.add_argument('--common-labels', default='common_labels.json', help='common_labels.json mapping scientific name -> common English name')
    p.add_argument('--topk', type=int, default=3)
    args = p.parse_args()

    model_path = os.path.abspath(args.model)
    if args.image:
        image = load_image_from_path(args.image)
    else:
        image = load_image_from_url(args.image_url)

    # Resolve external-data files relative to the model directory if needed.
    model_dir = os.path.dirname(model_path)
    if model_dir and os.path.isdir(model_dir):
        os.chdir(model_dir)

    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    x = preprocess(image).astype(np.float32)
    out = sess.run(None, {input_name: x})[0]
    scores = out[0]
    probs = softmax(scores)
    top_idx = probs.argsort()[-args.topk:][::-1]
    labels = load_labels(args.labels)
    common_lookup = load_common_labels(args.common_labels)

    for i in top_idx:
        name = labels[i] if labels and i < len(labels) else str(i)
        common_name = common_lookup.get(normalize_taxon_name(name), '')
        if common_name:
            print(f"{name} ({common_name})\t{probs[i]:.4f}")
        else:
            print(f"{name}\t{probs[i]:.4f}")

if __name__ == '__main__':
    main()
