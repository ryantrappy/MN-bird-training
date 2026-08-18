#!/usr/bin/env python3
"""
Fixed ONNX inference script. Use:
  python run_onnx_inference_fixed.py --model model.onnx --image example.jpg --labels labels.txt --topk 3
"""
import argparse
from PIL import Image
import numpy as np
import onnxruntime as ort

MEAN = np.array([0.485,0.456,0.406], dtype=np.float32)
STD  = np.array([0.229,0.224,0.225], dtype=np.float32)
INPUT_SIZE = 224


def preprocess(path, size=INPUT_SIZE):
    img = Image.open(path).convert('RGB').resize((size, size))
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='ONNX model file')
    p.add_argument('--image', required=True, help='Image file to classify')
    p.add_argument('--labels', default=None, help='labels.txt with one label per line')
    p.add_argument('--topk', type=int, default=3)
    args = p.parse_args()

    sess = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    x = preprocess(args.image).astype(np.float32)
    out = sess.run(None, {input_name: x})[0]
    scores = out[0]
    probs = softmax(scores)
    top_idx = probs.argsort()[-args.topk:][::-1]
    labels = load_labels(args.labels)
    for i in top_idx:
        name = labels[i] if labels and i < len(labels) else str(i)
        print(f"{name}\t{probs[i]:.4f}")

if __name__ == '__main__':
    main()
