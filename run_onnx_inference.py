#!/usr/bin/env python3
import argparse, numpy as np
from PIL import Image
import onnxruntime as ort

MEAN = np.array([0.485,0.456,0.406], dtype=np.float32)
STD  = np.array([0.229,0.224,0.225], dtype=np.float32)

def preprocess(path, size=224):
    img = Image.open(path).convert("RGB").resize((size,size))
    x = np.array(img).astype(np.float32)/255.0
    x = (x - MEAN) / STD
    x = x.transpose(2,0,1)[None, ...]  # NCHW
    return x

def load_labels(path):
    if not path: return None
    with open(path,'r') as f: return [l.strip() for l in f if l.strip()]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--labels", default=None)
    p.add_argument("--topk", type=int, default=3)
    args=p.parse_args()

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    x = preprocess(args.image)
    out = sess.run(None, {input_name: x})[0]
    probs = np.softmax(out[0]) if hasattr(np, 'softmax') else np.exp(out[0]) / np.exp(out[0]).sum()
    top = probs.argsort()[-args.topk:][::-1]
    labels = load_labels(args.labels)
    for i in top:
        name = labels[i] if labels and i < len(labels) else str(i)
        print(f"{name}\t{probs[i]:.4f}")

if __name__=="__main__":
    main()