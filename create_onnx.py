#!/usr/bin/env python3
"""
Create and export an ONNX image classifier from a CSV of image URLs and labels.
Supports two modes:
  - Disk mode (default): downloads images into data-dir/<label>/ and uses ImageFolder
  - Streaming mode (--stream): does NOT save files; images are downloaded and cached in memory

Usage examples:
  python create_onnx.py --csv combined.csv --out model.onnx --epochs 3
  python create_onnx.py --csv combined.csv --stream --test-entries 50 --out model.onnx

Dependencies:
  pip install torch torchvision pandas requests pillow tqdm onnx onnxruntime
"""

import argparse
import hashlib
import io
import os
import pathlib
import sys
from collections import OrderedDict
from urllib.parse import urlparse

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, models, transforms

# async downloader for fast S3 downloads
import asyncio
import aiohttp
import aiofiles
import tempfile
import os
from tqdm import tqdm


def safe_label(s):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)[:200]


# persistent download & API usage tracker and rate limiter
import json
import time

USAGE_PATH = '.download_usage.json'

class DownloadManager:
    def __init__(self, api_per_min=60, api_per_day=10000, media_hour_bytes=5*(1024**3), media_day_bytes=24*(1024**3)):
        self.api_per_min = api_per_min
        self.api_per_day = api_per_day
        self.media_hour_bytes = media_hour_bytes
        self.media_day_bytes = media_day_bytes
        self.session = requests.Session()
        self._load_usage()

    def _load_usage(self):
        try:
            with open(USAGE_PATH, 'r', encoding='utf8') as f:
                self.usage = json.load(f)
        except Exception:
            self.usage = {'api_calls': [], 'media': []}
        # normalize lists
        self.usage.setdefault('api_calls', [])
        self.usage.setdefault('media', [])
        self._cleanup()

    def _save_usage(self):
        try:
            with open(USAGE_PATH, 'w', encoding='utf8') as f:
                json.dump(self.usage, f)
        except Exception:
            pass

    def _cleanup(self):
        now = time.time()
        day_ago = now - 86400
        hour_ago = now - 3600
        self.usage['api_calls'] = [t for t in self.usage.get('api_calls', []) if t >= day_ago]
        self.usage['media'] = [m for m in self.usage.get('media', []) if m[0] >= day_ago]
        # keep media entries only last day (we'll filter by hour/day when needed)
        self._save_usage()

    def is_exempt_url(self, url: str) -> bool:
        """Return True for URLs that are hosted in the inaturalist-open-data S3 bucket.
        This allows bypassing any rate limiting for those URLs.
        """
        try:
            p = urlparse(url)
            netloc = (p.netloc or '').lower()
            path = (p.path or '').lower()
            # common patterns: inaturalist-open-data.s3.amazonaws.com or s3.amazonaws.com/.../inaturalist-open-data
            if 'inaturalist-open-data' in netloc or '/inaturalist-open-data/' in path or 'inaturalist-open-data' in path:
                return True
        except Exception:
            pass
        return False

    def _count_api_since(self, seconds):
        cutoff = time.time() - seconds
        return sum(1 for t in self.usage.get('api_calls', []) if t >= cutoff)

    def _bytes_media_since(self, seconds):
        cutoff = time.time() - seconds
        return sum(m[1] for m in self.usage.get('media', []) if m[0] >= cutoff)

    def record_api(self):
        self.usage.setdefault('api_calls', []).append(time.time())
        self._save_usage()

    def record_media(self, bytes_count):
        self.usage.setdefault('media', []).append([time.time(), int(bytes_count)])
        self._save_usage()

    def wait_for_api_slot(self):
        # ensure not exceeding per-minute and per-day
        while True:
            per_min = self._count_api_since(60)
            per_day = self._count_api_since(86400)
            if per_min < self.api_per_min and per_day < self.api_per_day:
                return
            # sleep until earliest api call falls out of window
            now = time.time()
            if per_min >= self.api_per_min:
                # earliest call within last minute
                oldest = min(t for t in self.usage.get('api_calls', []) if t >= now-60)
                wait = (oldest + 60) - now + 0.1
            else:
                oldest = min(t for t in self.usage.get('api_calls', []) if t >= now-86400)
                wait = (oldest + 86400) - now + 1
            if wait < 0:
                wait = 0.5
            time.sleep(wait)
            self._load_usage()

    def wait_for_media_space(self, expected_bytes=0):
        # ensure hourly and daily media bytes under limits
        while True:
            bytes_hour = self._bytes_media_since(3600)
            bytes_day = self._bytes_media_since(86400)
            if bytes_hour + expected_bytes <= self.media_hour_bytes and bytes_day + expected_bytes <= self.media_day_bytes:
                return
            # compute time to wait until enough bytes expire (look at media list)
            now = time.time()
            # find earliest media entry within last day that if removed frees space
            hour_cutoff = now - 3600
            day_cutoff = now - 86400
            # find oldest that is within hour/day windows
            wait_candidates = []
            for ts, b in self.usage.get('media', []):
                if ts < hour_cutoff:
                    # already expired for hour
                    wait_candidates.append(0)
                else:
                    wait_candidates.append((ts + 3600) - now)
                if ts < day_cutoff:
                    wait_candidates.append(0)
                else:
                    wait_candidates.append((ts + 86400) - now)
            # default small sleep
            wait = max(1.0, min([w for w in wait_candidates if w > 0], default=5.0))
            time.sleep(wait)
            self._load_usage()

# singleton manager (can be re-created in main)
DM = DownloadManager()


def download_images_to_disk(csv_path, out_dir, url_col="imageLink", label_col="scientificName", test_entries=0, timeout=10, force=False, concurrency=64):
    df = pd.read_csv(csv_path)
    # auto-detect url/label columns if not provided
    if url_col not in df.columns or label_col not in df.columns:
        candidates_url = [c for c in df.columns if "image" in c.lower() or 'identifier' in c.lower()]
        candidates_label = [c for c in df.columns if "scientific" in c.lower() or "name" in c.lower()]
        if candidates_url:
            url_col = candidates_url[0]
        if candidates_label:
            label_col = candidates_label[0]
    # detect format column if present
    format_col = next((c for c in df.columns if 'format' in c.lower() or '`format`' in c.lower()), None)

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # collect entries to download concurrently
    entries = []
    records = df.to_dict('records')
    for r in records:
        url = r.get(url_col) or r.get('identifier') or ''
        label = r.get(label_col, 'unknown')
        if pd.isna(url) or str(url).strip() in ("", "nan", "None"):
            continue
        url = str(url)
        # only download images hosted on the inaturalist-open-data S3 bucket
        is_exempt = DM.is_exempt_url(url) if hasattr(DM, 'is_exempt_url') else False
        if not is_exempt:
            # per request: do not download non-open-data images at all
            continue
        # ensure media type is StillImage (use format/type column if available, else fallback to extension)
        media_type = (r.get('type') or r.get('Type') or (r.get(format_col) if format_col else '') or '').strip()
        def is_still_image(media_type_str, url_str):
            if media_type_str:
                t = media_type_str.lower()
                return 'still' in t or 'image' in t
            # fallback: check extension
            l = url_str.lower()
            return any(l.endswith(ext) for ext in ('.jpg', '.jpeg', '.png', '.gif', '.tif', '.tiff', '.bmp', '.webp'))
        if not is_still_image(media_type, url):
            continue

        label = safe_label(str(label if not pd.isna(label) else "unknown"))
        label_dir = out_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        suffix = pathlib.Path(url).suffix.split('?')[0] or '.jpg'
        fname = hashlib.sha1(url.encode('utf8')).hexdigest()[:16] + suffix
        path = label_dir / fname
        if path.exists() and not force:
            continue
        entries.append((url, path))
        if test_entries and len(entries) >= test_entries:
            break

    # async downloader
    async def _fetch(session, sem, url, path, timeout, retries=3):
        backoff = 1.0
        for attempt in range(retries):
            try:
                async with sem:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                        if resp.status == 200:
                            data = await resp.read()
                            tmp = str(path) + '.tmp'
                            try:
                                async with aiofiles.open(tmp, 'wb') as f:
                                    await f.write(data)
                                os.replace(tmp, str(path))
                                return True
                            finally:
                                # cleanup tmp if exists
                                try:
                                    if os.path.exists(tmp):
                                        os.remove(tmp)
                                except Exception:
                                    pass
                        else:
                            # non-200, treat as failure to allow retry
                            pass
            except Exception:
                await asyncio.sleep(backoff)
                backoff *= 2
        return False

    async def _download_all(entries, concurrency, timeout):
        if not entries:
            return 0
        connector = aiohttp.TCPConnector(limit_per_host=concurrency)
        sem = asyncio.Semaphore(concurrency)
        success = 0
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [asyncio.create_task(_fetch(session, sem, url, path, timeout)) for url, path in entries]
            for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc='downloading'):
                res = await f
                if res:
                    success += 1
        return success

    # run event loop
    try:
        downloaded = asyncio.run(_download_all(entries, concurrency=64, timeout=timeout))
    except RuntimeError:
        # already running event loop (rare in some environments), fall back to synchronous download
        downloaded = 0
        for url, path in tqdm(entries, desc='download (fallback)'):
            try:
                resp = DM.session.get(url, timeout=timeout)
                if resp.status_code == 200:
                    content = resp.content
                    try:
                        img = Image.open(io.BytesIO(content)).convert('RGB')
                        img.save(path)
                        downloaded += 1
                    except Exception:
                        continue
            except Exception:
                continue

    print(f"Downloaded approx {downloaded} images to {out_dir}")


class StreamingImageDataset(Dataset):
    """CSV-driven dataset that downloads images on-demand and caches tensors in memory.
    Respects API and media rate limits via the global DM DownloadManager.
    """

    def __init__(self, csv_path, url_col="imageLink", label_col="scientificName", transform=None, test_entries=0, cache_size=1000, timeout=10):
        df = pd.read_csv(csv_path)
        # auto-detect columns if needed
        if url_col not in df.columns or label_col not in df.columns:
            candidates_url = [c for c in df.columns if "image" in c.lower() or 'identifier' in c.lower()]
            candidates_label = [c for c in df.columns if "scientific" in c.lower() or "name" in c.lower()]
            if candidates_url:
                url_col = candidates_url[0]
            if candidates_label:
                label_col = candidates_label[0]
        # detect format column if present
        format_col = next((c for c in df.columns if 'format' in c.lower() or '`format`' in c.lower()), None)

        self.transform = transform
        self.rows = []
        records = df.to_dict('records')
        for i, row in enumerate(records):
            if test_entries and i >= test_entries:
                break
            url = row.get(url_col) or row.get('identifier') or ''
            label = row.get(label_col, 'unknown')
            if pd.isna(url) or str(url).strip() in ('', 'nan', 'None'):
                continue
            url = str(url)
            # only include inaturalist-open-data S3 URLs per request
            is_exempt = DM.is_exempt_url(url) if hasattr(DM, 'is_exempt_url') else False
            if not is_exempt:
                continue
            # check media format/type
            media_type = (row.get('type') or row.get('Type') or (row.get(format_col) if format_col else '') or '').strip()
            def is_still_image(media_type_str, url_str):
                if media_type_str:
                    t = media_type_str.lower()
                    return 'still' in t or 'image' in t
                l = url_str.lower()
                return any(l.endswith(ext) for ext in ('.jpg', '.jpeg', '.png', '.gif', '.tif', '.tiff', '.bmp', '.webp'))
            if not is_still_image(media_type, url):
                continue

            self.rows.append((url, safe_label(str(label if not pd.isna(label) else 'unknown'))))
        # map labels to idx
        labels = sorted({l for _, l in self.rows})
        self.class_to_idx = {c: i for i, c in enumerate(labels)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        # in-memory LRU cache of tensors
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.timeout = timeout

    def __len__(self):
        return len(self.rows)

    def _download_and_transform(self, url):
        try:
            # skip rate limits for inaturalist-open-data S3
            is_exempt = DM.is_exempt_url(url) if hasattr(DM, 'is_exempt_url') else False
            if not is_exempt:
                DM.wait_for_api_slot()
            resp = DM.session.get(url, timeout=self.timeout)
            if not is_exempt:
                DM.record_api()
            if resp.status_code == 200:
                content = resp.content
                # enforce media quotas for non-exempt URLs
                if not is_exempt:
                    DM.wait_for_media_space(len(content))
                try:
                    img = Image.open(io.BytesIO(content)).convert('RGB')
                    if self.transform:
                        tensor = self.transform(img)
                    else:
                        tensor = transforms.ToTensor()(img)
                    if not is_exempt:
                        DM.record_media(len(content))
                    return tensor
                except Exception:
                    return torch.zeros(3, 224, 224)
        except Exception:
            pass
        # return a zero tensor fallback
        return torch.zeros(3, 224, 224)

    def __getitem__(self, idx):
        url, label = self.rows[idx]
        # check cache
        if url in self.cache:
            tensor = self.cache.pop(url)
            self.cache[url] = tensor
        else:
            tensor = self._download_and_transform(url)
            self.cache[url] = tensor
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        target = self.class_to_idx[label]
        return tensor, target


def train_and_export_from_folder(data_dir, out_onnx, epochs=3, batch_size=32, lr=1e-3, input_size=224):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tfms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    dataset = datasets.ImageFolder(data_dir, transform=tfms)
    num_classes = len(dataset.classes)
    if num_classes == 0:
        raise SystemExit('No images found in dataset.')
    val_pct = 0.2
    val_len = int(len(dataset) * val_pct)
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])
    num_workers = 4
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in tqdm(train_loader, desc=f'train {epoch+1}/{epochs}'):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running += loss.item()
        # val
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                preds = out.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        acc = correct / total if total else 0
        print(f"Epoch {epoch+1}: train_loss_avg={running/len(train_loader):.4f} val_acc={acc:.4f}")

    model.eval()
    dummy = torch.randn(1,3,input_size,input_size, device=device)
    torch.onnx.export(model, dummy, out_onnx, opset_version=11, input_names=['input'], output_names=['output'], dynamic_axes={'input':{0:'batch_size'}, 'output':{0:'batch_size'}})
    print('Exported', out_onnx)


def train_and_export_streaming(csv_path, out_onnx, epochs=3, batch_size=32, lr=1e-3, input_size=224, test_entries=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tfms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    dataset = StreamingImageDataset(csv_path, transform=tfms, test_entries=test_entries)
    num_classes = len(dataset.class_to_idx)
    if num_classes == 0:
        raise SystemExit('No images/labels found in CSV for streaming mode.')
    # simple split
    val_pct = 0.2
    val_len = int(len(dataset) * val_pct)
    train_len = len(dataset) - val_len
    if train_len <= 0:
        raise SystemExit('Not enough entries for train/val split; reduce --test-entries or provide more data')
    train_ds, val_ds = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in tqdm(train_loader, desc=f'train {epoch+1}/{epochs}'):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running += loss.item()
        # val
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                preds = out.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        acc = correct / total if total else 0
        print(f"Epoch {epoch+1}: train_loss_avg={running/len(train_loader):.4f} val_acc={acc:.4f}")

    model.eval()
    dummy = torch.randn(1,3,input_size,input_size, device=device)
    torch.onnx.export(model, dummy, out_onnx, opset_version=11, input_names=['input'], output_names=['output'], dynamic_axes={'input':{0:'batch_size'}, 'output':{0:'batch_size'}})
    print('Exported', out_onnx)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--out', default='model.onnx')
    p.add_argument('--data-dir', default='data_images')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--download-only', action='store_true')
    p.add_argument('--stream', action='store_true', help='Do not save images to disk; stream and cache in memory')
    p.add_argument('--test-entries', type=int, default=0, help='If >0 limit to N CSV rows for a quick test (use 50 for quick run)')
    p.add_argument('--url-col', default='imageLink')
    p.add_argument('--label-col', default='scientificName')
    # rate limit configuration
    p.add_argument('--api-per-minute', type=int, default=60, help='Max API requests per minute (default 60)')
    p.add_argument('--api-per-day', type=int, default=10000, help='Max API requests per day (default 10000)')
    p.add_argument('--media-hourly-gb', type=float, default=5.0, help='Max media download GB per hour (default 5)')
    p.add_argument('--media-daily-gb', type=float, default=24.0, help='Max media download GB per day (default 24)')
    p.add_argument('--concurrency', type=int, default=64, help='Concurrent downloads for S3 (default 64)')

    args = p.parse_args()

    # recreate global DownloadManager with configured limits
    global DM
    DM = DownloadManager(api_per_min=args.api_per_minute, api_per_day=args.api_per_day,
                         media_hour_bytes=int(args.media_hourly_gb * 1024**3),
                         media_day_bytes=int(args.media_daily_gb * 1024**3))

    if args.stream:
        print('Streaming mode: images will not be saved to disk')
        if args.download_only:
            print('download-only has no effect with --stream')
        train_and_export_streaming(args.csv, args.out, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, input_size=224, test_entries=args.test_entries)
        return

    # disk mode
    if args.test_entries:
        print(f'Disk mode: will download up to {args.test_entries} entries')
    download_images_to_disk(args.csv, args.data_dir, url_col=args.url_col, label_col=args.label_col, test_entries=args.test_entries, timeout=10, force=False, concurrency=args.concurrency)
    if args.download_only:
        print('Download finished. Images at', args.data_dir)
        return
    train_and_export_from_folder(args.data_dir, args.out, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


if __name__ == '__main__':
    main()
