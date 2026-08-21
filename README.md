# MN Bird Training

This repository builds and runs a bird-species image classifier using GBIF/iNaturalist-style datasets and exports the final model to ONNX for lightweight inference on CPU.
The current data and model is training on a subset of the MN Bird dataset.

## Goals

- Train a bird recognition model from image URLs and scientific names.
- Normalize and combine GBIF observations into a single CSV dataset.
- Export the trained model to `model.onnx` for deployment or local inference.
- Run inference against a single image using a saved labels file.
- Validate the model against streamed images from the iNaturalist open-data S3 bucket.

The project is built around PyTorch + TorchVision and uses a standard image classifier pipeline with the model exported to ONNX for portability.

## Repository layout

- `combine_gbif.py` — combines `verbatim.txt` and `multimedia.txt` by `gbifID` into a CSV.
- `generate_labels.py` — generates `labels.txt` from the scientific-name column.
- `create_onnx.py` — downloads image data, trains or fine-tunes a model, and exports it as ONNX.
- `run_onnx_inference.py` — runs inference on one image and prints the top predictions.
- `run_onnx_inference_fixed.py` — an alternate inference script, similar to the main inference script.
- `validate_model_streamed.py` — validates the exported model against streamed validation images.
- `generate_common_labels.py` — looks up common English names from the iNaturalist API.
- `requirements.txt` — Python dependencies.

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Note: the PyTorch install for your machine may need a CUDA-specific wheel or CPU-only wheel depending on your environment. The project includes the standard PyTorch/TorchVision package lines, but the exact install may need to be adjusted for your platform.

## Typical workflow

### 1) Prepare the dataset

The scripts expect GBIF export files such as:

- `verbatim.txt`
- `multimedia.txt`

These are often tab-delimited files with scientific names and image identifiers. You can combine them into a single CSV:

```bash
python combine_gbif.py --verbatim verbatim.txt --multimedia multimedia.txt --out combined.csv
```

This creates a CSV with rows including at least:

- `gbifID`
- `scientificName`
- `identifier`
- `format`

### 2) Generate the label list

```bash
python generate_labels.py --csv combined.csv --out labels.txt
```

This writes one label per line, matching the normalized class names used during training.

### 3) Train and export the model

Build the ONNX classifier:

```bash
python create_onnx.py --csv combined.csv --out model.onnx --epochs 3
```

This script can:

- download images to disk under a label directory,
- work in streaming mode without saving all files locally,
- enforce download-rate and media-size limits,
- train a model from the dataset and export it as `model.onnx`.

For the streaming mode:

```bash
python create_onnx.py --csv combined.csv --stream --test-entries 50 --out model.onnx
```

### 4) Run inference on a single image

```bash
python run_onnx_inference.py --model model.onnx --image example.jpg --labels labels.txt --topk 5
```

This outputs the highest-confidence class predictions for the image.

### 5) Validate the model

To run a streamed validation pass against a selection of known-good S3 image URLs:

```bash
python validate_model_streamed.py --validation-dir validation_data --model model.onnx --labels labels.txt --sample-size 50
```

This script:

- builds a validation CSV from the validation directory,
- filters for still-image URLs,
- samples the dataset,
- streams each image in memory,
- evaluates top-2 accuracy with a minimum confidence threshold.

## Optional: generate common names

If you want a lookup of scientific names to common English names:

```bash
python generate_common_labels.py --csv combined.csv --out common_labels.json
```

This calls the iNaturalist API and writes a JSON map like:

```json
{
  "Corvus brachyrhynchos": "American Crow"
}
```

## Notes

- The project is geared toward bird species classification, but the training pipeline is general enough to adapt to other image-classification tasks.
- Many scripts specifically filter for still-image URLs and open-data S3 media to avoid unsupported or non-image sources.
- Model quality depends heavily on the size and cleanliness of your dataset and the number of training epochs.
- Because the training pipeline downloads data from remote sources, use a stable network connection and respect rate limits.

## Expected outputs

By default, a typical run creates files such as:

- `combined.csv`
- `labels.txt`
- `model.onnx`
- `common_labels.json`
- optional validation output and downloaded training data directories

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python combine_gbif.py --verbatim verbatim.txt --multimedia multimedia.txt --out combined.csv
python generate_labels.py --csv combined.csv --out labels.txt
python create_onnx.py --csv combined.csv --out model.onnx --epochs 3
python run_onnx_inference.py --model model.onnx --image example.jpg --labels labels.txt --topk 5
```

This gives you a working end-to-end training and inference flow for the repository.
