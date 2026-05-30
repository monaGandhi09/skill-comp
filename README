# README

## Overview

`comp_test.py` and `decomp_test.py` evaluate VL models on compositional integration using positive captions and structurally controlled hard negatives.

Supported models:

| Argument   | Model             |
| ---------- | ----------------- |
| `clip`     | OpenCLIP ViT-g-14 |
| `siglipv2` | Google SigLIP2    |
| `peclip`   | PE-CLIP           |
| `blip`     | BLIP ITM Large    |
| `negclip`  | NegCLIP           |
| `ceclip`   | CE-CLIP           |

The scripts support three structural evaluation settings:

| Structural Level | Description                           |
| ---------------- | ------------------------------------- |
| `l3`             | Object-Attribute-Relation composition |
| `l2_or`          | Object-Relation composition           |
| `l2_oa`          | Object-Attribute composition          |

---

# Qwen3-VL-Embedding Evaluation

Qwen3-VL-Embedding uses separate evaluation scripts because it operates as a retrieval-oriented dual-encoder embedding model.

The following scripts are provided specifically for Qwen3:

| Script                 | Purpose                                                    |
| ---------------------- | ---------------------------------------------------------- |
| `qwen3_comp_test.py`   | Compositional evaluation using structured hard negatives   |
| `qwen3_decomp_test.py` | Decompositional evaluation using component-level negatives |

Unlike the other models, Qwen3-VL-Embedding:

* embeds images and captions independently
* performs batched two-tower retrieval evaluation
* computes cosine similarity directly in embedding space
* supports Flash Attention 2 and BF16 inference
* is optimized for large-scale retrieval benchmarking

The Qwen3 scripts mirror the interfaces and output formats of `comp_test.py` and `decomp_test.py` so results remain directly comparable across models.

---

# Installation

## Base Dependencies

```bash
pip install torch torchvision pillow tqdm numpy open_clip_torch transformers
```

## Additional Dependencies for Qwen3-VL-Embedding

```bash
pip install accelerate sentencepiece einops python-dotenv qwen-vl-utils
```

Optional (recommended for faster inference):

```bash
pip install flash-attn
```

Clone the official Qwen3-VL-Embedding repository:

```bash
git clone https://github.com/QwenLM/Qwen3-VL-Embedding
```

---

# Dataset Setup

Download the dataset from HuggingFace:

```bash
git clone https://huggingface.co/datasets/Anon-compass/COMPASS
```

The dataset should contain:

```text
gt-caption/l3
gt-caption/l2-OR
gt-caption/l2-OA

compositional-integration/l3/composed
compositional-integration/l2-OR/composed
compositional-integration/l2-OA/composed

compositional-integration/l3/decomposed
compositional-integration/l2-OR/decomposed
compositional-integration/l2-OA/decomposed
```

Update the image root path after importing the Visual Genome Dataset:

https://homes.cs.washington.edu/~ranjay/visualgenome/api.html

```python
IMAGE_PATH = "VISUAL_GENOME/images"
```

---

# Model Setup

| Model                | Setup                                                                       |
| -------------------- | --------------------------------------------------------------------------- |
| `clip`               | Automatically loaded from OpenCLIP                                          |
| `siglipv2`           | Automatically loaded from HuggingFace                                       |
| `peclip`             | Requires PE-CLIP dependencies                                               |
| `blip`               | Automatically loaded from HuggingFace                                       |
| `negclip`            | Automatically downloads checkpoints                                         |
| `ceclip`             | Automatically downloads checkpoints                                         |
| `Qwen3-VL-Embedding` | Requires the official Qwen3-VL-Embedding repository and HuggingFace weights |

---

# 1. Compositional Evaluation (`comp_test.py`)

## Purpose

Evaluates retrieval performance using:

* one ground-truth caption
* structurally matched hard negatives

For each image:

1. the GT caption is randomly inserted among negatives
2. the model computes similarity scores
3. scores and retrieval results are saved as JSON

---

## Run

### Basic Usage

```bash
python comp_test.py \
    --model clip \
    --structural-level l3
```

### Example Variants

```bash
python comp_test.py --model siglipv2 --structural-level l2_or
```

```bash
python comp_test.py --model negclip --structural-level l2_oa
```

---

# 2. Decompositional Evaluation (`decomp_test.py`)

## Purpose

Evaluates fine-grained compositional understanding using decomposed caption sets.

Each sample contains:

* a GT caption
* multiple structured negatives
* the replaced compositional component

This allows analysis of which compositional elements models fail to recognize.

---

## Run

### Basic Usage

```bash
python decomp_test.py \
    --model clip \
    --structural-level l3
```

### Example Variants

```bash
python decomp_test.py --model peclip --structural-level l2_or
```

```bash
python decomp_test.py --model ceclip --structural-level l2_oa
```

---

# 3. Qwen3 Compositional Evaluation (`qwen3_comp_test.py`)

## Purpose

Evaluates Qwen3-VL-Embedding on compositional retrieval tasks using the same hard-negative setup as `comp_test.py`.

The implementation uses:

* batched image embedding
* batched caption embedding
* cosine similarity in embedding space
* two-tower retrieval evaluation

This reduces GPU forward passes compared to pairwise scoring pipelines and is optimized for retrieval evaluation.

---

## Run

### Basic Usage

```bash
python qwen3_comp_test.py \
    --structural-level l3 \
    --outfolder results_l3
```

### Flash Attention + BF16

```bash
python qwen3_comp_test.py \
    --structural-level l3 \
    --outfolder results_l3 \
    --flash-attn \
    --bf16
```

### Process a File Range

```bash
python qwen3_comp_test.py \
    --structural-level l2_or \
    --start 0 \
    --end 1000 \
    --batch-size 16 \
    --outfolder results_or
```

---

## Arguments

| Argument                    | Description                    |
| --------------------------- | ------------------------------ |
| `-sl`, `--structural-level` | `l3`, `l2_or`, or `l2_oa`      |
| `-o`, `--outfolder`         | Output directory               |
| `-s`, `--start`             | Start index                    |
| `-e`, `--end`               | End index                      |
| `--batch-size`              | Batch size for embedding       |
| `--flash-attn`              | Enable Flash Attention 2       |
| `--bf16`                    | Enable bfloat16 inference      |
| `--max-length`              | Maximum token length           |
| `--query-instruction`       | Optional retrieval instruction |

---

# 4. Qwen3 Decompositional Evaluation (`qwen3_decomp_test.py`)

## Purpose

Evaluates Qwen3-VL-Embedding on decomposed compositional negatives.

The script measures whether the model can correctly distinguish:

* object substitutions
* attribute substitutions
* relation substitutions

using embedding-space retrieval.

---

## Run

### Basic Usage

```bash
python qwen3_decomp_test.py \
    --structural-level l3 \
    --outfolder results_decomp
```

### Filter by Negative Type

```bash
python qwen3_decomp_test.py \
    --structural-level l3 \
    --type relation \
    --outfolder relation_results
```

### Large Batch Inference

```bash
python qwen3_decomp_test.py \
    --structural-level l2_oa \
    --batch-size 32 \
    --bf16 \
    --flash-attn \
    --outfolder oa_results
```

---

## Arguments

| Argument                    | Description                                 |
| --------------------------- | ------------------------------------------- |
| `-sl`, `--structural-level` | `l3`, `l2_or`, or `l2_oa`                   |
| `-t`, `--type`              | `object`, `relation`, `attribute`, or `all` |
| `-o`, `--outfolder`         | Output directory                            |
| `--batch-size`              | Batch size                                  |
| `--flash-attn`              | Enable Flash Attention 2                    |
| `--bf16`                    | Enable bfloat16 inference                   |
| `--query-instruction`       | Optional retrieval instruction              |

---

# Notes

* CUDA is used automatically when available.
* Existing output files are skipped automatically.
* NegCLIP and CE-CLIP checkpoints are downloaded automatically if missing.
* Qwen3-VL-Embedding evaluation uses a two-tower batched embedding pipeline.
* Flash Attention 2 and BF16 are recommended for Qwen3 evaluation on modern GPUs.