"""
Qwen3-VL-Embedding-8B image-text similarity scoring.

Uses Qwen3VLEmbedder from the official Qwen3-VL-Embedding repository:
  https://github.com/QwenLM/Qwen3-VL-Embedding

The model embeds an image (document side) and each caption (query side) into a
shared embedding space; similarity is computed as the cosine dot product of the
already-normalised embeddings, then softmax-normalised to match the output
format of vllm_test.py.

The evaluation pipeline (get_results / get_accuracy / CLI) mirrors vllm_test.py
exactly, so the two scripts are interchangeable.
"""

import os
from dotenv import load_dotenv

# Load .env before any HuggingFace imports so HF_HOME is respected
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import json
import random
import argparse
import numpy as np
from tqdm import tqdm

from qwen3_vl_embedding import Qwen3VLEmbedder


# ---------------------------------------------------------------------------
# Module-level interface  (drop-in replacement for vllm_test.py / clip_test.py)
# ---------------------------------------------------------------------------

_embedder: Qwen3VLEmbedder | None = None
_query_instruction: str = ""

IMAGE_PATH = "VISUAL_GENOME/images"

def initialize_model(
    model_name_or_path: str,
    max_length: int = 8192,
    query_instruction: str = "",
    **kwargs,
) -> None:
    """
    Initialise the global Qwen3VLEmbedder.  Call once before using scores().

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model ID or local path.
    max_length : int
        Maximum token length for the processor (default 8192).
    query_instruction : str
        Optional instruction prepended to caption queries, e.g.
        "Retrieve images or text relevant to the user's query."
        Leave empty to use the model's default instruction for all inputs.
    **kwargs
        Forwarded to Qwen3VLEmbedder (e.g. torch_dtype, attn_implementation).
    """
    global _embedder, _query_instruction
    _embedder = Qwen3VLEmbedder(
        model_name_or_path=model_name_or_path,
        max_length=max_length,
        **kwargs,
    )
    _query_instruction = query_instruction
    print(f"Initialized Qwen3VLEmbedder: model={model_name_or_path}")


def scores(image_path: str, text_array: list[str]) -> np.ndarray:
    """
    Compute softmax similarity probabilities for an image and a list of captions.

    The image is treated as the document (no instruction); each caption is
    treated as a query (instruction applied when query_instruction is set).

    Returns
    -------
    np.ndarray, shape (len(text_array),), values sum to ~1.
    """
    # Build input list: image first, then captions
    inputs = [{"image": image_path}] + [
        {"text": cap, "instruction": _query_instruction} if _query_instruction
        else {"text": cap}
        for cap in text_array
    ]

    # process() returns L2-normalised embeddings as a torch.Tensor
    embeddings = _embedder.process(inputs)  # (1 + N, dim)

    img_emb  = embeddings[0]    # (dim,)
    txt_embs = embeddings[1:]   # (N, dim)

    # Cosine similarities (vectors already normalised → plain dot product)
    sim = (txt_embs @ img_emb).float().cpu().numpy()  # (N,)

    # Softmax
    sim -= sim.max()
    exp_sim = np.exp(sim)
    return exp_sim / exp_sim.sum()


# ---------------------------------------------------------------------------
# Experiment helpers  (mirrors vllm_test.py)
# ---------------------------------------------------------------------------

def get_results(
    gt_captions: str,
    negative_captions: str,
    outfolder: str,
    s: int = 0,
    e: int = 100,
    batch_size: int = 8,
) -> None:
    """
    Embed images and captions using a two-tower batched approach.

    Instead of one forward pass per (image, complexity) pair (12 per file),
    we embed all images in a batch and all captions in a batch, then compute
    dot products.  This reduces forward passes from 12*N to 2 per batch of N
    files and maximises GPU utilisation.
    """
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    all_files = os.listdir(negative_captions)
    # Skip already-done files up front
    pending = [f for f in all_files[s:e] if not os.path.exists(f"{outfolder}/{f}")]

    for batch_start in tqdm(range(0, len(pending), batch_size), unit="batch"):
        batch_files = pending[batch_start:batch_start + batch_size]

        # ── Load data for this batch ─────────────────────────────────────────
        batch_data = []  # [(file, img_path, [(complexity, negs_with_gt, ind), ...])]
        for file in batch_files:
            image_name = f"{file.split('.')[0]}.jpg"
            img_path   = os.path.join(IMAGE_PATH, image_name)

            try:
                with open(f"{negative_captions}/{file}", "r") as f:
                    negatives = json.load(f)
            except Exception:
                print(f"Error reading negative captions file: {file}")
                continue

            try:
                with open(f"{gt_captions}/{file}", "r") as f:
                    gt = json.load(f)
            except Exception:
                print(f"Error reading ground truth file: {file}")
                continue

            complexities = []
            for complexity, negs in negatives.items():
                caption = gt[complexity]
                if len(negs) == 4:
                    ind = random.randint(1, len(negs))
                    negs = negs.copy()
                    negs.insert(ind, caption)
                    complexities.append((complexity, negs, ind))

            if complexities:
                batch_data.append((file, img_path, complexities))

        if not batch_data:
            continue

        # ── Build inputs for two-tower batched embedding ─────────────────────
        # Image tower: one entry per file
        img_inputs = [{"image": img_path} for _, img_path, _ in batch_data]

        # Caption tower: all complexities of all files, flattened
        cap_inputs = []
        cap_meta   = []  # (file_idx, complexity, negs, ind, cap_offset, n_caps)
        for file_idx, (_, _, complexities) in enumerate(batch_data):
            for complexity, negs, ind in complexities:
                cap_offset = len(cap_inputs)
                for cap in negs:
                    cap_inputs.append(
                        {"text": cap, "instruction": _query_instruction}
                        if _query_instruction else {"text": cap}
                    )
                cap_meta.append((file_idx, complexity, negs, ind, cap_offset, len(negs)))

        # ── Two forward passes for the whole batch ───────────────────────────
        img_embeddings = _embedder.process(img_inputs)   # (B, dim)
        cap_embeddings = _embedder.process(cap_inputs)   # (total_caps, dim)

        # ── Compute similarities and collect per-file results ────────────────
        file_results = [{} for _ in batch_data]
        for file_idx, complexity, negs, ind, cap_offset, n_caps in cap_meta:
            img_emb  = img_embeddings[file_idx]
            txt_embs = cap_embeddings[cap_offset:cap_offset + n_caps]

            sim = (txt_embs @ img_emb).float().cpu().numpy()
            sim -= sim.max()
            exp_sim = np.exp(sim)
            probs = exp_sim / exp_sim.sum()

            file_results[file_idx][complexity] = (negs, probs.tolist(), ind)

        # ── Save ─────────────────────────────────────────────────────────────
        for (file, _, _), results in zip(batch_data, file_results):
            with open(f"{outfolder}/{file}", "w") as f:
                json.dump(results, f, indent=4)


def get_accuracy(
    folder: str,
    k: int = 1,
    s: int = 0,
    e: int = 100,
    cl: int = 1,
    ch: int = 12,
) -> None:
    files = os.listdir(folder)

    complexities = [str(x) for x in range(cl, ch + 1)]
    correct = {c: 0 for c in complexities}
    total   = {c: 0 for c in complexities}

    for file in tqdm(files[s:e]):
        try:
            with open(f"{folder}/{file}", "r") as f:
                results = json.load(f)
        except Exception:
            os.remove(f"{folder}/{file}")
            continue

        for c, info in results.items():
            _, text_probs, ind = info
            sorted_ids = np.argsort(text_probs)[::-1]
            if ind in sorted_ids[:k]:
                correct[c] += 1
            total[c] += 1

    accuracy = {c: correct[c] / total[c] if total[c] else 0 for c in complexities}
    print("Accuracy:", accuracy)
    print("Total:   ", total)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Qwen3-VL-Embedding image-text similarity scoring"
    )

    parser.add_argument(
        '-m', '--model',
        type=str,
        help="HuggingFace model name or local path",
        default='Qwen/Qwen3-VL-Embedding-8B'
    )

    parser.add_argument(
        '-sl', '--structural-level',
        type=str,
        choices=['l3', 'l2_or', 'l2_oa'],
        required=True,
        help="Structural level to evaluate"
    )

    parser.add_argument(
        '-o', '--outfolder',
        type=str,
        help="Folder to store the results",
        default="qwen_results_obj"
    )

    parser.add_argument(
        '-k', '--recall-k',
        type=int,
        help="Recall at k that should be calculated, max 5",
        default=1
    )

    parser.add_argument(
        '-s', '--start',
        type=int,
        help="Starting file number",
        default=0
    )

    parser.add_argument(
        '-e', '--end',
        type=int,
        help="Ending file number",
        default=10000
    )

    parser.add_argument(
        '-cl', '--complexity_low',
        type=int,
        help="Lowest Complexity",
        default=1
    )

    parser.add_argument(
        '-ch', '--complexity_high',
        type=int,
        help="Highest Complexity",
        default=12
    )

    parser.add_argument(
        '-im', '--image-dir',
        type=str,
        default='COMPASS/images',
        help="Directory containing images"
    )

    parser.add_argument(
        '--max-length',
        type=int,
        default=8192,
        help="Maximum token length for the processor"
    )

    parser.add_argument(
        '--query-instruction',
        type=str,
        default="",
        help="Optional instruction prepended to caption queries"
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help="Number of images to embed per forward pass"
    )

    parser.add_argument(
        '--flash-attn',
        action='store_true',
        help="Use Flash Attention 2"
    )

    parser.add_argument(
        '--bf16',
        action='store_true',
        help="Load model in bfloat16"
    )

    # Parse arguments
    args = parser.parse_args()

    # Map structural level to gt captions + negatives
    structural_mapping = {
        'l3': {
            'gt_captions': 'COMPASS/gt-caption/l3',
            'negatives': 'COMPASS/compositional-integration/l3/composed'
        },
        'l2_or': {
            'gt_captions': 'COMPASS/gt-caption/l2-OR',
            'negatives': 'COMPASS/compositional-integration/l2-OR/composed'
        },
        'l2_oa': {
            'gt_captions': 'COMPASS/gt-caption/l2-OA',
            'negatives': 'COMPASS/compositional-integration/l2-OA/composed'
        }
    }

    gt_captions = structural_mapping[args.structural_level]['gt_captions']
    negatives = structural_mapping[args.structural_level]['negatives']

    print("Running composed evaluation on decomposed captions with model:", args.model)

    # Model kwargs
    model_kwargs = {}

    if args.flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    if args.bf16:
        import torch
        model_kwargs["torch_dtype"] = torch.bfloat16

    # Initialize model
    initialize_model(
        args.model,
        max_length=args.max_length,
        query_instruction=args.query_instruction,
        **model_kwargs,
    )

    # Run evaluation
    get_results(
        gt_captions=gt_captions,
        negative_captions=negatives,
        outfolder=args.outfolder,
        s=args.start,
        e=args.end,
        batch_size=args.batch_size,
    )

    # Compute accuracy
    # get_accuracy(
    #     k=args.recall_k,
    #     folder=args.outfolder,
    #     s=args.start,
    #     e=args.end,
    #     cl=args.complexity_low,
    #     ch=args.complexity_high,
    # )


if __name__ == "__main__":
    main()
