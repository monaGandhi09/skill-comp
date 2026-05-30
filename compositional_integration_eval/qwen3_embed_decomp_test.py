"""
Qwen3-VL-Embedding-8B evaluation on decomposed negatives.

Each JSON file contains complexity-keyed entries of the form:
    { "GT sentence": ["neg1", ...], "replaced": "object|relation|attribute" }

We filter by --type (object / relation / attribute / all) and for each entry
compute cosine similarity between the image embedding and each sentence
embedding, then check whether the GT sentence ranks highest.
"""

import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import json
import random
import argparse
import sys
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "qwen3-vl-embedding-test"))
from qwen3_vl_embedding import Qwen3VLEmbedder


_embedder: Qwen3VLEmbedder | None = None
_query_instruction: str = ""

IMAGE_PATH = "VISUAL_GENOME/images"

def initialize_model(
    model_name_or_path: str,
    max_length: int = 8192,
    query_instruction: str = "",
    **kwargs,
) -> None:
    global _embedder, _query_instruction
    _embedder = Qwen3VLEmbedder(
        model_name_or_path=model_name_or_path,
        max_length=max_length,
        **kwargs,
    )
    _query_instruction = query_instruction
    print(f"Initialized Qwen3VLEmbedder: model={model_name_or_path}")


def _parse_entries(data: dict, neg_type: str, cl: int, ch: int):
    """
    Yield (complexity, gt_sentence, [neg_sentences]) for each matching entry.
    complexity is an int in [cl, ch]; neg_type filters on the "replaced" field.
    """
    for complexity_str, entries in data.items():
        complexity = int(complexity_str)
        if not (cl <= complexity <= ch):
            continue
        for item in entries:
            replaced = item.get("replaced", "")
            if neg_type != "all" and replaced != neg_type:
                continue
            gt_key = next(k for k in item if k != "replaced")
            negs = item[gt_key]
            if not negs:
                continue
            yield complexity, gt_key, negs


def get_results(
    decomposed_dir: str,
    outfolder: str,
    neg_type: str = "all",
    s: int = 0,
    e: int = 10000,
    cl: int = 3,
    ch: int = 12,
    batch_size: int = 8,
) -> None:
    """
    Embed images and sentences, score GT vs negatives for each entry.

    Output per file (JSON):
        { "complexity": [ [sentence_list, prob_list, gt_index], ... ], ... }
    """
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    all_files = sorted(os.listdir(decomposed_dir))
    pending = all_files[s:e]

    for batch_start in tqdm(range(0, len(pending), batch_size), unit="batch"):
        batch_files = pending[batch_start : batch_start + batch_size]

        # ── Load data for this batch ─────────────────────────────────────────
        batch_data = []  # [(file, img_path, [(complexity, sentences, gt_ind), ...], existing_out)]
        for file in batch_files:
            img_name = f"{file.split('.')[0]}.jpg"
            img_path = os.path.join(IMAGE_PATH, img_name)

            try:
                with open(os.path.join(decomposed_dir, file)) as f:
                    data = json.load(f)
            except Exception:
                print(f"Error reading file: {file}")
                continue

            existing_out = {}
            out_path = os.path.join(outfolder, file)
            if os.path.exists(out_path):
                try:
                    with open(out_path) as f:
                        existing_out = json.load(f)
                except Exception:
                    pass

            items = []
            for complexity, gt_sentence, neg_sentences in _parse_entries(data, neg_type, cl, ch):
                if str(complexity) in existing_out:
                    continue
                all_sentences = neg_sentences.copy()
                gt_ind = random.randint(0, len(all_sentences))
                all_sentences.insert(gt_ind, gt_sentence)
                items.append((complexity, all_sentences, gt_ind))

            if items:
                batch_data.append((file, img_path, items, existing_out))

        if not batch_data:
            continue

        # ── Build two-tower batched inputs ───────────────────────────────────
        img_inputs = [{"image": img_path} for _, img_path, _, _ in batch_data]

        cap_inputs = []
        cap_meta = []  # (file_idx, complexity, sentences, gt_ind, cap_offset, n_caps)
        for file_idx, (_, _, items, _) in enumerate(batch_data):
            for complexity, sentences, gt_ind in items:
                cap_offset = len(cap_inputs)
                for sent in sentences:
                    cap_inputs.append(
                        {"text": sent, "instruction": _query_instruction}
                        if _query_instruction
                        else {"text": sent}
                    )
                cap_meta.append((file_idx, complexity, sentences, gt_ind, cap_offset, len(sentences)))

        # ── Two forward passes ───────────────────────────────────────────────
        img_embeddings = _embedder.process(img_inputs)
        cap_embeddings = _embedder.process(cap_inputs)

        # ── Collect per-file results ─────────────────────────────────────────
        # file_results[file_idx][complexity] = list of (sentences, probs, gt_ind)
        file_results = [{} for _ in batch_data]
        for file_idx, complexity, sentences, gt_ind, cap_offset, n_caps in cap_meta:
            img_emb = img_embeddings[file_idx]
            txt_embs = cap_embeddings[cap_offset : cap_offset + n_caps]

            sim = (txt_embs @ img_emb).float().cpu().numpy()
            sim -= sim.max()
            exp_sim = np.exp(sim)
            probs = (exp_sim / exp_sim.sum()).tolist()

            c_str = str(complexity)
            if c_str not in file_results[file_idx]:
                file_results[file_idx][c_str] = []
            file_results[file_idx][c_str].append((sentences, probs, gt_ind))

        # ── Save ─────────────────────────────────────────────────────────────
        for (file, _, _, existing_out), results in zip(batch_data, file_results):
            merged = {**existing_out, **results}
            with open(os.path.join(outfolder, file), "w") as f:
                json.dump(merged, f, indent=4)


def get_accuracy(
    folder: str,
    k: int = 1,
    s: int = 0,
    e: int = 10000,
    cl: int = 3,
    ch: int = 12,
) -> None:
    files = sorted(os.listdir(folder))
    complexities = [str(x) for x in range(cl, ch + 1)]
    correct = {c: 0 for c in complexities}
    total = {c: 0 for c in complexities}

    for file in tqdm(files[s:e]):
        try:
            with open(os.path.join(folder, file)) as f:
                results = json.load(f)
        except Exception:
            os.remove(os.path.join(folder, file))
            continue

        for c_str, entries in results.items():
            if c_str not in correct:
                continue
            for sentences, probs, gt_ind in entries:
                sorted_ids = np.argsort(probs)[::-1]
                if gt_ind in sorted_ids[:k]:
                    correct[c_str] += 1
                total[c_str] += 1

    accuracy = {c: correct[c] / total[c] if total[c] else 0 for c in complexities}
    print("Accuracy:", accuracy)
    print("Total:   ", total)


def main() -> None:
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Qwen3-VL-Embedding evaluation on decomposed negatives"
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
        required=True,
        help="Folder to save output results"
    )

    parser.add_argument(
        '-t', '--type',
        type=str,
        default='all',
        choices=['object', 'relation', 'attribute', 'all'],
        help="Filter entries by replaced type"
    )

    parser.add_argument(
        '-k', '--recall-k',
        type=int,
        default=1,
        help="Recall@K value"
    )

    parser.add_argument(
        '-s', '--start',
        type=int,
        default=0,
        help="Start index for processing files"
    )

    parser.add_argument(
        '-e', '--end',
        type=int,
        default=1,
        help="End index for processing files"
    )

    parser.add_argument(
        '-cl', '--complexity_low',
        type=int,
        default=3,
        help="Lowest complexity level"
    )

    parser.add_argument(
        '-ch', '--complexity_high',
        type=int,
        default=12,
        help="Highest complexity level"
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

    # Map structural level to decomposed negatives folder
    structural_mapping = {
        'l3': {
            'decomposed_dir': 'COMPASS/compositional-integration/l3/decomposed',
        },
        'l2_or': {
            'decomposed_dir': 'COMPASS/compositional-integration/l2-OR/decomposed',
        },
        'l2_oa': {
            'decomposed_dir': 'COMPASS/compositional-integration/l2-OA/decomposed',
        }
    }

    decomposed_dir = structural_mapping[args.structural_level]['decomposed_dir']

    print("Running decomposed evaluation with model:", args.model)

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
        decomposed_dir=decomposed_dir,
        outfolder=args.outfolder,
        neg_type=args.type,
        s=args.start,
        e=args.end,
        cl=args.complexity_low,
        ch=args.complexity_high,
        batch_size=args.batch_size,
    )

    # Compute accuracy
    get_accuracy(
        folder=args.outfolder,
        k=args.recall_k,
        s=args.start,
        e=args.end,
        cl=args.complexity_low,
        ch=args.complexity_high,
    )


if __name__ == "__main__":
    main()
