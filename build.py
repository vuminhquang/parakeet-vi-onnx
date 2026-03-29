#!/usr/bin/env python3
"""
Export nvidia/parakeet-ctc-0.6b-vi → sherpa-onnx NeMo CTC format.

sherpa-onnx NeMo CTC expects encoder+decoder only (mel features as input),
NOT the full pipeline with preprocessor. This script exports only the
encoder+decoder subgraph so sherpa-onnx can apply its own feature extraction.

Output:
  model.onnx       float32
  model.int8.onnx  int8 quantized
  tokens.txt       "<token> <id>" per line

Usage:
  python build.py [--out-dir ./output] [--skip-fp32]
"""

import argparse
import os

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

HF_MODEL_ID = "nvidia/parakeet-ctc-0.6b-vi"


def load_nemo_model(model_id: str):
    import nemo.collections.asr as nemo_asr
    print(f"[1/4] loading {model_id}...")
    try:
        model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_id)
    except Exception:
        model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_id)
    model.eval()
    return model


def export_onnx(model, out_path: str):
    """Export encoder+decoder only using NeMo's native export with mel input_example.

    Passing input_example as mel features (not raw audio) causes NeMo to skip
    the preprocessor in the ONNX graph — exactly what sherpa-onnx expects.
    opset_version=13 ensures broad ONNX Runtime compatibility.
    """
    print(f"[2/4] exporting encoder+decoder ONNX → {out_path}")

    feat_dim = model.cfg.preprocessor.features  # 80 for Parakeet
    T = 256
    input_example = (
        torch.zeros(1, feat_dim, T, dtype=torch.float32),  # [B, D, T] mel features
        torch.tensor([T], dtype=torch.int64),
    )

    model.export(
        out_path,
        input_example   = input_example,
        check_trace     = False,
        onnx_opset_version = 13,
    )

    # verify
    onnx.checker.check_model(out_path)
    m = onnx.load(out_path, load_external_data=False)
    print(f"      inputs : {[i.name for i in m.graph.input]}")
    print(f"      outputs: {[o.name for o in m.graph.output]}")
    print(f"      feat_dim: {feat_dim}")


def quantize_int8(src: str, dst: str):
    print(f"[3/4] quantizing int8 → {dst}")
    quantize_dynamic(src, dst, weight_type=QuantType.QInt8)


def export_tokens(model, out_path: str):
    print(f"[4/4] exporting tokens → {out_path}")
    if hasattr(model, "tokenizer"):
        vocab  = model.tokenizer.tokenizer.get_vocab()
        tokens = sorted(vocab.items(), key=lambda x: x[1])
        with open(out_path, "w", encoding="utf-8") as f:
            for token, idx in tokens:
                f.write(f"{token} {idx}\n")
            f.write(f"<blk> {len(tokens)}\n")
    elif hasattr(model.decoder, "vocabulary"):
        vocab = model.decoder.vocabulary
        with open(out_path, "w", encoding="utf-8") as f:
            for idx, token in enumerate(vocab):
                f.write(f"{token} {idx}\n")
            f.write(f"<blk> {len(vocab)}\n")
    else:
        raise RuntimeError("cannot find vocabulary")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir",  default="./output")
    parser.add_argument("--model-id", default=HF_MODEL_ID)
    parser.add_argument("--skip-fp32", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fp32_path = os.path.join(args.out_dir, "model.onnx")
    int8_path = os.path.join(args.out_dir, "model.int8.onnx")
    tok_path  = os.path.join(args.out_dir, "tokens.txt")

    model = load_nemo_model(args.model_id)
    export_onnx(model, fp32_path)
    export_tokens(model, tok_path)
    quantize_int8(fp32_path, int8_path)

    if args.skip_fp32:
        os.remove(fp32_path)
        # remove external data files for fp32 if any
        for f in os.listdir(args.out_dir):
            if f not in ("model.int8.onnx", "tokens.txt") and not f.endswith(".onnx"):
                try:
                    os.remove(os.path.join(args.out_dir, f))
                except OSError:
                    pass

    print("\ndone:")
    for f in sorted(os.listdir(args.out_dir)):
        p = os.path.join(args.out_dir, f)
        print(f"  {f}  ({os.path.getsize(p) // 1024 // 1024} MB)")


if __name__ == "__main__":
    main()
