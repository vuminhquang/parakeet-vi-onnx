#!/usr/bin/env python3
"""
Export nvidia/parakeet-ctc-0.6b-Vietnamese → sherpa-onnx NeMo CTC format.

Output (in --out-dir):
  model.onnx       float32 full precision
  model.int8.onnx  int8 quantized (used by relay_telegram)
  tokens.txt       one token per line: "<token> <id>"

Usage:
  python build.py [--out-dir ./output]
"""

import argparse
import os
import sys
import shutil

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType


HF_MODEL_ID = "nvidia/parakeet-ctc-0.6b-vi"


def load_nemo_model(model_id: str):
    import nemo.collections.asr as nemo_asr
    print(f"[1/4] loading {model_id} from HuggingFace...")
    # Try BPE variant first (Parakeet uses SentencePiece BPE)
    try:
        model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_id)
    except Exception:
        model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_id)
    model.eval()
    return model


def export_onnx(model, out_path: str):
    print(f"[2/4] exporting ONNX → {out_path}")
    model.export(out_path, check_trace=False, onnx_opset_version=17)

    # Verify
    m = onnx.load(out_path)
    onnx.checker.check_model(m)
    inputs  = [i.name for i in m.graph.input]
    outputs = [o.name for o in m.graph.output]
    print(f"      inputs : {inputs}")
    print(f"      outputs: {outputs}")


def quantize_int8(src: str, dst: str):
    print(f"[3/4] quantizing int8 → {dst}")
    quantize_dynamic(src, dst, weight_type=QuantType.QInt8)


def export_tokens(model, out_path: str):
    print(f"[4/4] exporting tokens → {out_path}")

    # BPE tokenizer (Parakeet)
    if hasattr(model, "tokenizer"):
        vocab = model.tokenizer.tokenizer.get_vocab()  # dict token→id
        tokens = sorted(vocab.items(), key=lambda x: x[1])
        with open(out_path, "w", encoding="utf-8") as f:
            for token, idx in tokens:
                f.write(f"{token} {idx}\n")
            # blank token = last id + 1 in NeMo CTC
            f.write(f"<blk> {len(tokens)}\n")
    # Character-level fallback
    elif hasattr(model.decoder, "vocabulary"):
        vocab = model.decoder.vocabulary
        with open(out_path, "w", encoding="utf-8") as f:
            for idx, token in enumerate(vocab):
                f.write(f"{token} {idx}\n")
            f.write(f"<blk> {len(vocab)}\n")
    else:
        raise RuntimeError("cannot find vocabulary — inspect model manually")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="./output", help="output directory")
    parser.add_argument("--model-id", default=HF_MODEL_ID, help="HuggingFace model id")
    parser.add_argument("--skip-fp32", action="store_true", help="skip float32 export (only int8)")
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

    print("\ndone:")
    for f in os.listdir(args.out_dir):
        p = os.path.join(args.out_dir, f)
        print(f"  {p}  ({os.path.getsize(p) // 1024 // 1024} MB)")

    print(f"""
to use with relay_telegram, copy output to the stt model dir:
  cp {args.out_dir}/model.int8.onnx <model_dir>/model.int8.onnx
  cp {args.out_dir}/tokens.txt      <model_dir>/tokens.txt

then set relay_telegram.json:
  "stt_model": "nemo-parakeet-ctc-0.6b-vi"
""")


if __name__ == "__main__":
    main()
