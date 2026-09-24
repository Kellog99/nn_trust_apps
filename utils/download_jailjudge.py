"""Download the pre-quantized JailJudge Guard GGUF judge into the model repository."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "mradermacher/JailJudge-guard-GGUF"
DEFAULT_QUANT = "Q4_K_M"
DEFAULT_OUTPUT = Path("~/Desktop/StableAI/model_repository/jailjudge-guard-q4")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--quant", default=DEFAULT_QUANT, help="e.g. Q4_K_M, Q4_K_S, IQ4_XS")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"JailJudge-guard.{args.quant}.gguf"

    print(f"Downloading {args.repo_id}/{filename}...", flush=True)
    gguf_path = hf_hub_download(
        repo_id=args.repo_id,
        filename=filename,
        local_dir=output_dir,
    )

    info = {
        "type": "model",
        "id": f"jailjudge-guard-{args.quant.lower()}",
        "name": f"JailJudge Guard ({args.quant})",
        "task": "language",
        "domain": "text",
        "input_dimensionality": [],
        "model_type": "Llamacpp",
        "is_judge": True,
        "judge_type": "jailjudge",
        "parameters": 7_000_000_000,
        "description": (
            "Jailbreak judge instruction-tuned to score a target response against the "
            "attack goal, from 1 (refusal / non-jailbroken) to 10 (fully jailbroken)."
        ),
    }
    (output_dir / "info.json").write_text(json.dumps(info, indent=2))

    size_gb = Path(gguf_path).stat().st_size / 1e9
    print(f"Done: {gguf_path} ({size_gb:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
