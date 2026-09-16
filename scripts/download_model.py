"""One-time public pretrained model download; no patient data is sent."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from huggingface_hub import snapshot_download

from src.models import BASE_MODEL, BASE_REVISION

if __name__ == '__main__':
    snapshot_download(
        BASE_MODEL,
        revision=BASE_REVISION,
        local_dir=ROOT / 'artifacts/pretrained/minilm',
        allow_patterns=[
            'config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json',
            'special_tokens_map.json', 'vocab.txt',
        ],
    )
