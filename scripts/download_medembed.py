"""Download the pinned public MedEmbed model for local reference search."""
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = 'abhinand/MedEmbed-small-v0.1'
REVISION = '40a5850d046cfdb56154e332b4d7099b63e8d50e'

if __name__ == '__main__':
    directory = ROOT / 'artifacts/pretrained/medembed-small'
    snapshot_download(
        MODEL_ID,
        revision=REVISION,
        local_dir=directory,
        allow_patterns=[
            '1_Pooling/config.json', 'config.json', 'config_sentence_transformers.json',
            'model.safetensors', 'modules.json', 'sentence_bert_config.json',
            'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json',
            'vocab.txt',
        ],
    )
    print(f'MedEmbed ready at {directory}')
