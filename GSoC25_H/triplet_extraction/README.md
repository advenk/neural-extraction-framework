# Hindi Relation Extraction using OpenHathi

This implementation fine-tunes the OpenHathi model for relation extraction from Hindi text using MLX, optimized for Apple Silicon Macs.

## Prerequisites

- MacBook with Apple Silicon (M1/M2/M3)
- Python 3.8+
- MLX framework

## Installation

1. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Convert Model

First, convert the OpenHathi model to MLX format:

```bash
python convert_model.py \
  --hf_path sarvamai/OpenHathi-7B-Hi-v0.1-Base \
  --mlx_path OpenHathi-7B-Hi-v0.1-Base-mlx
```

### 2. Fine-tune Model

Fine-tune the model on your relation extraction dataset:

```bash
python train.py \
  --model_path OpenHathi-7B-Hi-v0.1-Base-mlx \
  --tokenizer_name sarvamai/OpenHathi-7B-Hi-v0.1-Base \
  --dataset_path dataset.jsonl \
  --output_dir OpenHathi-finetuned-mlx \
  --num_epochs 3 \
  --learning_rate 1e-5
```

If no dataset is provided, a sample dataset will be created automatically.

### Dataset Format

The training data should be in JSONL format with each line containing:
```json
{
    "input_text": "Hindi text...",
    "target_triplets": "subject | relation | object ; subject2 | relation2 | object2"
}
```

### Model Architecture

The implementation uses a decoder-only transformer architecture with the following components:
- Token embeddings
- Transformer decoder layers
- Layer normalization
- Language model head

### Output Format

The model extracts relation triplets in the format:
```
(subject, relation, object)
```

Example:
```python
from model import RelationExtractor

extractor = RelationExtractor(
    model_path="OpenHathi-finetuned-mlx",
    tokenizer_name="sarvamai/OpenHathi-7B-Hi-v0.1-Base"
)

text = "जवाहरलाल नेहरू भारत के प्रथम प्रधानमंत्री थे।"
triplets = extractor.process_text(text)
extractor.save_triplets(triplets, "output.json")
```

## Performance Optimization

- The implementation is optimized for Apple Silicon using MLX
- Uses efficient memory management for large model handling
- Supports model quantization (when available in MLX)
