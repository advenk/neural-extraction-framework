# Hindi Triplet Extraction using Llama-3-Nanda

This prototype demonstrates triplet extraction from Hindi text using the Llama-3-Nanda-10B-Chat model. It's designed to extract subject-predicate-object relationships from Hindi Wikipedia articles or any Hindi text.

## Features

- Local inference using Llama-3-Nanda-10B-Chat (GGUF format)
- Batch processing for long articles
- Structured output in JSON format
- Configurable prompt templates
- Memory-efficient processing of large texts

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download the Llama-3-Nanda model:
```bash
# Download the GGUF model from Hugging Face
# You'll need git-lfs installed
git lfs install
git clone https://huggingface.co/TheBloke/Llama-3-Nanda-10B-Chat-GGUF
```

## Usage

The main script can be run directly:

```bash
python extractor.py
```

Or import the extractor in your code:

```python
from extractor import TripletExtractor

# Initialize the extractor
extractor = TripletExtractor()

# Process Hindi text
text = """
जवाहरलाल नेहरू भारत के प्रथम प्रधानमंत्री थे। 
वे स्वतंत्र भारत के पहले प्रधानमंत्री बने और 1947 से 1964 तक इस पद पर रहे।
"""
triplets = extractor.process_article(text)

# Save results
from extractor import save_triplets
save_triplets(triplets, 'output.json')
```

## Configuration

You can modify the model parameters and prompt templates in `config.py`:

- Model parameters (temperature, top_p, etc.)
- Batch size for processing
- System and extraction prompts
- Maximum segment length

## Output Format

The extracted triplets are saved in JSON format:

```json
[
  {
    "subject": "जवाहरलाल नेहरू",
    "predicate": "थे",
    "object": "भारत के प्रथम प्रधानमंत्री"
  },
  {
    "subject": "नेहरू",
    "predicate": "रहे",
    "object": "1947 से 1964 तक प्रधानमंत्री पद पर"
  }
]
```

## Notes

- The model uses quantized weights (Q4_K_M) for efficient local inference
- Adjust batch_size in config.py based on your available memory
- The system works best with well-structured Hindi text
- Consider pre-processing very long articles into smaller segments
