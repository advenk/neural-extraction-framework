import json
import jsonlines
from pathlib import Path
from typing import List, Dict, Union
import argparse


def create_sample_dataset() -> List[Dict[str, str]]:
    """Create a sample Hindi relation extraction dataset."""
    return [
        {
            "input_text": "जवाहरलाल नेहरू भारत के प्रथम प्रधानमंत्री थे।",
            "target_triplets": "जवाहरलाल नेहरू | था | भारत के प्रथम प्रधानमंत्री"
        },
        {
            "input_text": "वे स्वतंत्र भारत के पहले प्रधानमंत्री बने और 1947 से 1964 तक इस पद पर रहे।",
            "target_triplets": "जवाहरलाल नेहरू | बने | स्वतंत्र भारत के पहले प्रधानमंत्री ; जवाहरलाल नेहरू | रहे | इस पद पर 1947 से 1964 तक"
        },
        {
            "input_text": "नेहरू का जन्म इलाहाबाद में हुआ था।",
            "target_triplets": "नेहरू | का जन्म | इलाहाबाद में"
        },
        {
            "input_text": "उनके पिता मोतीलाल नेहरू एक प्रसिद्ध वकील थे।",
            "target_triplets": "मोतीलाल नेहरू | थे | एक प्रसिद्ध वकील"
        },
        {
            "input_text": "नेहरू ने कैम्ब्रिज विश्वविद्यालय से शिक्षा प्राप्त की।",
            "target_triplets": "नेहरू | ने शिक्षा प्राप्त की | कैम्ब्रिज विश्वविद्यालय से"
        }
    ]


def format_for_training(example: Dict[str, str]) -> Dict[str, str]:
    """Format a single example for training."""
    return {
        "input": f"Text: {example['input_text']}\nTriplets:",
        "target": example['target_triplets']
    }


def save_dataset(data: List[Dict[str, str]], output_dir: Union[str, Path], split: str = "train"):
    """Save dataset in JSONL format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    formatted_data = [format_for_training(example) for example in data]
    
    output_file = output_dir / f"{split}.jsonl"
    with jsonlines.open(output_file, mode='w') as writer:
        writer.write_all(formatted_data)
    
    print(f"Saved {len(formatted_data)} examples to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for relation extraction")
    parser.add_argument("--output_dir", type=str, default="data",
                      help="Directory to save the processed data")
    parser.add_argument("--input_file", type=str, default=None,
                      help="Optional input file with custom examples")

    args = parser.parse_args()
    
    if args.input_file:
        # Load custom dataset if provided
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # Use sample dataset
        data = create_sample_dataset()
    
    # Save training data
    save_dataset(data, args.output_dir, "train")
    
    # Create a small validation set (20% of data)
    val_size = max(1, len(data) // 5)
    val_data = data[-val_size:]
    save_dataset(val_data, args.output_dir, "val")


if __name__ == "__main__":
    main()
