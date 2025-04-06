"""
Hindi Triplet Extraction using Llama-3-Nanda-10B-Chat
This module implements triplet extraction from Hindi text using local LLM inference.
"""

import json
from typing import List, Dict, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import config

class TripletExtractor:
    def __init__(self):
        """Initialize the triplet extractor with the Llama model."""
        print("Loading model and tokenizer...")
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.MODEL_CONFIG["model_name"],
            use_fast=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_CONFIG["model_name"],
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
        print(f"Model loaded successfully. Using device: {self.device}")

    def _create_prompt(self, text: str) -> str:
        """Create a formatted prompt for the model."""
        return f"{config.SYSTEM_PROMPT}\n\n{config.EXTRACTION_PROMPT.format(text=text)}"

    def _parse_triplets(self, text: str) -> List[Dict[str, str]]:
        """Parse the model output into structured triplets."""
        triplets = []
        for line in text.strip().split('\n'):
            if '|' in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) == 3:
                    triplet = {
                        'subject': parts[0],
                        'predicate': parts[1],
                        'object': parts[2]
                    }
                    triplets.append(triplet)
        return triplets

    def extract_triplets(self, text: str) -> List[Dict[str, str]]:
        """Extract triplets from the given Hindi text."""
        prompt = self._create_prompt(text)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                              max_length=config.MODEL_CONFIG["max_length"])
        inputs = inputs.to(self.device)

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=config.MODEL_CONFIG["temperature"],
                top_p=config.MODEL_CONFIG["top_p"],
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the generated triplets part
        response_parts = response.split("त्रिक:")
        if len(response_parts) > 1:
            triplets_text = response_parts[1].strip()
            return self._parse_triplets(triplets_text)
        return []

    def process_article(self, article_text: str, batch_size: int = None) -> List[Dict[str, str]]:
        """Process a full article by breaking it into manageable segments."""
        if batch_size is None:
            batch_size = config.BATCH_SIZE

        # Split article into sentences or paragraphs
        segments = [s.strip() for s in article_text.split('।') if s.strip()]
        
        all_triplets = []
        for i in tqdm(range(0, len(segments), batch_size)):
            batch = segments[i:i + batch_size]
            batch_text = '। '.join(batch) + '।'
            triplets = self.extract_triplets(batch_text)
            all_triplets.extend(triplets)

        return all_triplets

def save_triplets(triplets: List[Dict[str, str]], output_file: str):
    """Save extracted triplets to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(triplets, f, ensure_ascii=False, indent=2)

def main():
    # Example usage
    extractor = TripletExtractor()
    
    # Sample Hindi text from Wikipedia
    sample_text = """
    जवाहरलाल नेहरू भारत के प्रथम प्रधानमंत्री थे। वे स्वतंत्र भारत के पहले प्रधानमंत्री बने और 1947 से 1964 तक इस पद पर रहे।
    नेहरू का जन्म इलाहाबाद में हुआ था। उनके पिता मोतीलाल नेहरू एक प्रसिद्ध वकील थे।
    नेहरू ने कैम्ब्रिज विश्वविद्यालय से शिक्षा प्राप्त की और बाद में भारतीय स्वतंत्रता आंदोलन में महात्मा गांधी के साथ जुड़ गए।
    """
    
    print("Extracting triplets from sample text...")
    triplets = extractor.process_article(sample_text)
    
    # Save results
    save_triplets(triplets, 'sample_triplets.json')
    print(f"Extracted {len(triplets)} triplets. Results saved to sample_triplets.json")

if __name__ == "__main__":
    main()
