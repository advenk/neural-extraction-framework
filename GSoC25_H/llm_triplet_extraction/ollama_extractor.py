import json
from typing import List, Dict
import subprocess
from tqdm import tqdm
import config
import ollama

class TripletExtractor:
    def __init__(self):
        """Initialize the triplet extractor using Ollama."""
        print("Using Ollama for model inference.")
        # No need to load a local model or tokenizer with Ollama

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

    def get_prompt(self, text):
        template = {
            "triplets": [
                {
                    "subject": "Hindi Subject",
                    "predicate": "Hindi Predicate",
                    "object": "Hindi Object",
                }
            ],
        }
        max_triplets = 3
        example_output = {
            "triplets": [
                {
                    "subject": "एबिलीन टेक्सास",
                    "predicate": "देश",
                    "object": "संयुक्त राज्य अमेरिक",
                }
            ]
        }
        """
        Some examples are also given below:
        Text: एबिलीन, टेक्सास संयुक्त राज्य अमेरिका में है।
        {json.dumps(example_output, ensure_ascii= False)}
        """

        prompt = f"""
        Generate Hindi knowledge triplets from the following text. Each triplet should be in the format (subject, predicate, object). Extract up to {max_triplets} triplets and present them in JSON format. The output should be in hindi and should look like this:
        {json.dumps(template, ensure_ascii=False)}
        Here's the text to analyze:
        {text}"""
        return prompt
    
    def extract_triplets(self, text: str) -> List[Dict[str, str]]:
        """Extract triplets from the given Hindi text using Ollama."""
        # prompt = self._create_prompt(text)
        prompt = self.get_prompt(text)
        

        # Call Ollama via subprocess.
        # The prompt is sent via standard input.
        result = subprocess.run(
            ["ollama", "run", "hf.co/mradermacher/Llama-3-Nanda-10B-Chat-GGUF:IQ4_XS"],
            input=prompt,
            capture_output=True,
            text=True,
            check=True
        )
        response = result.stdout

        # Extract the generated triplets part using the marker "त्रिक:"
        response_parts = response.split("त्रिक:")
        if len(response_parts) > 1:
            triplets_text = response_parts[1].strip()
            return self._parse_triplets(triplets_text)
        return []

    def process_article(self, article_text: str, batch_size: int = None) -> List[Dict[str, str]]:
        """Process a full article by breaking it into manageable segments."""
        if batch_size is None:
            batch_size = config.BATCH_SIZE

        # Split the article into sentences or paragraphs using '।' as the delimiter
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