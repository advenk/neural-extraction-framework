"""Configuration for the LLM-based triplet extraction system."""

# Model configuration
MODEL_CONFIG = {
    "model_name": "MBZUAI/Llama-3-Nanda-10B-Chat",  # GGUF format for local inference
    "model_file": "llama-3-nanda-10b-chat.Q4_K_M.gguf",    # Quantized model file
    "max_length": 2048,
    "temperature": 0.1,  # Lower temperature for more focused extraction
    "top_p": 0.9,
}

# # Prompt templates
# SYSTEM_PROMPT = """You are a helpful assistant specialized in extracting subject-predicate-object triplets from Hindi text.
# Your task is to identify relationships between entities in the given text.
# Output format should be: [subject] | [predicate] | [object]
# Each triplet should be on a new line."""

# EXTRACTION_PROMPT = """निम्नलिखित हिंदी पाठ से सभी संभव त्रिक (subject-predicate-object) निकालें।
# केवल वे त्रिक निकालें जो पाठ में स्पष्ट रूप से मौजूद हैं।
# प्रत्येक त्रिक को नई पंक्ति में [subject] | [predicate] | [object] प्रारूप में लिखें।

# पाठ:
# {text}

# त्रिक:"""

SYSTEM_PROMPT = "Extract triplets from Hindi text. Format: subject | predicate | object."

EXTRACTION_PROMPT = "पाठ: {text}\nत्रिक (Format: subject | predicate | object):"

# Processing configuration
BATCH_SIZE = 5  # Number of text segments to process in parallel
MAX_SEGMENT_LENGTH = 512  # Maximum length of text segment to process at once
