import json
import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer
import jsonlines
from typing import List, Tuple, Optional


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_dim: int):
        super().__init__()
        self.attention = nn.MultiHeadAttention(hidden_dim, num_heads)
        self.mlp = nn.Sequential([
            nn.Linear(hidden_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, hidden_dim)
        ])
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        # Self attention
        normed_x = self.layer_norm1(x)
        attention_output = self.attention(normed_x, normed_x, normed_x, mask)
        x = x + attention_output

        # MLP
        normed_x = self.layer_norm2(x)
        mlp_output = self.mlp(normed_x)
        x = x + mlp_output

        return x


class OpenHathiModel(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, hidden_dim: int, mlp_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Create transformer layers as a list of attributes
        self.layers = []
        for i in range(num_layers):
            layer = TransformerBlock(hidden_dim, num_heads, mlp_dim)
            setattr(self, f"layer_{i}", layer)
            self.layers.append(layer)
            
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def __call__(self, input_ids: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x, attention_mask)
            
        x = self.layer_norm(x)
        logits = self.output(x)
        return logits
    
    def save_weights(self, path: str):
        """Save model weights to a file."""
        mx.savez(path, **self.parameters())
    
    def load_weights(self, path: str):
        """Load model weights from a file."""
        weights = mx.load(path)
        self.update(weights)


class RelationExtractor:
    def __init__(self, model_path: str, tokenizer_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Load model config
        with open(f"{model_path}/config.json", "r") as f:
            config = json.load(f)
        
        self.model = OpenHathiModel(
            vocab_size=config["vocab_size"],
            num_layers=config["num_hidden_layers"],
            num_heads=config["num_attention_heads"],
            hidden_dim=config["hidden_size"],
            mlp_dim=config["intermediate_size"]
        )
        self.model.load_weights(f"{model_path}/weights.npz")
        self.model.eval()

    def generate(self, input_ids: mx.array, max_length: int = 512, temperature: float = 0.7) -> mx.array:
        """Generate text using simple greedy decoding."""
        output_ids = input_ids
        for _ in range(max_length - input_ids.shape[1]):
            logits = self.model(output_ids)
            next_token_logits = logits[:, -1, :] / temperature
            next_token = mx.argmax(next_token_logits, axis=-1, keepdims=True)
            output_ids = mx.concatenate([output_ids, next_token], axis=1)
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        return output_ids

    def process_text(self, text: str) -> List[Tuple[str, str, str]]:
        """Process Hindi text and extract relation triplets."""
        prompt = f"Text: {text} Triplets:"
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids_mx = mx.array(input_ids.numpy())
        output_ids = self.generate(input_ids_mx)
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Parse triplets
        try:
            triplets_str = output_text.split("Triplets:")[1].strip()
            triplets = [tuple(triplet.strip().split(" | ")) 
                       for triplet in triplets_str.split(";") 
                       if triplet.strip()]
            return triplets
        except IndexError:
            return []

    def save_triplets(self, triplets: List[Tuple[str, str, str]], filename: str):
        """Save extracted triplets to a JSON file."""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(triplets, f, ensure_ascii=False, indent=2)
