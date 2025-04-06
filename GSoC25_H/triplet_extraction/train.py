import os
import json
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from transformers import AutoTokenizer
import jsonlines
from pathlib import Path
import argparse
from tqdm import tqdm
from model import OpenHathiModel
from typing import List, Dict, Tuple
import random


def load_jsonl(file_path: str) -> List[Dict[str, str]]:
    """Load data from a JSONL file."""
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    return data


def prepare_batch(examples: List[Dict[str, str]], tokenizer, max_length: int = 512) -> Tuple[mx.array, mx.array]:
    """Prepare a batch of examples for training."""
    input_ids = []
    labels = []
    
    for example in examples:
        # Tokenize input and target
        inputs = tokenizer(example["input_text"], return_tensors="pt", 
                         truncation=True, max_length=max_length).input_ids
        targets = tokenizer(example["target_triplets"], return_tensors="pt",
                          truncation=True, max_length=max_length).input_ids
        
        # Combine input and target for training
        combined = mx.concatenate([
            mx.array(inputs.numpy()),
            mx.array(targets.numpy())
        ], axis=1)
        
        input_ids.append(combined[:, :-1])  # All tokens except last
        labels.append(combined[:, 1:])      # All tokens except first
    
    # Pad sequences to same length
    max_len = max(x.shape[1] for x in input_ids)
    padded_inputs = []
    padded_labels = []
    
    for inp, lab in zip(input_ids, labels):
        pad_len = max_len - inp.shape[1]
        if pad_len > 0:
            padded_inputs.append(mx.pad(inp, ((0, 0), (0, pad_len)), constant_values=tokenizer.pad_token_id))
            padded_labels.append(mx.pad(lab, ((0, 0), (0, pad_len)), constant_values=-100))
        else:
            padded_inputs.append(inp)
            padded_labels.append(lab)
    
    return mx.stack(padded_inputs), mx.stack(padded_labels)


def train_step(model: OpenHathiModel, 
               inputs: mx.array,
               labels: mx.array,
               optimizer) -> float:
    """Perform one training step."""
    def loss_fn(model):
        logits = model(inputs)
        return mx.mean(mx.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                      labels.reshape(-1)))
    
    loss, grads = nn.value_and_grad(model, loss_fn)(model)
    optimizer.update(model, grads)
    return loss.item()


def validate(model: OpenHathiModel,
            val_data: List[Dict[str, str]],
            tokenizer,
            batch_size: int) -> float:
    """Run validation."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    for i in range(0, len(val_data), batch_size):
        batch = val_data[i:i + batch_size]
        inputs, labels = prepare_batch(batch, tokenizer)
        
        logits = model(inputs)
        loss = mx.mean(mx.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                      labels.reshape(-1)))
        total_loss += loss.item()
        num_batches += 1
    
    model.train()
    return total_loss / num_batches


def train(args):
    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    # Load MLX model weights
    with open(f"{args.model_path}/config.json", "r") as f:
        config = json.load(f)
    
    model = OpenHathiModel(
        vocab_size=config["vocab_size"],
        num_layers=config["num_hidden_layers"],
        num_heads=config["num_attention_heads"],
        hidden_dim=config["hidden_size"],
        mlp_dim=config["intermediate_size"]
    )
    model.load_weights(f"{args.model_path}/weights.npz")
    model.train()

    # Load datasets
    print("Loading datasets...")
    train_data = load_jsonl(os.path.join(args.data_dir, "train.jsonl"))
    val_data = load_jsonl(os.path.join(args.data_dir, "val.jsonl"))

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), learning_rate=args.learning_rate)

    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    patience = args.patience
    patience_counter = 0
    
    for epoch in range(args.num_epochs):
        # Shuffle training data
        random.shuffle(train_data)
        total_loss = 0
        num_batches = 0
        
        # Training
        progress_bar = tqdm(range(0, len(train_data), args.batch_size),
                          desc=f"Epoch {epoch + 1}/{args.num_epochs}")
        for i in progress_bar:
            batch = train_data[i:i + args.batch_size]
            inputs, labels = prepare_batch(batch, tokenizer)
            
            loss = train_step(model, inputs, labels, optimizer)
            total_loss += loss
            num_batches += 1
            
            progress_bar.set_postfix({'loss': loss})
        
        avg_train_loss = total_loss / num_batches
        
        # Validation
        val_loss = validate(model, val_data, tokenizer, args.batch_size)
        print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_dir = Path(args.output_dir) / f"checkpoint_epoch_{epoch + 1}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model weights
            model.save_weights(str(checkpoint_dir / "weights.npz"))
            
            # Save config
            with open(checkpoint_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)
                
            print(f"Saved checkpoint to {checkpoint_dir}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    print("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune OpenHathi model for relation extraction")
    parser.add_argument("--model_path", type=str, required=True,
                      help="Path to the MLX-converted OpenHathi model")
    parser.add_argument("--tokenizer_name", type=str,
                      default="sarvamai/OpenHathi-7B-Hi-v0.1-Base",
                      help="Name or path of the tokenizer")
    parser.add_argument("--data_dir", type=str, default="data",
                      help="Directory containing train.jsonl and val.jsonl")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                      help="Directory to save model checkpoints")
    parser.add_argument("--num_epochs", type=int, default=10,
                      help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                      help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                      help="Learning rate for training")
    parser.add_argument("--patience", type=int, default=3,
                      help="Number of epochs to wait for validation loss improvement")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
