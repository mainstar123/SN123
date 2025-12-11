#!/usr/bin/env python3
"""
Wrapper for embedding generator to use with evaluate_embeddings.py
This loads your trained models and generates embeddings for all challenges
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.generate_embeddings_from_models import EmbeddingGenerator

# Initialize generator (will load models when needed)
# Models will be loaded from models/tuning/ directory
generator = EmbeddingGenerator(
    model_dir='models/tuning',
    data_dir='data/raw'
)

def generate_embeddings(block: int):
    """
    Function signature expected by evaluate_embeddings.py
    
    Args:
        block: Block number
        
    Returns:
        List of embeddings (one per challenge in config.CHALLENGES order)
    """
    return generator.generate_embeddings(block)

if __name__ == '__main__':
    # Test mode
    import sys
    if len(sys.argv) > 1:
        block = int(sys.argv[1])
        print(f"Generating embeddings for block {block}...")
        emb = generate_embeddings(block)
        print(f"Generated {len(emb)} embeddings")
        for i, e in enumerate(emb):
            non_zero = sum(1 for x in e if abs(x) > 1e-6)
            print(f"  Challenge {i}: dim={len(e)}, non-zero={non_zero}")
