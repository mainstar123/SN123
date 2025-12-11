"""
Generate embeddings from trained models for local testing
This script loads your trained models and generates embeddings for all challenges
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
import argparse
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.training.model_architecture import VMDTMFGLSTMXGBoost
from scripts.training.train_model import load_data, prepare_train_val_test_split
from scripts.data_collection.data_fetcher import DataFetcher
import config


class EmbeddingGenerator:
    """Generate embeddings from trained models"""
    
    def __init__(self, model_dir: str, data_dir: str):
        """
        Initialize generator
        
        Args:
            model_dir: Directory containing trained models (e.g., models/tuning/)
            data_dir: Directory containing data files
        """
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.models = {}
        self.data_cache = {}
        
        # Load all models
        self._load_models()
    
    def _load_models(self):
        """Load all trained models"""
        print("Loading trained models...")
        
        for challenge in config.CHALLENGES:
            ticker = challenge['ticker']
            embedding_dim = challenge['dim']
            
            # Try to find best model
            ticker_dir = os.path.join(self.model_dir, ticker)
            
            if not os.path.exists(ticker_dir):
                print(f"  ⚠ No model directory for {ticker}, skipping")
                continue
            
            # Find best config (look for best_configs JSON or most recent)
            best_config_path = None
            
            # Try to find best_configs JSON
            for file in os.listdir(self.model_dir):
                if file.startswith('best_configs_') and file.endswith('.json'):
                    try:
                        with open(os.path.join(self.model_dir, file), 'r') as f:
                            best_configs = json.load(f)
                            if ticker in best_configs:
                                config_name = best_configs[ticker]['config']
                                # Build config name from config dict
                                config_str = f"h{config_name['lstm_hidden']}_f{config_name['tmfg_n_features']}_d{config_name['dropout']}_lr{config_name['learning_rate']}"
                                best_config_path = os.path.join(ticker_dir, config_str, ticker)
                                break
                    except:
                        pass
            
            # If not found, try most recent directory
            if not best_config_path or not os.path.exists(best_config_path):
                subdirs = [d for d in os.listdir(ticker_dir) if os.path.isdir(os.path.join(ticker_dir, d))]
                if subdirs:
                    # Use most recently modified
                    subdirs.sort(key=lambda x: os.path.getmtime(os.path.join(ticker_dir, x)), reverse=True)
                    best_config_path = os.path.join(ticker_dir, subdirs[0], ticker)
            
            if not os.path.exists(best_config_path):
                print(f"  ⚠ No model found for {ticker} at {best_config_path}")
                continue
            
            try:
                model = VMDTMFGLSTMXGBoost(embedding_dim=embedding_dim)
                model.load(best_config_path)
                self.models[ticker] = model
                print(f"  ✓ Loaded model for {ticker}")
            except Exception as e:
                print(f"  ✗ Failed to load model for {ticker}: {e}")
    
    def _get_data_for_block(self, ticker: str, block: int):
        """
        Get data up to a specific block for generating embeddings
        
        Args:
            ticker: Ticker symbol
            block: Block number
            
        Returns:
            DataFrame with data up to block
        """
        # Use price_key if available
        challenge = config.CHALLENGE_MAP.get(ticker)
        if not challenge:
            return None
        
        data_ticker = challenge.get('price_key', ticker)
        
        # Check cache
        cache_key = f"{data_ticker}_{block}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        # Load data
        try:
            fetcher = DataFetcher(data_dir=self.data_dir)
            ohlcv, funding, oi = fetcher.load_data(data_ticker)
            
            if ohlcv.empty:
                return None
            
            # Filter data up to block (approximate - using datetime)
            # Block to datetime conversion (approximate)
            # Each block is ~12 seconds, so block N is approximately N * 12 seconds from genesis
            # This is a simplification - adjust based on your needs
            
            # For now, just use all available data
            # In production, you'd want to filter by actual block time
            
            self.data_cache[cache_key] = (ohlcv, funding, oi)
            return (ohlcv, funding, oi)
            
        except Exception as e:
            print(f"  ⚠ Error loading data for {ticker}: {e}")
            return None
    
    def generate_embeddings(self, block: int) -> List[List[float]]:
        """
        Generate embeddings for all challenges at a given block
        
        Args:
            block: Block number
            
        Returns:
            List of embeddings (one per challenge in config.CHALLENGES order)
        """
        embeddings = []
        
        for challenge in config.CHALLENGES:
            ticker = challenge['ticker']
            dim = challenge['dim']
            
            # Check if we have a model for this ticker
            if ticker not in self.models:
                # Return zero vector if no model
                embeddings.append([0.0] * dim)
                continue
            
            model = self.models[ticker]
            
            # Get data
            data = self._get_data_for_block(ticker, block)
            if not data:
                embeddings.append([0.0] * dim)
                continue
            
            ohlcv, funding, oi = data
            
            if ohlcv.empty:
                embeddings.append([0.0] * dim)
                continue
            
            try:
                # Prepare data
                X_seq, _, _ = model.prepare_data(ohlcv, funding, oi)
                
                if len(X_seq) == 0:
                    embeddings.append([0.0] * dim)
                    continue
                
                # Use most recent sequence
                X_latest = X_seq[-1:].reshape(1, model.time_steps, -1)
                
                # Generate embedding
                embedding = model.predict_embeddings(X_latest)
                
                if len(embedding) > 0:
                    # Convert to list and clip to [-1, 1]
                    emb_list = embedding[0].tolist()
                    emb_list = [max(-1.0, min(1.0, float(x))) for x in emb_list]
                    
                    # Ensure correct dimension
                    if len(emb_list) != dim:
                        # Pad or truncate
                        if len(emb_list) < dim:
                            emb_list.extend([0.0] * (dim - len(emb_list)))
                        else:
                            emb_list = emb_list[:dim]
                    
                    embeddings.append(emb_list)
                else:
                    embeddings.append([0.0] * dim)
                    
            except Exception as e:
                print(f"  ⚠ Error generating embedding for {ticker} at block {block}: {e}")
                embeddings.append([0.0] * dim)
        
        return embeddings


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description="Generate embeddings from trained models"
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models/tuning',
        help='Directory containing trained models'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw',
        help='Directory containing data files'
    )
    parser.add_argument(
        '--block',
        type=int,
        required=True,
        help='Block number to generate embeddings for'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file (optional)'
    )
    
    args = parser.parse_args()
    
    # Create generator
    generator = EmbeddingGenerator(args.model_dir, args.data_dir)
    
    # Generate embeddings
    print(f"\nGenerating embeddings for block {args.block}...")
    embeddings = generator.generate_embeddings(args.block)
    
    # Print results
    print(f"\nGenerated {len(embeddings)} embeddings:")
    for i, (challenge, emb) in enumerate(zip(config.CHALLENGES, embeddings)):
        non_zero = sum(1 for x in emb if abs(x) > 1e-6)
        print(f"  {challenge['ticker']}: dim={len(emb)}, non-zero={non_zero}")
    
    # Save if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump({
                'block': args.block,
                'embeddings': embeddings,
                'challenges': [c['ticker'] for c in config.CHALLENGES]
            }, f, indent=2)
        print(f"\nSaved to {args.output}")


# For use with evaluate_embeddings.py
if __name__ == '__main__':
    # If called directly, run main
    if len(sys.argv) > 1:
        main()
    else:
        # Otherwise, this module can be imported and used
        pass

