"""
Local Salience Testing for MANTIS Mining
Simulates validator evaluation to test embedding quality offline
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ledger import DataLog, ChallengeData
from model import multi_salience
from scripts.training.model_architecture import VMDTMFGLSTMXGBoost
from scripts.training.train_model import load_data
import config


def create_mock_datalog(ticker: str, model_path: str, data_dir: str,
                       n_samples: int = 10000, n_competitors: int = 5) -> DataLog:
    """
    Create mock DataLog for testing
    
    Args:
        ticker: Ticker symbol
        model_path: Path to trained model
        data_dir: Data directory
        n_samples: Number of samples to generate
        n_competitors: Number of dummy competitor embeddings
        
    Returns:
        DataLog object
    """
    print(f"Creating mock DataLog for {ticker}...")
    
    # Load model
    challenge = config.CHALLENGE_MAP.get(ticker)
    if not challenge:
        raise ValueError(f"Challenge not found for {ticker}")
    
    embedding_dim = challenge['dim']
    model = VMDTMFGLSTMXGBoost(embedding_dim=embedding_dim)
    model.load(model_path)
    
    # Load data
    ohlcv, funding, oi, cross_exchange = load_data(ticker, data_dir)
    if ohlcv.empty:
        raise ValueError(f"No data available for {ticker}")
    
    # Prepare data
    X_seq, y, _ = model.prepare_data(ohlcv, funding, oi, cross_exchange)
    
    if len(X_seq) < n_samples:
        print(f"Warning: Only {len(X_seq)} samples available, using all")
        n_samples = len(X_seq)
    
    # Select random samples
    sample_indices = np.random.choice(len(X_seq), n_samples, replace=False)
    X_sample = X_seq[sample_indices]
    y_sample = y[sample_indices]
    
    # Generate embeddings
    print("Generating embeddings...")
    your_embeddings = model.predict_embeddings(X_sample)
    
    # Generate dummy competitor embeddings
    competitor_embeddings = []
    for i in range(n_competitors):
        # Random embeddings (baseline)
        dummy = np.random.uniform(-1, 1, size=(n_samples, embedding_dim))
        competitor_embeddings.append(dummy)
    
    # Create DataLog
    datalog = DataLog(root='./test_storage')
    
    # Populate with samples
    print("Populating DataLog...")
    your_hotkey = "test_your_hotkey"
    competitor_hotkeys = [f"test_competitor_{i}" for i in range(n_competitors)]
    
    for i in tqdm(range(n_samples)):
        block = i * 300  # Every 300 blocks (MANTIS delay)
        challenge_id = ticker
        
        # Get price target (1 hour ahead)
        price_idx = sample_indices[i] + 20  # time_steps offset
        if price_idx < len(ohlcv):
            price_target = ohlcv.iloc[price_idx]['close']
        else:
            price_target = ohlcv.iloc[-1]['close']
        
        # Create embeddings dict
        embeddings_dict = {
            your_hotkey: your_embeddings[i].tolist()
        }
        for j, hk in enumerate(competitor_hotkeys):
            embeddings_dict[hk] = competitor_embeddings[j][i].tolist()
        
        # Append to DataLog
        datalog.append_step(
            block=block,
            challenge=challenge_id,
            embeddings=embeddings_dict,
            price_target=price_target
        )
    
    return datalog


def evaluate_salience(datalog: DataLog, your_hotkey: str = "test_your_hotkey") -> Dict[str, float]:
    """
    Evaluate salience scores from DataLog
    
    Args:
        datalog: DataLog object
        your_hotkey: Your hotkey identifier
        
    Returns:
        Dictionary of salience scores
    """
    print("Evaluating salience...")
    
    # Get training data from DataLog
    training_data = {}
    
    for ticker, challenge_data in datalog.challenges.items():
        # Extract embeddings and prices
        X_list = []
        hotkey_to_idx = {}
        prices = []
        
        for sidx, data in challenge_data.sidx.items():
            prices.append(data.get('price', 0.0))
            
            embeddings = data.get('emb', {})
            for hk, emb in embeddings.items():
                if hk not in hotkey_to_idx:
                    hotkey_to_idx[hk] = len(hotkey_to_idx)
            
            # Create row for this sample
            row = [0.0] * (len(hotkey_to_idx) * challenge_data.dim)
            for hk, emb in embeddings.items():
                idx = hotkey_to_idx[hk]
                start = idx * challenge_data.dim
                end = start + challenge_data.dim
                row[start:end] = emb[:challenge_data.dim]
            
            X_list.append(row)
        
        if len(X_list) > 0:
            X_flat = np.array(X_list, dtype=np.float32)
            prices_arr = np.array(prices, dtype=np.float32)
            
            # Calculate returns for binary prediction
            returns = np.diff(prices_arr) / prices_arr[:-1]
            returns = np.concatenate([[0], returns])  # Pad first value
            
            training_data[ticker] = ((X_flat, hotkey_to_idx), returns)
    
    # Compute salience
    salience_scores = multi_salience(training_data)
    
    return salience_scores


def main():
    parser = argparse.ArgumentParser(description="Test salience locally")
    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Ticker to test"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Data directory"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Number of samples for testing"
    )
    parser.add_argument(
        "--n-competitors",
        type=int,
        default=5,
        help="Number of dummy competitors"
    )
    parser.add_argument(
        "--save-datalog",
        type=str,
        default=None,
        help="Path to save DataLog (optional)"
    )
    parser.add_argument(
        "--load-datalog",
        type=str,
        default=None,
        help="Path to load DataLog (optional)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MANTIS Local Salience Testing")
    print("=" * 80)
    
    # Create or load DataLog
    if args.load_datalog:
        print(f"Loading DataLog from {args.load_datalog}...")
        with open(args.load_datalog, 'rb') as f:
            datalog = pickle.load(f)
    else:
        datalog = create_mock_datalog(
            ticker=args.ticker,
            model_path=args.model_path,
            data_dir=args.data_dir,
            n_samples=args.n_samples,
            n_competitors=args.n_competitors
        )
        
        if args.save_datalog:
            print(f"Saving DataLog to {args.save_datalog}...")
            os.makedirs(os.path.dirname(args.save_datalog), exist_ok=True)
            with open(args.save_datalog, 'wb') as f:
                pickle.dump(datalog, f)
    
    # Evaluate salience
    your_hotkey = "test_your_hotkey"
    salience_scores = evaluate_salience(datalog, your_hotkey)
    
    # Display results
    print("\n" + "=" * 80)
    print("Salience Results")
    print("=" * 80)
    
    if your_hotkey in salience_scores:
        your_score = salience_scores[your_hotkey]
        all_scores = list(salience_scores.values())
        
        print(f"Your Salience Score: {your_score:.6f}")
        print(f"Baseline (uniform): {1.0 / len(all_scores):.6f}")
        print(f"Max Score: {max(all_scores):.6f}")
        print(f"Min Score: {min(all_scores):.6f}")
        print(f"Mean Score: {np.mean(all_scores):.6f}")
        
        # Percentile
        percentile = (np.array(all_scores) <= your_score).sum() / len(all_scores) * 100
        print(f"Your Percentile: {percentile:.1f}%")
        
        # Top-K
        sorted_scores = sorted(salience_scores.items(), key=lambda x: x[1], reverse=True)
        your_rank = next(i for i, (hk, _) in enumerate(sorted_scores) if hk == your_hotkey) + 1
        print(f"Your Rank: {your_rank}/{len(sorted_scores)}")
        
        print("\nTop 5 Hotkeys:")
        for i, (hk, score) in enumerate(sorted_scores[:5]):
            marker = " <-- YOU" if hk == your_hotkey else ""
            print(f"  {i+1}. {hk}: {score:.6f}{marker}")
        
        # Pass criteria
        print("\n" + "=" * 80)
        print("Pass Criteria Check")
        print("=" * 80)
        
        target_percentile = 90
        if percentile >= target_percentile:
            print(f"✓ PASS: Your percentile ({percentile:.1f}%) >= {target_percentile}%")
        else:
            print(f"✗ FAIL: Your percentile ({percentile:.1f}%) < {target_percentile}%")
        
        if your_rank <= len(sorted_scores) * 0.1:
            print(f"✓ PASS: Your rank ({your_rank}) is in top 10%")
        else:
            print(f"✗ FAIL: Your rank ({your_rank}) is not in top 10%")
    else:
        print(f"✗ Your hotkey '{your_hotkey}' not found in salience scores")
        print("Available hotkeys:", list(salience_scores.keys()))


if __name__ == "__main__":
    main()
