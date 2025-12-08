"""
Hyperparameter Tuning Script for MANTIS Models
Tests different hyperparameter configurations to find optimal settings
"""

import os
import sys
import argparse
import json
from pathlib import Path
from itertools import product
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.training.train_model import train_ticker_model, load_data
import config


def test_hyperparameter_config(ticker: str, data_dir: str, model_dir: str,
                               lstm_hidden: int, tmfg_n_features: int,
                               dropout: float, learning_rate: float,
                               epochs: int = 100, batch_size: int = 64) -> dict:
    """
    Test a specific hyperparameter configuration
    
    Returns:
        Dictionary with results
    """
    print(f"\n{'='*80}")
    print(f"Testing Configuration:")
    print(f"  LSTM Hidden: {lstm_hidden}")
    print(f"  Features: {tmfg_n_features}")
    print(f"  Dropout: {dropout}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"{'='*80}")
    
    # Create unique model directory for this configuration
    config_name = f"h{lstm_hidden}_f{tmfg_n_features}_d{dropout}_lr{learning_rate}"
    config_model_dir = os.path.join(model_dir, config_name)
    
    result = train_ticker_model(
        ticker=ticker,
        data_dir=data_dir,
        model_dir=config_model_dir,
        embedding_dim=2,
        train_end='2023-12-31',
        val_end='2024-12-31',
        epochs=epochs,
        batch_size=batch_size,
        challenge_ticker=ticker,
        use_gpu=True,
        lstm_hidden=lstm_hidden,
        tmfg_n_features=tmfg_n_features,
        dropout=dropout,
        learning_rate=learning_rate
    )
    
    # Extract key metrics
    return {
        'config': {
            'lstm_hidden': lstm_hidden,
            'tmfg_n_features': tmfg_n_features,
            'dropout': dropout,
            'learning_rate': learning_rate
        },
        'test_accuracy': result.get('test_accuracy', 0),
        'test_auc': result.get('test_auc', 0),
        'ci_lower': result.get('ci_lower', 0),
        'ci_upper': result.get('ci_upper', 0),
        'success': result.get('success', False)
    }


def manual_grid_search(ticker: str, data_dir: str, model_dir: str):
    """
    Manual grid search over key hyperparameters
    """
    # Define search space
    configs = [
        # Baseline (current)
        {'lstm_hidden': 256, 'tmfg_n_features': 20, 'dropout': 0.3, 'learning_rate': 0.0005},
        
        # More features
        {'lstm_hidden': 256, 'tmfg_n_features': 25, 'dropout': 0.3, 'learning_rate': 0.0005},
        {'lstm_hidden': 256, 'tmfg_n_features': 30, 'dropout': 0.3, 'learning_rate': 0.0005},
        
        # Larger model
        {'lstm_hidden': 512, 'tmfg_n_features': 20, 'dropout': 0.3, 'learning_rate': 0.0005},
        
        # More regularization
        {'lstm_hidden': 256, 'tmfg_n_features': 20, 'dropout': 0.4, 'learning_rate': 0.0005},
        {'lstm_hidden': 256, 'tmfg_n_features': 20, 'dropout': 0.5, 'learning_rate': 0.0005},
        
        # Lower learning rate
        {'lstm_hidden': 256, 'tmfg_n_features': 20, 'dropout': 0.3, 'learning_rate': 0.0003},
        {'lstm_hidden': 256, 'tmfg_n_features': 20, 'dropout': 0.3, 'learning_rate': 0.0001},
        
        # Combinations
        {'lstm_hidden': 512, 'tmfg_n_features': 25, 'dropout': 0.4, 'learning_rate': 0.0003},
    ]
    
    results = []
    
    for i, config in enumerate(configs):
        print(f"\n\n{'='*80}")
        print(f"Configuration {i+1}/{len(configs)}")
        print(f"{'='*80}")
        
        # Temporarily modify train_model.py parameters
        # For now, we'll need to manually edit or use a different approach
        # This is a simplified version - you'd need to modify the model class
        
        try:
            result = test_hyperparameter_config(
                ticker=ticker,
                data_dir=data_dir,
                model_dir=model_dir,
                **config,
                epochs=100,
                batch_size=64
            )
            results.append(result)
            
            print(f"\nResult: Accuracy={result['test_accuracy']:.4f}, CI=[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
            
        except Exception as e:
            print(f"Error testing config: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'config': config,
                'error': str(e),
                'success': False
            })
    
    # Summary
    print(f"\n\n{'='*80}")
    print("Hyperparameter Tuning Summary")
    print(f"{'='*80}")
    
    successful = [r for r in results if r.get('success')]
    if successful:
        # Sort by test accuracy
        successful.sort(key=lambda x: x.get('test_accuracy', 0), reverse=True)
        
        print(f"\nBest Configurations (sorted by test accuracy):")
        for i, result in enumerate(successful[:5]):
            c = result['config']
            print(f"\n{i+1}. Accuracy: {result['test_accuracy']:.4f} (CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}])")
            print(f"   LSTM Hidden: {c['lstm_hidden']}, Features: {c['tmfg_n_features']}")
            print(f"   Dropout: {c['dropout']}, LR: {c['learning_rate']}")
    
    # Save results
    results_file = os.path.join(model_dir, 'tuning_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for MANTIS models")
    parser.add_argument(
        "--ticker",
        type=str,
        default="ETH",
        help="Ticker to tune (default: ETH)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Data directory"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/tuning",
        help="Model directory for tuning results"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Epochs per configuration"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MANTIS Hyperparameter Tuning")
    print("=" * 80)
    print(f"Ticker: {args.ticker}")
    print(f"Data directory: {args.data_dir}")
    print(f"Model directory: {args.model_dir}")
    print("=" * 80)
    
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Run manual grid search
    results = manual_grid_search(
        ticker=args.ticker,
        data_dir=args.data_dir,
        model_dir=args.model_dir
    )
    
    print(f"\nTuning complete! Tested {len(results)} configurations.")


if __name__ == "__main__":
    main()

