"""
Comprehensive Hyperparameter Tuning for All MANTIS Challenges
Tunes hyperparameters for each challenge type separately
"""

import os
import sys
import argparse
import json
from pathlib import Path
from itertools import product
import numpy as np
from datetime import datetime

# Set TensorFlow environment variables BEFORE importing TensorFlow
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

# Limit thread usage to prevent resource exhaustion
os.environ['OMP_NUM_THREADS'] = '8'  # Limit OpenMP threads

# Fix TensorFlow threading issues in containers
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTRA_OP_PARALLELISM'] = '8'  # TensorFlow intra-op parallelism
os.environ['TF_NUM_INTER_OP_PARALLELISM'] = '4'  # TensorFlow inter-op parallelism
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations that can cause threading issues

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.training.train_model import train_ticker_model
from scripts.training.check_training_results import check_model_results
import config


def make_json_serializable(obj):
    """Convert frozendict and other non-serializable objects to JSON-serializable format"""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, '__dict__'):
        # Handle frozendict and similar objects
        try:
            return dict(obj)
        except (TypeError, ValueError):
            return str(obj)
    else:
        return obj


def get_search_space(challenge_type='binary'):
    """
    Get hyperparameter search space based on challenge type
    
    Args:
        challenge_type: 'binary', 'lbfgs', or 'hitfirst'
    
    Returns:
        List of hyperparameter configurations
    """
    if challenge_type == 'binary':
        # Binary challenges: Focus on preventing overfitting and improving generalization
        search_space = [
            # Current best (baseline)
            {'lstm_hidden': 256, 'tmfg_n_features': 25, 'dropout': 0.3, 'learning_rate': 0.0005},
            
            # More regularization (for overfitting)
            {'lstm_hidden': 256, 'tmfg_n_features': 20, 'dropout': 0.4, 'learning_rate': 0.0005},
            {'lstm_hidden': 256, 'tmfg_n_features': 20, 'dropout': 0.5, 'learning_rate': 0.0005},
            {'lstm_hidden': 128, 'tmfg_n_features': 20, 'dropout': 0.4, 'learning_rate': 0.0005},
            
            # More features (for better signal)
            {'lstm_hidden': 256, 'tmfg_n_features': 30, 'dropout': 0.3, 'learning_rate': 0.0005},
            {'lstm_hidden': 256, 'tmfg_n_features': 35, 'dropout': 0.3, 'learning_rate': 0.0005},
            
            # Larger model
            {'lstm_hidden': 512, 'tmfg_n_features': 25, 'dropout': 0.3, 'learning_rate': 0.0005},
            {'lstm_hidden': 512, 'tmfg_n_features': 30, 'dropout': 0.4, 'learning_rate': 0.0003},
            
            # Lower learning rate (for stability)
            {'lstm_hidden': 256, 'tmfg_n_features': 25, 'dropout': 0.3, 'learning_rate': 0.0003},
            {'lstm_hidden': 256, 'tmfg_n_features': 25, 'dropout': 0.3, 'learning_rate': 0.0001},
            
            # Higher learning rate (for faster convergence)
            {'lstm_hidden': 256, 'tmfg_n_features': 25, 'dropout': 0.3, 'learning_rate': 0.001},
            
            # Balanced combinations
            {'lstm_hidden': 256, 'tmfg_n_features': 30, 'dropout': 0.4, 'learning_rate': 0.0003},
            {'lstm_hidden': 512, 'tmfg_n_features': 25, 'dropout': 0.4, 'learning_rate': 0.0003},
        ]
    elif challenge_type == 'lbfgs':
        # LBFGS challenges: 17D embeddings, may need different approach
        search_space = [
            # Baseline
            {'lstm_hidden': 256, 'tmfg_n_features': 25, 'dropout': 0.3, 'learning_rate': 0.0005},
            
            # Larger model for higher dimensionality
            {'lstm_hidden': 512, 'tmfg_n_features': 30, 'dropout': 0.3, 'learning_rate': 0.0005},
            {'lstm_hidden': 512, 'tmfg_n_features': 35, 'dropout': 0.3, 'learning_rate': 0.0003},
            
            # More features
            {'lstm_hidden': 256, 'tmfg_n_features': 30, 'dropout': 0.3, 'learning_rate': 0.0005},
            {'lstm_hidden': 256, 'tmfg_n_features': 35, 'dropout': 0.3, 'learning_rate': 0.0005},
            
            # More regularization
            {'lstm_hidden': 256, 'tmfg_n_features': 25, 'dropout': 0.4, 'learning_rate': 0.0005},
            {'lstm_hidden': 512, 'tmfg_n_features': 30, 'dropout': 0.4, 'learning_rate': 0.0003},
        ]
    elif challenge_type == 'hitfirst':
        # HITFIRST challenges: 3D embeddings
        search_space = [
            # Baseline
            {'lstm_hidden': 256, 'tmfg_n_features': 25, 'dropout': 0.3, 'learning_rate': 0.0005},
            
            # More regularization
            {'lstm_hidden': 256, 'tmfg_n_features': 20, 'dropout': 0.4, 'learning_rate': 0.0005},
            {'lstm_hidden': 256, 'tmfg_n_features': 25, 'dropout': 0.4, 'learning_rate': 0.0003},
            
            # More features
            {'lstm_hidden': 256, 'tmfg_n_features': 30, 'dropout': 0.3, 'learning_rate': 0.0005},
            
            # Larger model
            {'lstm_hidden': 512, 'tmfg_n_features': 25, 'dropout': 0.3, 'learning_rate': 0.0005},
        ]
    else:
        # Default: same as binary
        search_space = [
            {'lstm_hidden': 256, 'tmfg_n_features': 25, 'dropout': 0.3, 'learning_rate': 0.0005},
        ]
    
    return search_space


def tune_challenge(ticker: str, challenge: dict, data_dir: str, 
                   tuning_dir: str, epochs: int = 100, batch_size: int = 64,
                   min_accuracy: float = 0.55) -> dict:
    """
    Tune hyperparameters for a single challenge
    
    Returns:
        Dictionary with best configuration and results
    """
    print(f"\n{'='*80}")
    print(f"Tuning: {challenge['name']} ({ticker})")
    print(f"{'='*80}")
    print(f"Embedding dim: {challenge['dim']}")
    print(f"Loss function: {challenge['loss_func']}")
    print(f"{'='*80}")
    
    # Determine challenge type
    loss_func = challenge.get('loss_func', 'binary')
    if loss_func == 'lbfgs':
        challenge_type = 'lbfgs'
    elif loss_func == 'hitfirst':
        challenge_type = 'hitfirst'
    else:
        challenge_type = 'binary'
    
    # Get search space
    search_space = get_search_space(challenge_type)
    
    # Get data ticker (price_key if available, otherwise ticker)
    data_ticker = challenge.get('price_key', ticker)
    embedding_dim = challenge['dim']
    
    print(f"\nTesting {len(search_space)} configurations...")
    print(f"Data ticker: {data_ticker}")
    print(f"Challenge ticker: {ticker}")
    print()
    
    results = []
    
    for i, hp_config in enumerate(search_space):
        config_name = f"h{hp_config['lstm_hidden']}_f{hp_config['tmfg_n_features']}_d{hp_config['dropout']}_lr{hp_config['learning_rate']}"
        config_model_dir = os.path.join(tuning_dir, ticker, config_name)
        model_path = os.path.join(config_model_dir, ticker)
        
        print(f"\n[{i+1}/{len(search_space)}] Testing: {config_name}")
        print(f"  LSTM Hidden: {hp_config['lstm_hidden']}, Features: {hp_config['tmfg_n_features']}")
        print(f"  Dropout: {hp_config['dropout']}, LR: {hp_config['learning_rate']}")
        
        # Check if model already exists and is complete
        required_files = ['lstm_model.h5', 'xgb_model.json', 'scaler.pkl', 
                         'feature_indices.pkl', 'config.json']
        model_exists = os.path.exists(model_path)
        if model_exists:
            all_files_exist = all(os.path.exists(os.path.join(model_path, f)) for f in required_files)
            if all_files_exist:
                print(f"  ⏭️  Model already exists, skipping training...")
                train_result = {'success': True, 'skipped': True}
            else:
                print(f"  ⚠️  Model directory exists but incomplete, retraining...")
                train_result = None
        else:
            train_result = None
        
        try:
            # Train model if needed
            if train_result is None:
                train_result = train_ticker_model(
                    ticker=data_ticker,
                    data_dir=data_dir,
                    model_dir=config_model_dir,
                    embedding_dim=embedding_dim,
                    train_end='2023-12-31',
                    val_end='2024-12-31',
                    epochs=epochs,
                    batch_size=batch_size,
                    challenge_ticker=ticker,
                    use_gpu=True,
                    lstm_hidden=hp_config['lstm_hidden'],
                    tmfg_n_features=hp_config['tmfg_n_features'],
                    dropout=hp_config['dropout'],
                    learning_rate=hp_config['learning_rate']
                )
            
            if not train_result.get('success'):
                print(f"  ✗ Training failed: {train_result.get('error', 'Unknown error')}")
                results.append({
                    'config': hp_config,
                    'config_name': config_name,
                    'success': False,
                    'error': train_result.get('error', 'Training failed')
                })
                continue
            
            # Evaluate model
            eval_result = check_model_results(
                ticker=ticker,
                model_dir=config_model_dir,
                data_dir=data_dir,
                min_accuracy=min_accuracy
            )
            
            if eval_result.get('success'):
                result = {
                    'config': hp_config,
                    'config_name': config_name,
                    'success': True,
                    'test_accuracy': eval_result.get('test_accuracy', 0),
                    'test_auc': eval_result.get('test_auc', 0),
                    'ci_lower': eval_result.get('ci_lower', 0),
                    'ci_upper': eval_result.get('ci_upper', 0),
                    'passed': eval_result.get('passed', False),
                    'n_test_samples': eval_result.get('n_test_samples', 0)
                }
                
                print(f"  ✓ Accuracy: {result['test_accuracy']:.4f} (CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}])")
                if result.get('test_auc') and not np.isnan(result['test_auc']):
                    print(f"    AUC: {result['test_auc']:.4f}")
                print(f"    Passed: {'✓' if result['passed'] else '✗'}")
                
                results.append(result)
            else:
                print(f"  ✗ Evaluation failed: {eval_result.get('error', 'Unknown error')}")
                results.append({
                    'config': hp_config,
                    'config_name': config_name,
                    'success': False,
                    'error': eval_result.get('error', 'Evaluation failed')
                })
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'config': hp_config,
                'config_name': config_name,
                'success': False,
                'error': str(e)
            })
    
    # Find best configuration
    successful = [r for r in results if r.get('success')]
    
    if not successful:
        print(f"\n✗ No successful configurations for {ticker}")
        return {
            'ticker': ticker,
            'challenge_name': challenge['name'],
            'best_config': None,
            'best_result': None,
            'all_results': results
        }
    
    # Sort by test accuracy (or CI lower bound if available)
    successful.sort(key=lambda x: x.get('ci_lower', x.get('test_accuracy', 0)), reverse=True)
    
    best_result = successful[0]
    best_config = best_result['config']
    
    print(f"\n{'='*80}")
    print(f"Best Configuration for {ticker}:")
    print(f"{'='*80}")
    print(f"  LSTM Hidden: {best_config['lstm_hidden']}")
    print(f"  Features: {best_config['tmfg_n_features']}")
    print(f"  Dropout: {best_config['dropout']}")
    print(f"  Learning Rate: {best_config['learning_rate']}")
    print(f"  Accuracy: {best_result['test_accuracy']:.4f}")
    print(f"  CI: [{best_result['ci_lower']:.4f}, {best_result['ci_upper']:.4f}]")
    print(f"  Passed: {'✓' if best_result['passed'] else '✗'}")
    print(f"{'='*80}")
    
    return {
        'ticker': ticker,
        'challenge_name': challenge['name'],
        'challenge_type': challenge_type,
        'best_config': best_config,
        'best_result': best_result,
        'all_results': results,
        'n_configs_tested': len(search_space),
        'n_successful': len(successful)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive hyperparameter tuning for all MANTIS challenges"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Specific ticker to tune (default: all challenges)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Data directory"
    )
    parser.add_argument(
        "--tuning-dir",
        type=str,
        default="models/tuning",
        help="Directory for tuning results"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Epochs per configuration (default: 100)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size (default: 128 for GPU, use 64 for CPU)"
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.55,
        help="Minimum accuracy threshold (default: 0.55)"
    )
    parser.add_argument(
        "--challenge-type",
        type=str,
        default=None,
        choices=['binary', 'lbfgs', 'hitfirst'],
        help="Only tune specific challenge type (default: all)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MANTIS Comprehensive Hyperparameter Tuning")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Tuning directory: {args.tuning_dir}")
    print(f"Epochs per config: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Min accuracy: {args.min_accuracy}")
    
    # Verify GPU availability
    try:
        import tensorflow as tf

        # Fix TensorFlow threading issues in containers
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"\n✓ GPU Available: {len(gpus)} device(s)")
            for gpu in gpus:
                print(f"  - {gpu.name}")
            print(f"✓ Memory growth: {tf.config.experimental.get_memory_growth(gpus[0])}")
            try:
                policy = tf.keras.mixed_precision.global_policy()
                print(f"✓ Mixed precision: {policy.name}")
            except:
                pass
            print("✓ Threading configured for container environment")
        else:
            print("\n⚠️  WARNING: No GPU detected - training will be SLOW on CPU!")
            print("   Consider using a GPU instance for faster tuning.")
    except Exception as e:
        print(f"\n⚠️  GPU check failed: {e}")
    
    print("=" * 80)
    
    os.makedirs(args.tuning_dir, exist_ok=True)
    
    # Determine which challenges to tune
    if args.ticker:
        challenge = config.CHALLENGE_MAP.get(args.ticker)
        if not challenge:
            print(f"✗ Challenge not found: {args.ticker}")
            return
        challenges_to_tune = [(args.ticker, challenge)]
    else:
        challenges_to_tune = [(c['ticker'], c) for c in config.CHALLENGES]
        if args.challenge_type:
            challenges_to_tune = [
                (t, c) for t, c in challenges_to_tune 
                if c.get('loss_func') == args.challenge_type
            ]
    
    print(f"\nTuning {len(challenges_to_tune)} challenge(s)...")
    
    all_tuning_results = {}
    
    for ticker, challenge in challenges_to_tune:
        try:
            result = tune_challenge(
                ticker=ticker,
                challenge=challenge,
                data_dir=args.data_dir,
                tuning_dir=args.tuning_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                min_accuracy=args.min_accuracy
            )
            all_tuning_results[ticker] = result
        except Exception as e:
            print(f"\n✗ Error tuning {ticker}: {e}")
            import traceback
            traceback.print_exc()
            all_tuning_results[ticker] = {
                'ticker': ticker,
                'error': str(e),
                'success': False
            }
    
    # Summary
    print(f"\n\n{'='*80}")
    print("Tuning Summary")
    print(f"{'='*80}")
    
    successful = {k: v for k, v in all_tuning_results.items() if v.get('best_config')}
    failed = {k: v for k, v in all_tuning_results.items() if not v.get('best_config')}
    
    print(f"\nSuccessfully tuned: {len(successful)}/{len(challenges_to_tune)}")
    
    if successful:
        print(f"\nBest Configurations:")
        for ticker, result in sorted(successful.items()):
            best = result['best_result']
            hp_config = result['best_config']
            print(f"\n  {ticker} ({result['challenge_name']}):")
            print(f"    Accuracy: {best['test_accuracy']:.4f} (CI: [{best['ci_lower']:.4f}, {best['ci_upper']:.4f}])")
            print(f"    Config: h={hp_config['lstm_hidden']}, f={hp_config['tmfg_n_features']}, d={hp_config['dropout']}, lr={hp_config['learning_rate']}")
            print(f"    Passed: {'✓' if best['passed'] else '✗'}")
    
    if failed:
        print(f"\nFailed to tune: {len(failed)}")
        for ticker, result in failed.items():
            print(f"  {ticker}: {result.get('error', 'Unknown error')}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.tuning_dir, f'tuning_results_{timestamp}.json')
    with open(results_file, 'w') as f:
        json.dump(all_tuning_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")
    
    # Save best configs summary
    if successful:
        best_configs_file = os.path.join(args.tuning_dir, f'best_configs_{timestamp}.json')
        best_configs = {
            ticker: {
                'challenge_name': result['challenge_name'],
                'config': result['best_config'],
                'accuracy': result['best_result']['test_accuracy'],
                'ci_lower': result['best_result']['ci_lower'],
                'ci_upper': result['best_result']['ci_upper'],
                'passed': result['best_result']['passed']
            }
            for ticker, result in successful.items()
        }
        # Convert to JSON-serializable format
        best_configs = make_json_serializable(best_configs)
        with open(best_configs_file, 'w') as f:
            json.dump(best_configs, f, indent=2)
        print(f"Best configurations saved to {best_configs_file}")


if __name__ == "__main__":
    main()

