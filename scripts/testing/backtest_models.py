"""
Backtest Models on Historical Data

This script tests all trained models on recent historical data
to estimate performance before mainnet deployment.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.training.model_architecture import VMDTMFGLSTMXGBoost
from sklearn.metrics import accuracy_score, roc_auc_score

def backtest_challenge(challenge_name, data_file, model_path):
    """Backtest a single challenge on recent historical data"""
    
    print(f"\n{'='*60}")
    print(f"Backtesting: {challenge_name}")
    print(f"{'='*60}")
    
    # Load model
    try:
        model = VMDTMFGLSTMXGBoost.load(model_path)
        print("✓ Model loaded")
    except Exception as e:
        print(f"✗ Failed to load model: {str(e)[:100]}")
        return None
    
    # Load recent data (last 30 days)
    try:
        df = pd.read_csv(data_file)
        if 'datetime' not in df.columns and 'timestamp' in df.columns:
            df['datetime'] = df['timestamp']
        df['datetime'] = pd.to_datetime(df['datetime'])
    except Exception as e:
        print(f"✗ Failed to load data: {str(e)[:100]}")
        return None
    
    # Use last 30 days for backtesting
    cutoff_date = df['datetime'].max() - pd.Timedelta(days=30)
    test_df = df[df['datetime'] > cutoff_date].copy()
    
    print(f"Test period: {test_df['datetime'].min()} to {test_df['datetime'].max()}")
    print(f"Test samples: {len(test_df)}")
    
    # Prepare data
    try:
        X, y, features = model.prepare_data(test_df)
    except Exception as e:
        print(f"✗ Failed to prepare data: {str(e)[:100]}")
        return None
    
    if len(X) == 0:
        print("✗ No test sequences generated")
        return None
    
    print(f"Test sequences: {len(X)}")
    print(f"Features: {X.shape[2]}")
    
    # Generate predictions
    try:
        embeddings = model.predict_embeddings(X)
        print(f"Embedding shape: {embeddings.shape}")
    except Exception as e:
        print(f"✗ Failed to generate predictions: {str(e)[:100]}")
        return None
    
    # For binary challenges, calculate accuracy
    if 'BINARY' in challenge_name:
        try:
            predictions = model.predict_binary(X)
            actual = (y > 0).astype(int)
            
            accuracy = accuracy_score(actual, predictions)
            
            if len(np.unique(actual)) > 1 and len(np.unique(predictions)) > 1:
                auc = roc_auc_score(actual, embeddings[:, 1] if embeddings.shape[1] >= 2 else embeddings[:, 0])
            else:
                auc = 0.5
            
            print(f"\n📊 Results:")
            print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"  AUC: {auc:.4f}")
            print(f"  Predictions: Class 0: {(predictions==0).sum()}, Class 1: {(predictions==1).sum()}")
            
            # Estimate salience (higher accuracy = higher salience)
            estimated_salience = (accuracy - 0.5) * 4  # Rough estimate
            print(f"  Estimated Salience: {estimated_salience:.4f}")
            
            return {
                'challenge': challenge_name,
                'accuracy': float(accuracy),
                'auc': float(auc),
                'estimated_salience': float(estimated_salience),
                'samples': int(len(X)),
                'type': 'binary'
            }
        except Exception as e:
            print(f"✗ Failed to calculate metrics: {str(e)[:100]}")
            return None
    
    # For LBFGS challenges, check embedding distribution
    else:
        print(f"\n📊 Embedding Statistics:")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Mean (first 5): {embeddings.mean(axis=0)[:5]}")
        print(f"  Std (first 5): {embeddings.std(axis=0)[:5]}")
        print(f"  Min (first 5): {embeddings.min(axis=0)[:5]}")
        print(f"  Max (first 5): {embeddings.max(axis=0)[:5]}")
        
        # Check if embeddings are valid (not all zeros, reasonable range)
        is_valid = (
            embeddings.std() > 0.01 and
            not np.isnan(embeddings).any() and
            not np.isinf(embeddings).any()
        )
        
        print(f"  Valid embeddings: {'✓' if is_valid else '✗'}")
        
        # Rough salience estimate based on embedding variance
        estimated_salience = min(float(embeddings.std() * 2), 3.0)
        print(f"  Estimated Salience: {estimated_salience:.4f}")
        
        return {
            'challenge': challenge_name,
            'valid': is_valid,
            'embedding_std': float(embeddings.std()),
            'estimated_salience': float(estimated_salience),
            'samples': int(len(X)),
            'type': 'lbfgs'
        }

def run_all_backtests():
    """Run backtests for all challenges"""
    
    challenges = {
        'ETH-LBFGS': 'data/ETH_1h.csv',
        'BTC-LBFGS-6H': 'data/BTC_6h.csv',
        'ETH-HITFIRST-100M': 'data/ETH_1h.csv',
        'ETH-1H-BINARY': 'data/ETH_1h.csv',
        'EURUSD-1H-BINARY': 'data/EURUSD_1h.csv',
        'GBPUSD-1H-BINARY': 'data/GBPUSD_1h.csv',
        'CADUSD-1H-BINARY': 'data/CADUSD_1h.csv',
        'NZDUSD-1H-BINARY': 'data/NZDUSD_1h.csv',
        'CHFUSD-1H-BINARY': 'data/CHFUSD_1h.csv',
        'XAUUSD-1H-BINARY': 'data/XAUUSD_1h.csv',
        'XAGUSD-1H-BINARY': 'data/XAGUSD_1h.csv',
    }
    
    results = []
    
    print("=" * 80)
    print("🧪 MANTIS Model Backtesting Suite")
    print("=" * 80)
    
    for challenge, data_file in challenges.items():
        model_path = f'models/tuned/{challenge}'
        
        if not Path(model_path).exists():
            print(f"\n✗ {challenge}: Model not found at {model_path}")
            continue
        
        if not Path(data_file).exists():
            print(f"\n✗ {challenge}: Data file not found at {data_file}")
            continue
        
        try:
            result = backtest_challenge(challenge, data_file, model_path)
            if result:
                results.append(result)
        except Exception as e:
            print(f"\n✗ {challenge}: Error - {str(e)[:100]}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 BACKTEST SUMMARY")
    print(f"{'='*80}\n")
    
    binary_results = [r for r in results if r.get('type') == 'binary']
    lbfgs_results = [r for r in results if r.get('type') == 'lbfgs']
    
    # Binary challenges summary
    if binary_results:
        print("Binary Challenges:")
        for result in binary_results:
            print(f"  {result['challenge']}:")
            print(f"    Accuracy: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
            print(f"    AUC: {result['auc']:.4f}")
            print(f"    Est. Salience: {result['estimated_salience']:.4f}")
        
        avg_accuracy = np.mean([r['accuracy'] for r in binary_results])
        avg_salience = np.mean([r['estimated_salience'] for r in binary_results])
        print(f"\n  Average Accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
        print(f"  Average Est. Salience: {avg_salience:.4f}")
    
    # LBFGS challenges summary
    if lbfgs_results:
        print("\nLBFGS Challenges:")
        for result in lbfgs_results:
            print(f"  {result['challenge']}:")
            print(f"    Valid: {result['valid']}")
            print(f"    Embedding Std: {result['embedding_std']:.4f}")
            print(f"    Est. Salience: {result['estimated_salience']:.4f}")
        
        avg_salience = np.mean([r['estimated_salience'] for r in lbfgs_results])
        print(f"\n  Average Est. Salience: {avg_salience:.4f}")
    
    # Overall assessment
    print(f"\n{'='*80}")
    print("✅ MAINNET READINESS ASSESSMENT")
    print(f"{'='*80}\n")
    
    total_results = len(results)
    expected_results = 11
    
    print(f"Models tested: {total_results}/{expected_results}")
    
    if binary_results:
        avg_binary_acc = np.mean([r['accuracy'] for r in binary_results])
        print(f"Binary Avg Accuracy: {avg_binary_acc:.4f} ({avg_binary_acc*100:.2f}%)")
        
        if avg_binary_acc >= 0.70:
            print("  ✅ EXCELLENT - Ready for mainnet!")
        elif avg_binary_acc >= 0.60:
            print("  ✓ GOOD - Should perform well on mainnet")
        elif avg_binary_acc >= 0.55:
            print("  ⚠️ FAIR - Consider retraining weak challenges")
        else:
            print("  ❌ POOR - Recommend retraining before mainnet")
    
    all_salience = [r['estimated_salience'] for r in results]
    if all_salience:
        avg_salience = np.mean(all_salience)
        print(f"\nOverall Est. Salience: {avg_salience:.4f}")
        
        if avg_salience >= 2.0:
            print("  ✅ EXCELLENT - Competitive for top positions!")
        elif avg_salience >= 1.5:
            print("  ✓ GOOD - Should rank well")
        elif avg_salience >= 1.0:
            print("  ⚠️ FAIR - May need optimization")
        else:
            print("  ❌ POOR - Recommend improvement before mainnet")
    
    print(f"\n{'='*80}")
    print("📋 RECOMMENDATION")
    print(f"{'='*80}\n")
    
    if total_results >= 10 and avg_salience >= 1.5:
        print("✅ READY FOR MAINNET DEPLOYMENT")
        print("\nNext steps:")
        print("1. Save these baseline results")
        print("2. Follow Phase 2 in COMPLETE_ROADMAP_TO_FIRST_PLACE.md")
        print("3. Deploy to mainnet with confidence!")
    elif total_results >= 8 and avg_salience >= 1.2:
        print("✓ ACCEPTABLE FOR MAINNET")
        print("\nRecommendations:")
        print("1. Consider retraining 1-2 weakest challenges")
        print("2. Or proceed to mainnet and optimize later")
        print("3. Monitor closely after deployment")
    else:
        print("⚠️ RECOMMEND IMPROVEMENTS BEFORE MAINNET")
        print("\nAction plan:")
        print("1. Identify weakest challenges (see results above)")
        print("2. Retrain with: ./run_training.sh --trials 100 --challenge X")
        print("3. Re-run backtesting")
        print("4. Deploy when results improve")
    
    return results

if __name__ == "__main__":
    print("\n🧪 Starting backtest suite...\n")
    results = run_all_backtests()
    print(f"\n✅ Backtest complete! Tested {len(results)} challenges.\n")

