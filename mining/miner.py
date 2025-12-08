"""
MANTIS Miner
Main mining workflow that generates embeddings, encrypts, and uploads to R2
"""
import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import bittensor as bt
import boto3
from botocore.exceptions import ClientError

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.training.model_architecture import VMDTMFGLSTMXGBoost
from scripts.training.train_model import load_data
from generate_and_encrypt import generate_v2
import config


class MantisMiner:
    """MANTIS Miner"""
    
    def __init__(
        self,
        wallet: bt.wallet,
        subtensor: bt.subtensor,
        model_dir: str = "models/checkpoints",
        data_dir: str = "data/raw",
        r2_bucket: Optional[str] = None,
        r2_access_key: Optional[str] = None,
        r2_secret_key: Optional[str] = None,
        r2_endpoint: Optional[str] = None
    ):
        """
        Initialize miner
        
        Args:
            wallet: Bittensor wallet
            subtensor: Bittensor subtensor connection
            model_dir: Directory with trained models
            data_dir: Data directory
            r2_bucket: R2 bucket name
            r2_access_key: R2 access key
            r2_secret_key: R2 secret key
            r2_endpoint: R2 endpoint URL
        """
        self.wallet = wallet
        self.subtensor = subtensor
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.hotkey = wallet.hotkey.ss58_address
        
        # Initialize R2 client
        self.r2_client = None
        if r2_bucket and r2_access_key and r2_secret_key:
            self.r2_bucket = r2_bucket
            self.r2_client = boto3.client(
                's3',
                endpoint_url=r2_endpoint or "https://<account-id>.r2.cloudflarestorage.com",
                aws_access_key_id=r2_access_key,
                aws_secret_access_key=r2_secret_key
            )
        
        # Load models for each challenge
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load trained models for each challenge"""
        print("Loading models...")
        for challenge in config.CHALLENGES:
            ticker = challenge['ticker']
            embedding_dim = challenge['dim']
            
            model_path = os.path.join(self.model_dir, ticker)
            if os.path.exists(model_path):
                try:
                    model = VMDTMFGLSTMXGBoost(embedding_dim=embedding_dim)
                    model.load(model_path)
                    self.models[ticker] = model
                    print(f"  ✓ Loaded model for {ticker}")
                except Exception as e:
                    print(f"  ✗ Failed to load model for {ticker}: {e}")
            else:
                print(f"  ✗ Model not found for {ticker} at {model_path}")
    
    def get_latest_data(self, ticker: str, lookback_hours: int = 24) -> Optional[pd.DataFrame]:
        """
        Get latest data for a ticker
        
        Args:
            ticker: Ticker symbol
            lookback_hours: Hours to look back
            
        Returns:
            DataFrame with recent data
        """
        # Load data
        try:
            df, funding_df, oi_df, cross_exchange_df = load_data(
                ticker, self.data_dir, 
                start_date=(datetime.now() - pd.Timedelta(hours=lookback_hours)).strftime("%Y-%m-%d"),
                end_date=datetime.now().strftime("%Y-%m-%d")
            )
            
            if df.empty:
                return None
            
            # Get last N hours
            cutoff = df['datetime'].max() - pd.Timedelta(hours=lookback_hours)
            df = df[df['datetime'] >= cutoff].copy()
            
            return df, funding_df, oi_df, cross_exchange_df
        except Exception as e:
            print(f"Error loading data for {ticker}: {e}")
            return None
    
    def generate_embeddings(self) -> Dict[str, List[float]]:
        """
        Generate embeddings for all challenges
        
        Returns:
            Dictionary mapping ticker to embedding vector
        """
        embeddings = {}
        
        for challenge in config.CHALLENGES:
            ticker = challenge['ticker']
            
            if ticker not in self.models:
                # Use zero vector if model not available
                embeddings[ticker] = [0.0] * challenge['dim']
                continue
            
            # Get latest data
            data = self.get_latest_data(ticker, lookback_hours=24)
            if data is None:
                embeddings[ticker] = [0.0] * challenge['dim']
                continue
            
            df, funding_df, oi_df, cross_exchange_df = data
            
            # Prepare data
            model = self.models[ticker]
            try:
                X_seq, _, _ = model.prepare_data(df, funding_df, oi_df, cross_exchange_df)
                
                if len(X_seq) == 0:
                    embeddings[ticker] = [0.0] * challenge['dim']
                    continue
                
                # Generate embedding for most recent sequence
                embedding = model.predict_embeddings(X_seq[-1:])[0]
                embeddings[ticker] = embedding.tolist()
                
            except Exception as e:
                print(f"Error generating embedding for {ticker}: {e}")
                embeddings[ticker] = [0.0] * challenge['dim']
        
        return embeddings
    
    def encrypt_and_upload(self, embeddings: Dict[str, List[float]], lock_seconds: int = 30) -> Optional[str]:
        """
        Encrypt embeddings and upload to R2
        
        Args:
            embeddings: Dictionary of embeddings
            lock_seconds: Time lock seconds
            
        Returns:
            Public URL if successful, None otherwise
        """
        # Convert embeddings to list format
        embedding_list = [embeddings.get(c['ticker'], [0.0] * c['dim']) for c in config.CHALLENGES]
        
        # Generate encrypted payload
        try:
            payload = generate_v2(
                hotkey=self.hotkey,
                lock_seconds=lock_seconds,
                owner_pk_hex=config.OWNER_HPKE_PUBLIC_KEY_HEX,
                payload_text=None,
                embeddings=embedding_list
            )
        except Exception as e:
            print(f"Error encrypting payload: {e}")
            return None
        
        # Upload to R2
        if self.r2_client is None:
            print("R2 client not configured, saving locally...")
            local_path = f"mining/payloads/{self.hotkey}"
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'w') as f:
                json.dump(payload, f)
            return f"file://{os.path.abspath(local_path)}"
        
        # Upload to R2 (hotkey as key, no subdirs)
        try:
            payload_json = json.dumps(payload, separators=(',', ':'))
            payload_bytes = payload_json.encode('utf-8')
            
            # Check size
            if len(payload_bytes) > 25 * 1024 * 1024:  # 25MB
                print(f"Warning: Payload size {len(payload_bytes)} exceeds 25MB")
            
            self.r2_client.put_object(
                Bucket=self.r2_bucket,
                Key=self.hotkey,  # Hotkey as key, no subdirs
                Body=payload_bytes,
                ContentType='application/json'
            )
            
            # Construct public URL
            public_url = f"https://{self.r2_bucket}.r2.cloudflarestorage.com/{self.hotkey}"
            print(f"✓ Uploaded payload to {public_url}")
            return public_url
            
        except ClientError as e:
            print(f"Error uploading to R2: {e}")
            return None
    
    def commit_url(self, url: str):
        """
        Commit URL to subtensor
        
        Args:
            url: Public URL of payload
        """
        try:
            self.subtensor.commit(
                wallet=self.wallet,
                netuid=config.NETUID,
                data=url
            )
            print(f"✓ Committed URL to subtensor: {url}")
        except Exception as e:
            print(f"Error committing URL: {e}")
    
    def run_once(self, commit: bool = False):
        """Run mining cycle once"""
        print(f"\n{'='*80}")
        print(f"Mining Cycle - {datetime.now()}")
        print(f"{'='*80}")
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.generate_embeddings()
        print(f"  Generated embeddings for {len(embeddings)} challenges")
        
        # Encrypt and upload
        print("Encrypting and uploading...")
        url = self.encrypt_and_upload(embeddings, lock_seconds=30)
        
        if url and commit:
            # Commit URL (only once, or when URL changes)
            print("Committing URL to subtensor...")
            self.commit_url(url)
        
        return url
    
    def run_loop(self, interval_seconds: int = 60, commit: bool = False):
        """
        Run mining loop
        
        Args:
            interval_seconds: Seconds between cycles
            commit: Whether to commit URL to subtensor
        """
        print("=" * 80)
        print("MANTIS Miner - Starting Mining Loop")
        print("=" * 80)
        print(f"Hotkey: {self.hotkey}")
        print(f"Interval: {interval_seconds} seconds")
        print(f"Commit URL: {commit}")
        print("=" * 80)
        
        # Commit URL once at start if enabled
        if commit:
            url = self.run_once(commit=True)
            if url:
                print(f"\n✓ Initial URL committed: {url}")
                print("  (URL will not be committed again unless it changes)")
        
        # Mining loop
        while True:
            try:
                self.run_once(commit=False)  # Don't commit every cycle
                time.sleep(interval_seconds)
            except KeyboardInterrupt:
                print("\n\nStopping miner...")
                break
            except Exception as e:
                print(f"Error in mining cycle: {e}")
                time.sleep(interval_seconds)


def main():
    parser = argparse.ArgumentParser(description="MANTIS Miner")
    parser.add_argument(
        "--wallet.name",
        type=str,
        default="default",
        help="Wallet name"
    )
    parser.add_argument(
        "--wallet.hotkey",
        type=str,
        default="default",
        help="Hotkey name"
    )
    parser.add_argument(
        "--netuid",
        type=int,
        default=config.NETUID,
        help="Subnet UID"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/checkpoints",
        help="Model directory"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Data directory"
    )
    parser.add_argument(
        "--r2-bucket",
        type=str,
        default=None,
        help="R2 bucket name"
    )
    parser.add_argument(
        "--r2-access-key",
        type=str,
        default=None,
        help="R2 access key"
    )
    parser.add_argument(
        "--r2-secret-key",
        type=str,
        default=None,
        help="R2 secret key"
    )
    parser.add_argument(
        "--r2-endpoint",
        type=str,
        default=None,
        help="R2 endpoint URL"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Mining interval in seconds"
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Commit URL to subtensor"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit"
    )
    
    args = parser.parse_args()
    
    # Initialize wallet and subtensor
    wallet = bt.wallet(name=args.wallet.name, hotkey=args.wallet.hotkey)
    subtensor = bt.subtensor(network="finney")
    
    # Initialize miner
    miner = MantisMiner(
        wallet=wallet,
        subtensor=subtensor,
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        r2_bucket=args.r2_bucket,
        r2_access_key=args.r2_access_key,
        r2_secret_key=args.r2_secret_key,
        r2_endpoint=args.r2_endpoint
    )
    
    # Run
    if args.once:
        miner.run_once(commit=args.commit)
    else:
        miner.run_loop(interval_seconds=args.interval, commit=args.commit)


if __name__ == "__main__":
    main()

