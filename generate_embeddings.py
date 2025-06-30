#!/usr/bin/env python3
"""
Blog Post Embedding Generation
=============================

This script generates high-quality embeddings for the extracted blog posts using OpenAI's
text-embedding-3-large model. Handles academic content with Hebrew/English mixed text
and very long posts through intelligent chunking.

Usage:
    python generate_embeddings.py [--chunk-long-posts] [--force-regenerate]
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import json
import time
import logging
from datetime import datetime
import joblib

# OpenAI imports
try:
    from openai import OpenAI
except ImportError:
    print("Error: OpenAI library not installed. Run: pip install openai")
    sys.exit(1)

# Load configuration
import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(config.OUTPUT_DIR) / 'embedding_generation.log'),
        logging.StreamHandler() if config.LOG_TO_CONSOLE else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)


class BlogPostEmbedder:
    """Handles embedding generation for blog posts"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the embedder with OpenAI client"""
        self.api_key = api_key or os.getenv(config.OPENAI_API_KEY_ENV)
        if not self.api_key:
            raise ValueError(f"OpenAI API key not found. Set {config.OPENAI_API_KEY_ENV} environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = config.EMBEDDING_MODEL
        self.max_chunk_size = config.MAX_CHUNK_SIZE
        self.chunk_overlap = config.CHUNK_OVERLAP
        self.batch_size = config.EMBEDDING_BATCH_SIZE
        
        # Track API usage
        self.total_tokens = 0
        self.api_calls = 0
        
        logger.info(f"Initialized embedder with model: {self.model}")
    
    def chunk_text(self, text: str, max_size: int = None) -> List[str]:
        """Split very long text into overlapping chunks"""
        max_size = max_size or self.max_chunk_size
        
        if len(text) <= max_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_size
            
            # If we're not at the end, try to break at a natural boundary
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                chunk_end = text.rfind('.', start + max_size // 2, end)
                if chunk_end == -1:
                    chunk_end = text.rfind('\n', start + max_size // 2, end)
                if chunk_end == -1:
                    chunk_end = text.rfind(' ', start + max_size // 2, end)
                if chunk_end != -1:
                    end = chunk_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = max(start + 1, end - self.chunk_overlap)
            
            if start >= len(text):
                break
        
        return chunks
    
    def get_embedding(self, text: str, retries: int = None) -> np.ndarray:
        """Get embedding for a single text with retry logic"""
        retries = retries or config.OPENAI_MAX_RETRIES
        
        for attempt in range(retries + 1):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text,
                    encoding_format="float"
                )
                
                self.api_calls += 1
                self.total_tokens += response.usage.total_tokens
                
                return np.array(response.data[0].embedding)
                
            except Exception as e:
                if attempt < retries:
                    wait_time = config.OPENAI_RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"API call failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to get embedding after {retries + 1} attempts: {e}")
                    raise
    
    def get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for a batch of texts"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
                encoding_format="float"
            )
            
            self.api_calls += 1
            self.total_tokens += response.usage.total_tokens
            
            return [np.array(embedding.embedding) for embedding in response.data]
            
        except Exception as e:
            logger.warning(f"Batch embedding failed, falling back to individual calls: {e}")
            return [self.get_embedding(text) for text in texts]
    
    def embed_post(self, post_text: str, chunk_long_posts: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        Generate embedding for a blog post.
        Returns embedding vector and metadata about the process.
        """
        metadata = {
            'original_length': len(post_text),
            'chunks_used': 1,
            'chunking_method': 'none'
        }
        
        # Handle very long posts
        if chunk_long_posts and len(post_text) > self.max_chunk_size:
            chunks = self.chunk_text(post_text)
            metadata.update({
                'chunks_used': len(chunks),
                'chunking_method': 'overlapping',
                'chunk_sizes': [len(chunk) for chunk in chunks]
            })
            
            logger.debug(f"Split long post into {len(chunks)} chunks")
            
            # Get embeddings for all chunks
            chunk_embeddings = []
            for i, chunk in enumerate(chunks):
                try:
                    embedding = self.get_embedding(chunk)
                    chunk_embeddings.append(embedding)
                except Exception as e:
                    logger.error(f"Failed to embed chunk {i+1}/{len(chunks)}: {e}")
                    # Skip failed chunks and continue
                    continue
            
            if not chunk_embeddings:
                raise ValueError("All chunks failed to embed")
            
            # Average the chunk embeddings
            final_embedding = np.mean(chunk_embeddings, axis=0)
            metadata['successful_chunks'] = len(chunk_embeddings)
            
        else:
            # Single embedding for the full text (truncate if too long)
            if len(post_text) > self.max_chunk_size:
                post_text = post_text[:self.max_chunk_size]
                metadata['truncated'] = True
                metadata['truncated_length'] = len(post_text)
            
            final_embedding = self.get_embedding(post_text)
        
        return final_embedding, metadata


def load_blog_posts(data_dir: str = None) -> pd.DataFrame:
    """Load the extracted blog posts data"""
    data_dir = data_dir or config.OUTPUT_DIR
    csv_path = Path(data_dir) / "extracted_posts.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Blog posts data not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Filter successful extractions
    success_df = df[df['extraction_success'] == True].copy()
    logger.info(f"Loaded {len(success_df)} successfully extracted posts out of {len(df)} total")
    
    return success_df


def generate_embeddings(
    df: pd.DataFrame,
    output_dir: str = None,
    chunk_long_posts: bool = False,
    force_regenerate: bool = False
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Generate embeddings for all blog posts.
    
    Returns:
        - embeddings: numpy array of shape (n_posts, embedding_dim)
        - metadata_df: DataFrame with embedding metadata
    """
    output_dir = Path(output_dir or config.OUTPUT_DIR)
    embeddings_file = output_dir / "blog_embeddings.npy"
    metadata_file = output_dir / "embedding_metadata.csv"
    
    # Check if embeddings already exist
    if not force_regenerate and embeddings_file.exists() and metadata_file.exists():
        logger.info("Found existing embeddings, loading from disk...")
        embeddings = np.load(embeddings_file)
        metadata_df = pd.read_csv(metadata_file, index_col=0)
        
        if len(embeddings) == len(df):
            logger.info(f"Loaded {len(embeddings)} embeddings from disk")
            return embeddings, metadata_df
        else:
            logger.warning("Embedding count mismatch, regenerating...")
    
    # Initialize embedder
    embedder = BlogPostEmbedder()
    
    # Prepare data
    posts_to_embed = df['extracted_text'].tolist()
    post_ids = df['post_id'].tolist()
    
    embeddings = []
    embedding_metadata = []
    
    logger.info(f"Generating embeddings for {len(posts_to_embed)} posts...")
    
    # Process posts with progress bar
    for i, (post_id, post_text) in enumerate(tqdm(
        zip(post_ids, posts_to_embed), 
        total=len(posts_to_embed),
        desc="Generating embeddings"
    )):
        try:
            embedding, metadata = embedder.embed_post(post_text, chunk_long_posts)
            embeddings.append(embedding)
            
            # Add post ID and index to metadata
            metadata.update({
                'post_id': post_id,
                'post_index': i,
                'embedding_success': True,
                'error_message': None
            })
            embedding_metadata.append(metadata)
            
        except Exception as e:
            logger.error(f"Failed to embed post {post_id}: {e}")
            
            # Add zero embedding for failed post
            embeddings.append(np.zeros(config.EMBEDDING_DIMENSIONS))
            embedding_metadata.append({
                'post_id': post_id,
                'post_index': i,
                'original_length': len(post_text),
                'chunks_used': 0,
                'chunking_method': 'failed',
                'embedding_success': False,
                'error_message': str(e)
            })
        
        # Add a small delay to be respectful to the API
        if i % 10 == 0 and i > 0:
            time.sleep(0.1)
    
    # Convert to numpy array
    embeddings = np.array(embeddings)
    
    # Create metadata DataFrame
    metadata_df = pd.DataFrame(embedding_metadata)
    metadata_df.index = df.index
    
    # Save embeddings and metadata
    output_dir.mkdir(exist_ok=True)
    np.save(embeddings_file, embeddings)
    metadata_df.to_csv(metadata_file)
    
    # Log statistics
    successful_embeddings = metadata_df['embedding_success'].sum()
    logger.info(f"Generated {successful_embeddings}/{len(embeddings)} successful embeddings")
    logger.info(f"Total API calls: {embedder.api_calls}")
    logger.info(f"Total tokens used: {embedder.total_tokens}")
    
    # Save generation report
    report = {
        'generation_date': datetime.now().isoformat(),
        'total_posts': len(embeddings),
        'successful_embeddings': int(successful_embeddings),
        'failed_embeddings': len(embeddings) - int(successful_embeddings),
        'api_calls': embedder.api_calls,
        'total_tokens': embedder.total_tokens,
        'model_used': config.EMBEDDING_MODEL,
        'chunk_long_posts': chunk_long_posts,
        'config': {
            'max_chunk_size': config.MAX_CHUNK_SIZE,
            'chunk_overlap': config.CHUNK_OVERLAP,
            'embedding_dimensions': config.EMBEDDING_DIMENSIONS
        }
    }
    
    with open(output_dir / "embedding_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Saved embeddings to {embeddings_file}")
    logger.info(f"Saved metadata to {metadata_file}")
    
    return embeddings, metadata_df


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for blog posts")
    parser.add_argument(
        '--chunk-long-posts', 
        action='store_true',
        help="Split very long posts into chunks and average their embeddings"
    )
    parser.add_argument(
        '--force-regenerate',
        action='store_true', 
        help="Force regeneration even if embeddings already exist"
    )
    parser.add_argument(
        '--data-dir',
        default=config.OUTPUT_DIR,
        help="Directory containing the extracted posts data"
    )
    parser.add_argument(
        '--output-dir',
        default=config.OUTPUT_DIR,
        help="Directory to save embeddings"
    )
    
    args = parser.parse_args()
    
    try:
        # Load data
        logger.info("Loading blog posts data...")
        df = load_blog_posts(args.data_dir)
        
        # Generate embeddings
        embeddings, metadata = generate_embeddings(
            df,
            output_dir=args.output_dir,
            chunk_long_posts=args.chunk_long_posts,
            force_regenerate=args.force_regenerate
        )
        
        logger.info("Embedding generation completed successfully!")
        print(f"\nEmbedding Summary:")
        print(f"- Generated embeddings for {len(embeddings)} posts")
        print(f"- Embedding dimensions: {embeddings.shape[1]}")
        print(f"- Successful embeddings: {metadata['embedding_success'].sum()}")
        print(f"- Output saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
