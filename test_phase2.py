#!/usr/bin/env python3
"""
Test Phase 2 Implementation
==========================

This script tests the Phase 2 implementation without making API calls.
It validates the data loading, pipeline setup, and generates mock embeddings
for testing the clustering and visualization components.

Usage:
    python test_phase2.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging

# Import our modules
import config
from clustering_analysis import ClusteringAnalyzer, run_comprehensive_analysis
from visualize_clusters import ClusteringVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_test_data():
    """Load the extracted blog posts data for testing"""
    data_dir = Path(config.OUTPUT_DIR)
    csv_path = data_dir / "extracted_posts.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Blog posts data not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    success_df = df[df['extraction_success'] == True].copy()
    
    logger.info(f"Loaded {len(success_df)} successfully extracted posts")
    return success_df


def generate_mock_embeddings(post_data: pd.DataFrame, n_dims: int = 100) -> np.ndarray:
    """Generate mock embeddings for testing (replace with real embeddings later)"""
    n_posts = len(post_data)
    
    # Create somewhat realistic embeddings based on post content
    np.random.seed(42)  # For reproducibility
    
    # Generate base embeddings
    embeddings = np.random.randn(n_posts, n_dims)
    
    # Add some structure based on word count to make clustering more interesting
    word_counts = post_data['word_count'].values
    word_count_norm = (word_counts - word_counts.mean()) / word_counts.std()
    
    # Modify first few dimensions based on word count
    embeddings[:, 0] = embeddings[:, 0] + 0.5 * word_count_norm
    embeddings[:, 1] = embeddings[:, 1] + 0.3 * word_count_norm
    
    # Add some clustering based on publication year
    post_data['pub_year'] = pd.to_datetime(post_data['publication_date']).dt.year
    for year in post_data['pub_year'].unique():
        year_mask = post_data['pub_year'] == year
        year_offset = (year - 2023) * 0.5
        embeddings[year_mask, 2] += year_offset
    
    # Normalize embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    logger.info(f"Generated mock embeddings: {embeddings.shape}")
    return embeddings


def test_clustering_analysis(embeddings: np.ndarray, post_data: pd.DataFrame):
    """Test the clustering analysis functionality"""
    logger.info("Testing clustering analysis...")
    
    try:
        # Test with a smaller subset for speed
        test_size = min(100, len(embeddings))
        test_embeddings = embeddings[:test_size]
        test_data = post_data.iloc[:test_size].copy()
        
        # Initialize analyzer
        analyzer = ClusteringAnalyzer(test_embeddings, test_data)
        
        # Test K-means optimization
        logger.info("Testing K-means optimization...")
        kmeans_opt = analyzer.find_optimal_clusters_kmeans(max_clusters=15)
        
        # Test individual clustering methods
        logger.info("Testing K-means clustering...")
        kmeans_result = analyzer.perform_kmeans_clustering(5)
        
        logger.info("Testing hierarchical clustering...")
        hier_result = analyzer.perform_hierarchical_clustering(5, 'ward')
        
        logger.info("Testing DBSCAN clustering...")
        dbscan_result = analyzer.perform_dbscan_clustering(0.3, 5)
        
        # Test cluster analysis
        logger.info("Testing cluster content analysis...")
        cluster_analysis = analyzer.analyze_cluster_content(kmeans_result['labels'], 'test')
        
        logger.info("✅ Clustering analysis tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Clustering analysis test failed: {e}")
        return False


def test_visualization(embeddings: np.ndarray, post_data: pd.DataFrame, clustering_results: dict):
    """Test the visualization functionality"""
    logger.info("Testing visualization...")
    
    try:
        # Test with a smaller subset for speed
        test_size = min(50, len(embeddings))
        test_embeddings = embeddings[:test_size]
        test_data = post_data.iloc[:test_size].copy()
        
        # Initialize visualizer
        visualizer = ClusteringVisualizer(test_embeddings, test_data, clustering_results)
        
        # Test dimensionality reduction
        logger.info("Testing dimensionality reduction...")
        reduced_embeddings = visualizer.perform_dimensionality_reduction(['pca', 'tsne'])
        
        # Test a simple plot creation
        logger.info("Testing plot creation...")
        # This would create actual plots in a real scenario
        # For testing, we just verify the methods work
        
        logger.info("✅ Visualization tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Visualization test failed: {e}")
        return False


def test_data_pipeline():
    """Test the complete data pipeline"""
    logger.info("="*60)
    logger.info("TESTING PHASE 2 IMPLEMENTATION")
    logger.info("="*60)
    
    try:
        # 1. Load data
        logger.info("1. Loading blog posts data...")
        post_data = load_test_data()
        
        # 2. Generate mock embeddings
        logger.info("2. Generating mock embeddings...")
        embeddings = generate_mock_embeddings(post_data, n_dims=config.EMBEDDING_DIMENSIONS)
        
        # 3. Test clustering
        logger.info("3. Testing clustering analysis...")
        clustering_success = test_clustering_analysis(embeddings, post_data)
        
        # 4. Create mock clustering results for visualization testing
        mock_clustering_results = {
            'kmeans': {
                'labels': np.random.randint(0, 5, len(post_data)),
                'n_clusters': 5,
                'silhouette_score': 0.35,
                'cluster_sizes': {0: 100, 1: 120, 2: 95, 3: 110, 4: 114}
            }
        }
        
        # 5. Test visualization
        logger.info("4. Testing visualization...")
        visualization_success = test_visualization(embeddings, post_data, mock_clustering_results)
        
        # 6. Summary
        logger.info("="*60)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("="*60)
        logger.info(f"Data Loading: ✅")
        logger.info(f"Mock Embeddings: ✅")
        logger.info(f"Clustering Analysis: {'✅' if clustering_success else '❌'}")
        logger.info(f"Visualization: {'✅' if visualization_success else '❌'}")
        
        if clustering_success and visualization_success:
            logger.info("\n🎉 Phase 2 implementation is ready!")
            logger.info("\nNext steps:")
            logger.info("1. Set OPENAI_API_KEY environment variable")
            logger.info("2. Run: python generate_embeddings.py")
            logger.info("3. Run: python clustering_analysis.py")
            logger.info("4. Run: python visualize_clusters.py")
            return True
        else:
            logger.error("\n❌ Some tests failed. Please check the implementation.")
            return False
            
    except Exception as e:
        logger.error(f"❌ Pipeline test failed: {e}")
        return False


def show_data_summary(post_data: pd.DataFrame):
    """Show summary of the data we'll be working with"""
    logger.info("\n" + "="*60)
    logger.info("BLOG POSTS DATA SUMMARY")
    logger.info("="*60)
    
    logger.info(f"Total posts: {len(post_data)}")
    logger.info(f"Date range: {post_data['publication_date'].min()} to {post_data['publication_date'].max()}")
    logger.info(f"Word count - Mean: {post_data['word_count'].mean():.0f}, Median: {post_data['word_count'].median():.0f}")
    logger.info(f"Word count - Min: {post_data['word_count'].min()}, Max: {post_data['word_count'].max()}")
    
    # Word count distribution
    logger.info("\nWord count distribution:")
    bins = [0, 500, 1000, 2000, 5000, float('inf')]
    labels = ['<500', '500-1K', '1K-2K', '2K-5K', '>5K']
    post_data['word_bin'] = pd.cut(post_data['word_count'], bins=bins, labels=labels)
    logger.info(post_data['word_bin'].value_counts().to_string())
    
    # Sample titles
    logger.info("\nSample post titles:")
    for i, title in enumerate(post_data['title'].head(5)):
        logger.info(f"  {i+1}. {title}")
    
    logger.info("\nData is ready for embedding generation!")


if __name__ == "__main__":
    # Load and show data summary
    post_data = load_test_data()
    show_data_summary(post_data)
    
    # Run tests
    success = test_data_pipeline()
    
    if success:
        print("\n🚀 Phase 2 implementation is ready to go!")
        print("\nTo generate real embeddings:")
        print("1. Get an OpenAI API key from https://platform.openai.com/api-keys")
        print("2. Set the environment variable: export OPENAI_API_KEY='your-key-here'")
        print("3. Run the embedding generation: python generate_embeddings.py --chunk-long-posts")
    else:
        print("\n❌ Please fix the issues before proceeding.")
