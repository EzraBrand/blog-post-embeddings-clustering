#!/usr/bin/env python3
"""
Phase 2 Pipeline Runner
======================

This script runs the complete Phase 2 pipeline for blog post clustering:
1. Generate embeddings using OpenAI API
2. Perform comprehensive clustering analysis
3. Create visualizations

Usage:
    python run_phase2.py [--mock-embeddings] [--skip-embeddings] [--skip-clustering] [--skip-visualization]
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Load configuration
import config

# Configure logging
log_file = Path(config.OUTPUT_DIR) / f"phase2_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def check_prerequisites():
    """Check if all prerequisites are met"""
    logger.info("Checking prerequisites...")
    
    # Check if extracted data exists
    data_dir = Path(config.OUTPUT_DIR)
    posts_file = data_dir / "extracted_posts.csv"
    
    if not posts_file.exists():
        logger.error(f"Blog posts data not found at {posts_file}")
        logger.error("Please run Phase 1 (HTML extraction) first")
        return False
    
    # Check OpenAI API key (only if not using mock embeddings)
    api_key = os.getenv(config.OPENAI_API_KEY_ENV)
    if not api_key:
        logger.warning(f"OpenAI API key not found in {config.OPENAI_API_KEY_ENV} environment variable")
        logger.warning("You can still run with --mock-embeddings for testing")
        return "no_api_key"
    
    logger.info("✅ Prerequisites check passed")
    return True


def run_embedding_generation(use_mock: bool = False, chunk_long_posts: bool = True):
    """Run embedding generation step"""
    logger.info("="*60)
    logger.info("STEP 1: EMBEDDING GENERATION")
    logger.info("="*60)
    
    if use_mock:
        logger.info("Generating mock embeddings for testing...")
        from test_phase2 import load_test_data, generate_mock_embeddings
        import numpy as np
        
        # Load data and generate mock embeddings
        post_data = load_test_data()
        embeddings = generate_mock_embeddings(post_data, n_dims=config.EMBEDDING_DIMENSIONS)
        
        # Save mock embeddings
        output_dir = Path(config.OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True)
        
        embeddings_file = output_dir / "blog_embeddings.npy"
        np.save(embeddings_file, embeddings)
        
        # Create mock metadata
        import pandas as pd
        metadata_df = pd.DataFrame({
            'post_id': post_data['post_id'],
            'post_index': range(len(post_data)),
            'original_length': post_data['content_length'],
            'chunks_used': 1,
            'chunking_method': 'mock',
            'embedding_success': True,
            'error_message': None
        })
        metadata_df.to_csv(output_dir / "embedding_metadata.csv")
        
        logger.info(f"✅ Mock embeddings generated: {embeddings.shape}")
        return True
    
    else:
        logger.info("Generating real embeddings using OpenAI API...")
        
        try:
            # Import and run embedding generation
            from generate_embeddings import main as generate_main
            
            # Set command line arguments
            original_argv = sys.argv
            sys.argv = ['generate_embeddings.py']
            if chunk_long_posts:
                sys.argv.append('--chunk-long-posts')
            
            # Run embedding generation
            generate_main()
            
            # Restore original argv
            sys.argv = original_argv
            
            logger.info("✅ Real embeddings generated successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            return False


def run_clustering_analysis():
    """Run clustering analysis step"""
    logger.info("="*60)
    logger.info("STEP 2: CLUSTERING ANALYSIS")
    logger.info("="*60)
    
    try:
        # Import and run clustering analysis
        from clustering_analysis import main as clustering_main
        
        # Set command line arguments
        original_argv = sys.argv
        sys.argv = ['clustering_analysis.py']
        
        # Run clustering analysis
        clustering_main()
        
        # Restore original argv
        sys.argv = original_argv
        
        logger.info("✅ Clustering analysis completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Clustering analysis failed: {e}")
        return False


def run_visualization():
    """Run visualization step"""
    logger.info("="*60)
    logger.info("STEP 3: VISUALIZATION GENERATION")
    logger.info("="*60)
    
    try:
        # Import and run visualization
        from visualize_clusters import main as viz_main
        
        # Set command line arguments
        original_argv = sys.argv
        sys.argv = ['visualize_clusters.py']
        
        # Run visualization
        viz_main()
        
        # Restore original argv
        sys.argv = original_argv
        
        logger.info("✅ Visualizations generated successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Visualization generation failed: {e}")
        return False


def print_results_summary():
    """Print summary of generated results"""
    logger.info("="*60)
    logger.info("PHASE 2 RESULTS SUMMARY")
    logger.info("="*60)
    
    output_dir = Path(config.OUTPUT_DIR)
    
    # Check generated files
    files_to_check = [
        ("blog_embeddings.npy", "Embedding vectors"),
        ("embedding_metadata.csv", "Embedding metadata"),
        ("clustering_results.json", "Clustering results"),
        ("cluster_labels.csv", "Cluster assignments"),
        ("plots/", "Visualization plots")
    ]
    
    logger.info("Generated files:")
    for filename, description in files_to_check:
        filepath = output_dir / filename
        if filepath.exists():
            if filepath.is_dir():
                n_plots = len(list(filepath.glob("*")))
                logger.info(f"  ✅ {description}: {n_plots} files in {filename}")
            else:
                size_mb = filepath.stat().st_size / (1024 * 1024)
                logger.info(f"  ✅ {description}: {filename} ({size_mb:.1f} MB)")
        else:
            logger.info(f"  ❌ {description}: {filename} (not found)")
    
    # Show key results if available
    results_file = output_dir / "clustering_results.json"
    if results_file.exists():
        import json
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        logger.info("\nClustering Results:")
        
        if 'kmeans' in results:
            kmeans = results['kmeans']
            logger.info(f"  K-means: {kmeans['n_clusters']} clusters, "
                       f"silhouette score: {kmeans['silhouette_score']:.3f}")
        
        if 'hierarchical' in results:
            hier = results['hierarchical']
            for method, result in hier.items():
                logger.info(f"  Hierarchical ({method}): {result['n_clusters']} clusters, "
                           f"silhouette score: {result['silhouette_score']:.3f}")
        
        if 'dbscan' in results:
            dbscan = results['dbscan']
            logger.info(f"  DBSCAN: {dbscan['n_clusters']} clusters, "
                       f"{dbscan['n_noise']} noise points, "
                       f"silhouette score: {dbscan['silhouette_score']:.3f}")
    
    logger.info(f"\nAll results saved to: {output_dir}")
    logger.info(f"Log file: {log_file}")


def main():
    parser = argparse.ArgumentParser(description="Run the complete Phase 2 pipeline")
    parser.add_argument(
        '--mock-embeddings',
        action='store_true',
        help="Use mock embeddings instead of OpenAI API (for testing)"
    )
    parser.add_argument(
        '--skip-embeddings',
        action='store_true',
        help="Skip embedding generation (use existing embeddings)"
    )
    parser.add_argument(
        '--skip-clustering',
        action='store_true',
        help="Skip clustering analysis"
    )
    parser.add_argument(
        '--skip-visualization',
        action='store_true',
        help="Skip visualization generation"
    )
    parser.add_argument(
        '--no-chunk-long-posts',
        action='store_true',
        help="Don't chunk very long posts for embeddings"
    )
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting Phase 2: Blog Post Embedding Generation and Clustering Analysis")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    
    # Check prerequisites
    prereq_status = check_prerequisites()
    if prereq_status is False:
        logger.error("❌ Prerequisites not met. Exiting.")
        sys.exit(1)
    elif prereq_status == "no_api_key" and not args.mock_embeddings:
        logger.error("❌ No OpenAI API key found. Use --mock-embeddings or set OPENAI_API_KEY")
        sys.exit(1)
    
    success_steps = []
    failed_steps = []
    
    # Step 1: Embedding Generation
    if not args.skip_embeddings:
        if run_embedding_generation(
            use_mock=args.mock_embeddings, 
            chunk_long_posts=not args.no_chunk_long_posts
        ):
            success_steps.append("Embedding Generation")
        else:
            failed_steps.append("Embedding Generation")
            logger.error("❌ Embedding generation failed. Stopping pipeline.")
            sys.exit(1)
    else:
        logger.info("⏭️  Skipping embedding generation (using existing embeddings)")
    
    # Step 2: Clustering Analysis
    if not args.skip_clustering:
        if run_clustering_analysis():
            success_steps.append("Clustering Analysis")
        else:
            failed_steps.append("Clustering Analysis")
            # Continue with visualization even if clustering fails
    else:
        logger.info("⏭️  Skipping clustering analysis")
    
    # Step 3: Visualization
    if not args.skip_visualization:
        if run_visualization():
            success_steps.append("Visualization Generation")
        else:
            failed_steps.append("Visualization Generation")
    else:
        logger.info("⏭️  Skipping visualization generation")
    
    # Final summary
    logger.info("="*60)
    logger.info("PHASE 2 PIPELINE COMPLETED")
    logger.info("="*60)
    
    if success_steps:
        logger.info(f"✅ Successful steps: {', '.join(success_steps)}")
    
    if failed_steps:
        logger.error(f"❌ Failed steps: {', '.join(failed_steps)}")
    
    if not failed_steps:
        logger.info("🎉 Phase 2 completed successfully!")
        print_results_summary()
        
        print("\n" + "="*60)
        print("PHASE 2 COMPLETED SUCCESSFULLY! 🎉")
        print("="*60)
        print(f"Results saved to: {config.OUTPUT_DIR}")
        print("\nGenerated:")
        print("📊 Blog post embeddings")
        print("🔍 Clustering analysis (K-means, Hierarchical, DBSCAN)")
        print("📈 Comprehensive visualizations")
        print("📋 Detailed reports and metrics")
        
        print("\nNext steps (Phase 3):")
        print("- Advanced topic modeling")
        print("- Interactive exploration tool")
        print("- Semantic search capabilities")
        
    else:
        logger.error("❌ Phase 2 completed with errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
