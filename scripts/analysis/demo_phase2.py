#!/usr/bin/env python3
"""
Phase 2 Results Demonstration
============================

This script demonstrates the results of Phase 2 by loading and displaying
the generated embeddings, clustering results, and key insights.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

# Load configuration
import config


def load_results():
    """Load all Phase 2 results"""
    data_dir = Path(config.OUTPUT_DIR)
    
    # Load embeddings
    embeddings = np.load(data_dir / "blog_embeddings.npy")
    
    # Load post data
    posts_df = pd.read_csv(data_dir / "extracted_posts.csv")
    posts_df = posts_df[posts_df['extraction_success'] == True].copy()
    
    # Load clustering results
    with open(data_dir / "clustering_results.json", 'r') as f:
        clustering_results = json.load(f)
    
    # Load cluster assignments
    cluster_labels = pd.read_csv(data_dir / "cluster_labels.csv")
    
    return embeddings, posts_df, clustering_results, cluster_labels


def show_embedding_summary(embeddings, posts_df):
    """Display embedding summary statistics"""
    print("=" * 60)
    print("EMBEDDING SUMMARY")
    print("=" * 60)
    
    print(f"Number of posts: {len(embeddings)}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")
    print(f"Embedding data type: {embeddings.dtype}")
    print(f"Memory usage: {embeddings.nbytes / (1024**2):.1f} MB")
    
    # Embedding statistics
    print(f"\nEmbedding Statistics:")
    print(f"  Mean magnitude: {np.linalg.norm(embeddings, axis=1).mean():.3f}")
    print(f"  Std magnitude: {np.linalg.norm(embeddings, axis=1).std():.3f}")
    print(f"  Min value: {embeddings.min():.3f}")
    print(f"  Max value: {embeddings.max():.3f}")
    
    # Post length vs embedding relationship
    word_counts = posts_df['word_count'].values
    magnitudes = np.linalg.norm(embeddings, axis=1)
    correlation = np.corrcoef(word_counts, magnitudes)[0, 1]
    print(f"  Correlation with word count: {correlation:.3f}")


def show_clustering_summary(clustering_results):
    """Display clustering results summary"""
    print("\n" + "=" * 60)
    print("CLUSTERING SUMMARY")
    print("=" * 60)
    
    for algorithm, results in clustering_results.items():
        if 'optimization' in algorithm:
            continue
            
        print(f"\n{algorithm.upper()} Results:")
        
        if algorithm == 'hierarchical':
            for linkage_method, result in results.items():
                print(f"  {linkage_method.capitalize()} Linkage:")
                print(f"    Clusters: {result['n_clusters']}")
                print(f"    Silhouette Score: {result['silhouette_score']:.3f}")
                print(f"    Calinski-Harabasz: {result['calinski_harabasz_score']:.1f}")
                
                # Cluster sizes
                sizes = list(result['cluster_sizes'].values())
                print(f"    Cluster sizes: {min(sizes)}-{max(sizes)} posts")
        
        else:
            print(f"  Clusters: {results.get('n_clusters', 'N/A')}")
            print(f"  Silhouette Score: {results.get('silhouette_score', 0):.3f}")
            
            if 'calinski_harabasz_score' in results:
                print(f"  Calinski-Harabasz: {results['calinski_harabasz_score']:.1f}")
            
            if 'n_noise' in results:
                print(f"  Noise points: {results['n_noise']}")
            
            # Cluster sizes
            if 'cluster_sizes' in results:
                sizes = list(results['cluster_sizes'].values())
                if sizes:
                    print(f"  Cluster sizes: {min(sizes)}-{max(sizes)} posts")


def show_cluster_examples(posts_df, cluster_labels):
    """Show example posts from each cluster"""
    print("\n" + "=" * 60)
    print("CLUSTER EXAMPLES (K-means)")
    print("=" * 60)
    
    if 'kmeans_cluster' not in cluster_labels.columns:
        print("No K-means clustering results found.")
        return
    
    # Merge post data with cluster labels
    merged_df = posts_df.merge(cluster_labels[['post_id', 'kmeans_cluster']], on='post_id')
    
    # Show examples from each cluster
    for cluster_id in sorted(merged_df['kmeans_cluster'].unique()):
        cluster_posts = merged_df[merged_df['kmeans_cluster'] == cluster_id]
        
        print(f"\nCluster {cluster_id} ({len(cluster_posts)} posts):")
        print(f"  Average word count: {cluster_posts['word_count'].mean():.0f}")
        
        # Show 3 sample titles
        sample_titles = cluster_posts['title'].head(3).tolist()
        for i, title in enumerate(sample_titles, 1):
            print(f"  {i}. {title[:80]}{'...' if len(title) > 80 else ''}")


def show_optimization_results(clustering_results):
    """Show parameter optimization results"""
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    
    # K-means optimization
    if 'kmeans_optimization' in clustering_results:
        opt_results = clustering_results['kmeans_optimization']
        optimal_clusters = opt_results.get('optimal_clusters', {})
        
        print("K-means Optimal Cluster Counts:")
        for method, k in optimal_clusters.items():
            print(f"  {method.replace('_', ' ').title()}: {k} clusters")
    
    # DBSCAN optimization
    if 'dbscan_optimization' in clustering_results:
        dbscan_opt = clustering_results['dbscan_optimization']
        best_params = dbscan_opt.get('best_params')
        
        print(f"\nDBSCAN Optimal Parameters:")
        if best_params:
            print(f"  eps: {best_params[0]}, min_samples: {best_params[1]}")
            print(f"  Best silhouette score: {dbscan_opt.get('best_silhouette', 0):.3f}")
        else:
            print("  No optimal parameters found (likely all noise)")


def show_file_summary():
    """Show summary of generated files"""
    print("\n" + "=" * 60)
    print("GENERATED FILES")
    print("=" * 60)
    
    data_dir = Path(config.OUTPUT_DIR)
    
    # Core files
    core_files = [
        "blog_embeddings.npy",
        "embedding_metadata.csv", 
        "clustering_results.json",
        "cluster_labels.csv"
    ]
    
    print("Core Data Files:")
    for filename in core_files:
        filepath = data_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  ✅ {filename} ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ {filename} (missing)")
    
    # Visualizations
    plots_dir = data_dir / "plots"
    if plots_dir.exists():
        plot_files = list(plots_dir.glob("*"))
        print(f"\nVisualization Files:")
        print(f"  📊 {len(plot_files)} plots generated")
        
        # Group by type
        plot_types = {}
        for plot_file in plot_files:
            if 'scatter' in plot_file.name:
                plot_types.setdefault('Scatter plots', []).append(plot_file.name)
            elif 'interactive' in plot_file.name:
                plot_types.setdefault('Interactive plots', []).append(plot_file.name)
            elif 'optimization' in plot_file.name:
                plot_types.setdefault('Optimization plots', []).append(plot_file.name)
            else:
                plot_types.setdefault('Other plots', []).append(plot_file.name)
        
        for plot_type, files in plot_types.items():
            print(f"    {plot_type}: {len(files)} files")
    
    # Models
    models_dir = data_dir / "models"
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pkl"))
        print(f"\nSaved Models:")
        print(f"  🤖 {len(model_files)} clustering models")


def main():
    """Run the demonstration"""
    print("🚀 Phase 2 Results Demonstration")
    print("Timestamp:", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    try:
        # Load results
        print("\nLoading Phase 2 results...")
        embeddings, posts_df, clustering_results, cluster_labels = load_results()
        
        # Show summaries
        show_embedding_summary(embeddings, posts_df)
        show_clustering_summary(clustering_results)
        show_cluster_examples(posts_df, cluster_labels)
        show_optimization_results(clustering_results)
        show_file_summary()
        
        print("\n" + "=" * 60)
        print("PHASE 2 DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("🎉 All components working successfully!")
        print(f"📁 Results location: {config.OUTPUT_DIR}")
        print("📋 See PHASE2_README.md for detailed documentation")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: Required file not found - {e}")
        print("💡 Make sure to run Phase 2 first:")
        print("   python run_phase2.py --mock-embeddings")
        
    except Exception as e:
        print(f"\n❌ Error loading results: {e}")
        print("💡 Try regenerating the results with:")
        print("   python run_phase2.py --mock-embeddings")


if __name__ == "__main__":
    main()
