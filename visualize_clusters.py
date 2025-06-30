#!/usr/bin/env python3
"""
Blog Post Clustering Visualization
=================================

This script creates comprehensive visualizations for the clustering analysis results,
including dimensionality reduction plots (t-SNE, UMAP, PCA), cluster dendrograms,
and analysis dashboards.

Usage:
    python visualize_clusters.py [--results-file] [--embeddings-file] [--methods]
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import json
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Dimensionality reduction imports
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
try:
    import umap
except ImportError:
    print("Warning: UMAP not available. Install with: pip install umap-learn")
    umap = None

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Hierarchical clustering visualization
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

from tqdm import tqdm

# Load configuration
import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(config.OUTPUT_DIR) / 'visualization.log'),
        logging.StreamHandler() if config.LOG_TO_CONSOLE else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('default')
sns.set_palette("husl")


class ClusteringVisualizer:
    """Comprehensive visualization for clustering results"""
    
    def __init__(self, embeddings: np.ndarray, post_data: pd.DataFrame, clustering_results: Dict):
        """Initialize visualizer with data and results"""
        self.embeddings = embeddings
        self.post_data = post_data
        self.clustering_results = clustering_results
        
        # Create output directory for plots
        self.plots_dir = Path(config.OUTPUT_DIR) / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized visualizer for {len(embeddings)} posts")
    
    def perform_dimensionality_reduction(self, methods: List[str] = None) -> Dict[str, np.ndarray]:
        """Perform dimensionality reduction using specified methods"""
        methods = methods or config.REDUCTION_METHODS
        reduced_embeddings = {}
        
        logger.info("Performing dimensionality reduction...")
        
        # PCA (for preprocessing and standalone visualization)
        if 'pca' in methods:
            logger.info("Running PCA...")
            pca = PCA(n_components=config.PCA_N_COMPONENTS, random_state=42)
            pca_embeddings = pca.fit_transform(self.embeddings)
            
            # Take first 2 components for visualization
            reduced_embeddings['pca'] = pca_embeddings[:, :2]
            
            # Store full PCA for preprocessing other methods
            self.pca_embeddings = pca_embeddings
            
            logger.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_[:2]}")
        
        # t-SNE
        if 'tsne' in methods:
            logger.info("Running t-SNE...")
            
            # Use PCA preprocessing if available for large datasets
            input_data = self.pca_embeddings if hasattr(self, 'pca_embeddings') else self.embeddings
            
            tsne = TSNE(
                n_components=2,
                perplexity=min(config.TSNE_PERPLEXITY, len(self.embeddings) - 1),
                n_iter=config.TSNE_N_ITER,
                random_state=42,
                init='pca'
            )
            
            tsne_embeddings = tsne.fit_transform(input_data)
            reduced_embeddings['tsne'] = tsne_embeddings
        
        # UMAP
        if 'umap' in methods and umap:
            logger.info("Running UMAP...")
            
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(config.UMAP_N_NEIGHBORS, len(self.embeddings) - 1),
                min_dist=config.UMAP_MIN_DIST,
                random_state=42
            )
            
            umap_embeddings = reducer.fit_transform(self.embeddings)
            reduced_embeddings['umap'] = umap_embeddings
        
        return reduced_embeddings
    
    def create_cluster_scatter_plots(self, reduced_embeddings: Dict[str, np.ndarray]) -> None:
        """Create scatter plots for each clustering algorithm and reduction method"""
        logger.info("Creating cluster scatter plots...")
        
        # Prepare cluster labels
        cluster_data = {}
        
        if 'kmeans' in self.clustering_results:
            cluster_data['K-means'] = self.clustering_results['kmeans']['labels']
        
        if 'hierarchical' in self.clustering_results:
            for linkage_method, result in self.clustering_results['hierarchical'].items():
                cluster_data[f'Hierarchical ({linkage_method})'] = result['labels']
        
        if 'dbscan' in self.clustering_results:
            cluster_data['DBSCAN'] = self.clustering_results['dbscan']['labels']
        
        # Create plots for each combination
        for reduction_method, coords in reduced_embeddings.items():
            for cluster_method, labels in cluster_data.items():
                self._create_single_scatter_plot(
                    coords, labels, reduction_method, cluster_method
                )
    
    def _create_single_scatter_plot(self, coords: np.ndarray, labels: np.ndarray, 
                                  reduction_method: str, cluster_method: str) -> None:
        """Create a single scatter plot"""
        plt.figure(figsize=config.FIGURE_SIZE)
        
        # Handle noise points in DBSCAN
        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:  # Noise points in DBSCAN
                mask = labels == label
                plt.scatter(coords[mask, 0], coords[mask, 1], 
                          c='black', marker='x', s=20, alpha=0.6, label='Noise')
            else:
                mask = labels == label
                plt.scatter(coords[mask, 0], coords[mask, 1], 
                          c=[color], s=50, alpha=0.7, label=f'Cluster {label}')
        
        plt.title(f'{cluster_method} Clustering - {reduction_method.upper()} Visualization')
        plt.xlabel(f'{reduction_method.upper()} Component 1')
        plt.ylabel(f'{reduction_method.upper()} Component 2')
        
        # Only show legend if not too many clusters
        if len(unique_labels) <= 15:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        filename = f"cluster_scatter_{reduction_method}_{cluster_method.lower().replace(' ', '_').replace('(', '').replace(')', '')}.{config.PLOT_FORMAT}"
        filepath = self.plots_dir / filename
        plt.savefig(filepath, dpi=config.DPI, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Saved plot: {filepath}")
    
    def create_interactive_plots(self, reduced_embeddings: Dict[str, np.ndarray]) -> None:
        """Create interactive Plotly visualizations"""
        logger.info("Creating interactive plots...")
        
        # Prepare data for plotting
        plot_data = self.post_data.copy()
        
        # Add cluster labels
        if 'kmeans' in self.clustering_results:
            plot_data['kmeans_cluster'] = self.clustering_results['kmeans']['labels']
        
        if 'dbscan' in self.clustering_results:
            plot_data['dbscan_cluster'] = self.clustering_results['dbscan']['labels']
        
        # Create interactive plots for each reduction method
        for reduction_method, coords in reduced_embeddings.items():
            plot_data[f'{reduction_method}_x'] = coords[:, 0]
            plot_data[f'{reduction_method}_y'] = coords[:, 1]
            
            # Create plotly figure
            if 'kmeans_cluster' in plot_data.columns:
                fig = px.scatter(
                    plot_data,
                    x=f'{reduction_method}_x',
                    y=f'{reduction_method}_y',
                    color='kmeans_cluster',
                    hover_data=['title', 'word_count', 'publication_date'],
                    title=f'Blog Posts Clustering - {reduction_method.upper()} Visualization (K-means)',
                    labels={
                        f'{reduction_method}_x': f'{reduction_method.upper()} Component 1',
                        f'{reduction_method}_y': f'{reduction_method.upper()} Component 2',
                        'kmeans_cluster': 'Cluster'
                    }
                )
                
                fig.update_layout(
                    width=1000,
                    height=700,
                    showlegend=True
                )
                
                # Save interactive plot
                filename = f"interactive_clusters_{reduction_method}_kmeans.html"
                filepath = self.plots_dir / filename
                fig.write_html(str(filepath))
                
                logger.debug(f"Saved interactive plot: {filepath}")
    
    def create_cluster_size_plots(self) -> None:
        """Create bar plots showing cluster sizes for different algorithms"""
        logger.info("Creating cluster size plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        plot_idx = 0
        
        # K-means cluster sizes
        if 'kmeans' in self.clustering_results:
            cluster_sizes = self.clustering_results['kmeans']['cluster_sizes']
            clusters = list(cluster_sizes.keys())
            sizes = list(cluster_sizes.values())
            
            axes[plot_idx].bar(clusters, sizes, color='skyblue', alpha=0.7)
            axes[plot_idx].set_title('K-means Cluster Sizes')
            axes[plot_idx].set_xlabel('Cluster ID')
            axes[plot_idx].set_ylabel('Number of Posts')
            plot_idx += 1
        
        # Hierarchical cluster sizes (ward linkage if available)
        if 'hierarchical' in self.clustering_results:
            linkage_method = 'ward' if 'ward' in self.clustering_results['hierarchical'] else list(self.clustering_results['hierarchical'].keys())[0]
            cluster_sizes = self.clustering_results['hierarchical'][linkage_method]['cluster_sizes']
            clusters = list(cluster_sizes.keys())
            sizes = list(cluster_sizes.values())
            
            axes[plot_idx].bar(clusters, sizes, color='lightcoral', alpha=0.7)
            axes[plot_idx].set_title(f'Hierarchical Cluster Sizes ({linkage_method})')
            axes[plot_idx].set_xlabel('Cluster ID')
            axes[plot_idx].set_ylabel('Number of Posts')
            plot_idx += 1
        
        # DBSCAN cluster sizes
        if 'dbscan' in self.clustering_results:
            cluster_sizes = self.clustering_results['dbscan']['cluster_sizes']
            clusters = [str(c) for c in cluster_sizes.keys()]
            sizes = list(cluster_sizes.values())
            
            # Color noise points differently
            colors = ['red' if c == '-1' else 'lightgreen' for c in clusters]
            
            axes[plot_idx].bar(clusters, sizes, color=colors, alpha=0.7)
            axes[plot_idx].set_title('DBSCAN Cluster Sizes')
            axes[plot_idx].set_xlabel('Cluster ID (-1 = Noise)')
            axes[plot_idx].set_ylabel('Number of Posts')
            plot_idx += 1
        
        # Comparison plot
        if plot_idx < 4:
            comparison_data = {}
            
            if 'kmeans' in self.clustering_results:
                comparison_data['K-means'] = len(self.clustering_results['kmeans']['cluster_sizes'])
            
            if 'hierarchical' in self.clustering_results:
                for method, result in self.clustering_results['hierarchical'].items():
                    comparison_data[f'Hierarchical ({method})'] = len(result['cluster_sizes'])
            
            if 'dbscan' in self.clustering_results:
                n_clusters = len([c for c in self.clustering_results['dbscan']['cluster_sizes'].keys() if c != -1])
                comparison_data['DBSCAN'] = n_clusters
            
            if comparison_data:
                methods = list(comparison_data.keys())
                counts = list(comparison_data.values())
                
                axes[plot_idx].bar(methods, counts, color='gold', alpha=0.7)
                axes[plot_idx].set_title('Number of Clusters by Algorithm')
                axes[plot_idx].set_xlabel('Clustering Algorithm')
                axes[plot_idx].set_ylabel('Number of Clusters')
                axes[plot_idx].tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(plot_idx + 1, 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        filename = f"cluster_sizes_comparison.{config.PLOT_FORMAT}"
        filepath = self.plots_dir / filename
        plt.savefig(filepath, dpi=config.DPI, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Saved cluster size plots: {filepath}")
    
    def create_evaluation_metrics_plot(self) -> None:
        """Create plots comparing evaluation metrics across algorithms"""
        logger.info("Creating evaluation metrics plots...")
        
        # Collect metrics
        metrics_data = []
        
        if 'kmeans' in self.clustering_results:
            metrics_data.append({
                'Algorithm': 'K-means',
                'Silhouette Score': self.clustering_results['kmeans']['silhouette_score'],
                'Calinski-Harabasz Score': self.clustering_results['kmeans']['calinski_harabasz_score'],
                'Davies-Bouldin Score': self.clustering_results['kmeans']['davies_bouldin_score']
            })
        
        if 'hierarchical' in self.clustering_results:
            for linkage_method, result in self.clustering_results['hierarchical'].items():
                metrics_data.append({
                    'Algorithm': f'Hierarchical ({linkage_method})',
                    'Silhouette Score': result['silhouette_score'],
                    'Calinski-Harabasz Score': result['calinski_harabasz_score'],
                    'Davies-Bouldin Score': result['davies_bouldin_score']
                })
        
        if 'dbscan' in self.clustering_results:
            metrics_data.append({
                'Algorithm': 'DBSCAN',
                'Silhouette Score': self.clustering_results['dbscan']['silhouette_score'],
                'Calinski-Harabasz Score': self.clustering_results['dbscan']['calinski_harabasz_score'],
                'Davies-Bouldin Score': self.clustering_results['dbscan']['davies_bouldin_score']
            })
        
        if not metrics_data:
            logger.warning("No metrics data available for plotting")
            return
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Silhouette Score (higher is better)
        metrics_df.plot(x='Algorithm', y='Silhouette Score', kind='bar', ax=axes[0], color='skyblue')
        axes[0].set_title('Silhouette Score (Higher = Better)')
        axes[0].set_ylabel('Score')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Calinski-Harabasz Score (higher is better)
        metrics_df.plot(x='Algorithm', y='Calinski-Harabasz Score', kind='bar', ax=axes[1], color='lightcoral')
        axes[1].set_title('Calinski-Harabasz Score (Higher = Better)')
        axes[1].set_ylabel('Score')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Davies-Bouldin Score (lower is better)
        metrics_df.plot(x='Algorithm', y='Davies-Bouldin Score', kind='bar', ax=axes[2], color='lightgreen')
        axes[2].set_title('Davies-Bouldin Score (Lower = Better)')
        axes[2].set_ylabel('Score')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        filename = f"evaluation_metrics_comparison.{config.PLOT_FORMAT}"
        filepath = self.plots_dir / filename
        plt.savefig(filepath, dpi=config.DPI, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Saved evaluation metrics plot: {filepath}")
    
    def create_dendrogram(self, max_samples: int = 100) -> None:
        """Create hierarchical clustering dendrogram"""
        if len(self.embeddings) > max_samples:
            logger.info(f"Creating dendrogram with {max_samples} samples (subset of {len(self.embeddings)})")
            # Sample posts for dendrogram
            indices = np.random.choice(len(self.embeddings), max_samples, replace=False)
            sample_embeddings = self.embeddings[indices]
            sample_titles = self.post_data.iloc[indices]['title'].tolist()
        else:
            logger.info("Creating dendrogram with all samples")
            sample_embeddings = self.embeddings
            sample_titles = self.post_data['title'].tolist()
        
        # Calculate linkage
        linkage_matrix = linkage(sample_embeddings, method='ward')
        
        # Create dendrogram
        plt.figure(figsize=(15, 8))
        
        dendrogram(
            linkage_matrix,
            labels=sample_titles,
            leaf_rotation=90,
            leaf_font_size=8
        )
        
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Blog Posts')
        plt.ylabel('Distance')
        plt.tight_layout()
        
        filename = f"hierarchical_dendrogram.{config.PLOT_FORMAT}"
        filepath = self.plots_dir / filename
        plt.savefig(filepath, dpi=config.DPI, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Saved dendrogram: {filepath}")
    
    def create_cluster_word_clouds(self) -> None:
        """Create word clouds for each cluster (if wordcloud is available)"""
        try:
            from wordcloud import WordCloud
        except ImportError:
            logger.warning("WordCloud not available, skipping word cloud generation")
            return
        
        logger.info("Creating cluster word clouds...")
        
        # Only create word clouds for K-means for now
        if 'kmeans' not in self.clustering_results:
            return
        
        labels = self.clustering_results['kmeans']['labels']
        cluster_analysis = self.clustering_results['kmeans'].get('cluster_analysis', {})
        
        unique_clusters = np.unique(labels)
        
        # Create subplot grid
        n_clusters = len(unique_clusters)
        cols = min(3, n_clusters)
        rows = (n_clusters + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_clusters == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, cluster_id in enumerate(unique_clusters):
            row = i // cols
            col = i % cols
            
            # Get cluster posts
            cluster_mask = labels == cluster_id
            cluster_posts = self.post_data[cluster_mask]
            
            # Combine titles and extract sample content
            text_data = ' '.join(cluster_posts['title'].fillna('').astype(str))
            
            if text_data.strip():
                # Create word cloud
                wordcloud = WordCloud(
                    width=400, 
                    height=300, 
                    background_color='white',
                    colormap='viridis',
                    max_words=50
                ).generate(text_data)
                
                if rows > 1:
                    ax = axes[row, col]
                else:
                    ax = axes[col]
                
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.set_title(f'Cluster {cluster_id} ({len(cluster_posts)} posts)')
                ax.axis('off')
            else:
                if rows > 1:
                    axes[row, col].set_visible(False)
                else:
                    axes[col].set_visible(False)
        
        # Hide unused subplots
        for i in range(n_clusters, rows * cols):
            row = i // cols
            col = i % cols
            if rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.tight_layout()
        
        filename = f"cluster_wordclouds_kmeans.{config.PLOT_FORMAT}"
        filepath = self.plots_dir / filename
        plt.savefig(filepath, dpi=config.DPI, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Saved word clouds: {filepath}")
    
    def create_optimization_plots(self) -> None:
        """Create plots showing optimization results"""
        logger.info("Creating optimization plots...")
        
        # K-means optimization plot
        if 'kmeans_optimization' in self.clustering_results:
            self._plot_kmeans_optimization()
        
        # DBSCAN optimization plot
        if 'dbscan_optimization' in self.clustering_results:
            self._plot_dbscan_optimization()
    
    def _plot_kmeans_optimization(self) -> None:
        """Plot K-means optimization results"""
        opt_data = self.clustering_results['kmeans_optimization']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        cluster_range = opt_data['cluster_range']
        
        # Elbow method
        axes[0, 0].plot(cluster_range, opt_data['inertias'], 'bo-')
        axes[0, 0].set_title('Elbow Method for Optimal k')
        axes[0, 0].set_xlabel('Number of Clusters')
        axes[0, 0].set_ylabel('Inertia')
        
        # Mark optimal point if available
        if 'elbow' in opt_data['optimal_clusters']:
            optimal_k = opt_data['optimal_clusters']['elbow']
            optimal_idx = cluster_range.index(optimal_k)
            axes[0, 0].axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
            axes[0, 0].legend()
        
        # Silhouette scores
        axes[0, 1].plot(cluster_range, opt_data['silhouette_scores'], 'ro-')
        axes[0, 1].set_title('Silhouette Score vs Number of Clusters')
        axes[0, 1].set_xlabel('Number of Clusters')
        axes[0, 1].set_ylabel('Silhouette Score')
        
        if 'silhouette' in opt_data['optimal_clusters']:
            optimal_k = opt_data['optimal_clusters']['silhouette']
            axes[0, 1].axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
            axes[0, 1].legend()
        
        # Calinski-Harabasz scores
        axes[1, 0].plot(cluster_range, opt_data['calinski_harabasz_scores'], 'go-')
        axes[1, 0].set_title('Calinski-Harabasz Score vs Number of Clusters')
        axes[1, 0].set_xlabel('Number of Clusters')
        axes[1, 0].set_ylabel('Calinski-Harabasz Score')
        
        if 'calinski_harabasz' in opt_data['optimal_clusters']:
            optimal_k = opt_data['optimal_clusters']['calinski_harabasz']
            axes[1, 0].axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
            axes[1, 0].legend()
        
        # Davies-Bouldin scores
        axes[1, 1].plot(cluster_range, opt_data['davies_bouldin_scores'], 'mo-')
        axes[1, 1].set_title('Davies-Bouldin Score vs Number of Clusters')
        axes[1, 1].set_xlabel('Number of Clusters')
        axes[1, 1].set_ylabel('Davies-Bouldin Score')
        
        if 'davies_bouldin' in opt_data['optimal_clusters']:
            optimal_k = opt_data['optimal_clusters']['davies_bouldin']
            axes[1, 1].axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        filename = f"kmeans_optimization.{config.PLOT_FORMAT}"
        filepath = self.plots_dir / filename
        plt.savefig(filepath, dpi=config.DPI, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Saved K-means optimization plot: {filepath}")
    
    def _plot_dbscan_optimization(self) -> None:
        """Plot DBSCAN optimization results"""
        opt_data = self.clustering_results['dbscan_optimization']
        
        if not opt_data['all_results']:
            return
        
        # Extract data for heatmap
        results_df = pd.DataFrame([
            {
                'eps': r['eps'],
                'min_samples': r['min_samples'],
                'n_clusters': r['n_clusters'],
                'silhouette_score': r['silhouette_score'],
                'n_noise': r['n_noise']
            }
            for r in opt_data['all_results']
        ])
        
        # Create pivot tables for heatmaps
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Number of clusters heatmap
        clusters_pivot = results_df.pivot(index='min_samples', columns='eps', values='n_clusters')
        sns.heatmap(clusters_pivot, annot=True, fmt='d', cmap='viridis', ax=axes[0])
        axes[0].set_title('Number of Clusters')
        
        # Silhouette score heatmap
        silhouette_pivot = results_df.pivot(index='min_samples', columns='eps', values='silhouette_score')
        sns.heatmap(silhouette_pivot, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=axes[1])
        axes[1].set_title('Silhouette Score')
        
        # Noise ratio heatmap
        results_df['noise_ratio'] = results_df['n_noise'] / len(self.embeddings)
        noise_pivot = results_df.pivot(index='min_samples', columns='eps', values='noise_ratio')
        sns.heatmap(noise_pivot, annot=True, fmt='.3f', cmap='Reds', ax=axes[2])
        axes[2].set_title('Noise Ratio')
        
        plt.tight_layout()
        
        filename = f"dbscan_optimization.{config.PLOT_FORMAT}"
        filepath = self.plots_dir / filename
        plt.savefig(filepath, dpi=config.DPI, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Saved DBSCAN optimization plot: {filepath}")


def load_data_and_results(
    embeddings_file: str = None,
    results_file: str = None,
    data_dir: str = None
) -> Tuple[np.ndarray, pd.DataFrame, Dict]:
    """Load embeddings, post data, and clustering results"""
    data_dir = Path(data_dir or config.OUTPUT_DIR)
    
    # Load embeddings
    embeddings_file = embeddings_file or str(data_dir / "blog_embeddings.npy")
    if not os.path.exists(embeddings_file):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
    embeddings = np.load(embeddings_file)
    
    # Load post data
    posts_file = data_dir / "extracted_posts.csv"
    if not posts_file.exists():
        raise FileNotFoundError(f"Posts data not found: {posts_file}")
    post_data = pd.read_csv(posts_file)
    post_data = post_data[post_data['extraction_success'] == True].copy()
    
    # Load clustering results
    results_file = results_file or str(data_dir / "clustering_results.json")
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Clustering results not found: {results_file}")
    
    with open(results_file, 'r') as f:
        clustering_results = json.load(f)
    
    logger.info(f"Loaded data: {len(embeddings)} embeddings, {len(post_data)} posts")
    
    return embeddings, post_data, clustering_results


def create_all_visualizations(
    embeddings: np.ndarray,
    post_data: pd.DataFrame,
    clustering_results: Dict,
    methods: List[str] = None
) -> None:
    """Create all visualizations"""
    methods = methods or config.REDUCTION_METHODS
    
    visualizer = ClusteringVisualizer(embeddings, post_data, clustering_results)
    
    # Perform dimensionality reduction
    logger.info("Performing dimensionality reduction...")
    reduced_embeddings = visualizer.perform_dimensionality_reduction(methods)
    
    # Create all visualizations
    logger.info("Creating visualizations...")
    
    visualizer.create_cluster_scatter_plots(reduced_embeddings)
    visualizer.create_interactive_plots(reduced_embeddings)
    visualizer.create_cluster_size_plots()
    visualizer.create_evaluation_metrics_plot()
    visualizer.create_dendrogram()
    visualizer.create_cluster_word_clouds()
    visualizer.create_optimization_plots()
    
    logger.info(f"All visualizations saved to: {visualizer.plots_dir}")


def main():
    parser = argparse.ArgumentParser(description="Create visualizations for clustering results")
    parser.add_argument(
        '--embeddings-file',
        help="Path to embeddings numpy file"
    )
    parser.add_argument(
        '--results-file',
        help="Path to clustering results JSON file"
    )
    parser.add_argument(
        '--data-dir',
        default=config.OUTPUT_DIR,
        help="Directory containing data files"
    )
    parser.add_argument(
        '--methods',
        nargs='+',
        choices=['pca', 'tsne', 'umap'],
        default=['pca', 'tsne', 'umap'],
        help="Dimensionality reduction methods to use"
    )
    
    args = parser.parse_args()
    
    try:
        # Load data
        logger.info("Loading data and results...")
        embeddings, post_data, clustering_results = load_data_and_results(
            embeddings_file=args.embeddings_file,
            results_file=args.results_file,
            data_dir=args.data_dir
        )
        
        # Create visualizations
        logger.info("Creating all visualizations...")
        create_all_visualizations(
            embeddings,
            post_data,
            clustering_results,
            methods=args.methods
        )
        
        print("\n" + "="*60)
        print("VISUALIZATION GENERATION COMPLETED")
        print("="*60)
        print(f"All plots saved to: {Path(config.OUTPUT_DIR) / 'plots'}")
        print("\nGenerated visualizations:")
        print("- Cluster scatter plots (static)")
        print("- Interactive cluster plots (HTML)")
        print("- Cluster size comparisons")
        print("- Evaluation metrics comparison")
        print("- Hierarchical clustering dendrogram")
        print("- Cluster word clouds")
        print("- Algorithm optimization plots")
        
        logger.info("Visualization generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Visualization generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
