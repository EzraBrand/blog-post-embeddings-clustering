#!/usr/bin/env python3
"""
Blog Post Clustering Analysis
============================

This script performs comprehensive clustering analysis on the generated blog post embeddings
using multiple algorithms (K-means, hierarchical, DBSCAN) with parameter optimization
and evaluation metrics.

Usage:
    python clustering_analysis.py [--embeddings-file] [--algorithms] [--optimize-params]
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
from collections import Counter
import joblib

# Machine learning imports
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Text analysis imports
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from collections import Counter
    import re
except ImportError:
    print("Warning: NLTK not available, text analysis features will be limited")
    nltk = None

# Load configuration
import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(config.OUTPUT_DIR) / 'clustering_analysis.log'),
        logging.StreamHandler() if config.LOG_TO_CONSOLE else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)


class ClusteringAnalyzer:
    """Comprehensive clustering analysis for blog post embeddings"""
    
    def __init__(self, embeddings: np.ndarray, post_data: pd.DataFrame):
        """Initialize analyzer with embeddings and post metadata"""
        self.embeddings = embeddings
        self.post_data = post_data
        self.scaler = StandardScaler()
        
        # Normalize embeddings
        self.embeddings_normalized = self.scaler.fit_transform(embeddings)
        
        # Initialize NLTK if available
        if nltk:
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                self.stop_words = set(stopwords.words('english'))
            except:
                self.stop_words = set()
        else:
            self.stop_words = set()
        
        logger.info(f"Initialized analyzer with {len(embeddings)} embeddings")
    
    def find_optimal_clusters_kmeans(self, max_clusters: int = 50) -> Dict[str, Any]:
        """Find optimal number of clusters using elbow method and silhouette analysis"""
        logger.info("Finding optimal number of clusters for K-means...")
        
        cluster_range = range(
            config.CLUSTERING_ALGORITHMS['kmeans']['min_clusters'],
            min(max_clusters, config.CLUSTERING_ALGORITHMS['kmeans']['max_clusters']) + 1,
            config.CLUSTERING_ALGORITHMS['kmeans']['step']
        )
        
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        davies_bouldin_scores = []
        
        for n_clusters in tqdm(cluster_range, desc="Testing K-means clusters"):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.embeddings_normalized)
            
            inertias.append(kmeans.inertia_)
            
            if n_clusters > 1:
                silhouette_scores.append(silhouette_score(self.embeddings_normalized, cluster_labels))
                calinski_scores.append(calinski_harabasz_score(self.embeddings_normalized, cluster_labels))
                davies_bouldin_scores.append(davies_bouldin_score(self.embeddings_normalized, cluster_labels))
            else:
                silhouette_scores.append(0)
                calinski_scores.append(0)
                davies_bouldin_scores.append(float('inf'))
        
        # Find optimal clusters
        optimal_clusters = {}
        
        # Elbow method (look for the "elbow" in inertia curve)
        if len(inertias) > 2:
            # Simple elbow detection: maximum distance from line connecting first and last points
            n_points = len(inertias)
            coords = np.array([[i, inertias[i]] for i in range(n_points)])
            
            # Normalize coordinates
            coords[:, 0] = coords[:, 0] / coords[-1, 0]
            coords[:, 1] = coords[:, 1] / coords[0, 1]
            
            # Calculate distances from the line
            line_vec = coords[-1] - coords[0]
            line_vec_norm = line_vec / np.linalg.norm(line_vec)
            
            distances = []
            for point in coords:
                vec_to_point = point - coords[0]
                dist = np.linalg.norm(vec_to_point - np.dot(vec_to_point, line_vec_norm) * line_vec_norm)
                distances.append(dist)
            
            elbow_idx = np.argmax(distances)
            optimal_clusters['elbow'] = list(cluster_range)[elbow_idx]
        
        # Best silhouette score
        if silhouette_scores:
            best_silhouette_idx = np.argmax(silhouette_scores)
            optimal_clusters['silhouette'] = list(cluster_range)[best_silhouette_idx]
        
        # Best Calinski-Harabasz score
        if calinski_scores:
            best_calinski_idx = np.argmax(calinski_scores)
            optimal_clusters['calinski_harabasz'] = list(cluster_range)[best_calinski_idx]
        
        # Best Davies-Bouldin score (lower is better)
        if davies_bouldin_scores:
            best_db_idx = np.argmin(davies_bouldin_scores)
            optimal_clusters['davies_bouldin'] = list(cluster_range)[best_db_idx]
        
        results = {
            'cluster_range': list(cluster_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'calinski_harabasz_scores': calinski_scores,
            'davies_bouldin_scores': davies_bouldin_scores,
            'optimal_clusters': optimal_clusters
        }
        
        logger.info(f"Optimal clusters found: {optimal_clusters}")
        return results
    
    def perform_kmeans_clustering(self, n_clusters: int) -> Dict[str, Any]:
        """Perform K-means clustering with specified number of clusters"""
        logger.info(f"Performing K-means clustering with {n_clusters} clusters...")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(self.embeddings_normalized)
        
        # Calculate metrics
        silhouette_avg = silhouette_score(self.embeddings_normalized, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(self.embeddings_normalized, cluster_labels)
        davies_bouldin = davies_bouldin_score(self.embeddings_normalized, cluster_labels)
        
        # Analyze cluster sizes
        cluster_sizes = Counter(cluster_labels)
        
        results = {
            'algorithm': 'kmeans',
            'n_clusters': n_clusters,
            'labels': cluster_labels,
            'centroids': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'silhouette_score': silhouette_avg,
            'calinski_harabasz_score': calinski_harabasz,
            'davies_bouldin_score': davies_bouldin,
            'cluster_sizes': dict(cluster_sizes),
            'model': kmeans
        }
        
        logger.info(f"K-means completed - Silhouette: {silhouette_avg:.3f}")
        return results
    
    def perform_hierarchical_clustering(self, n_clusters: int, linkage_method: str = 'ward') -> Dict[str, Any]:
        """Perform hierarchical clustering"""
        logger.info(f"Performing hierarchical clustering with {n_clusters} clusters (linkage: {linkage_method})...")
        
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_method,
            metric='euclidean' if linkage_method == 'ward' else 'cosine'
        )
        cluster_labels = clustering.fit_predict(self.embeddings_normalized)
        
        # Calculate metrics
        silhouette_avg = silhouette_score(self.embeddings_normalized, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(self.embeddings_normalized, cluster_labels)
        davies_bouldin = davies_bouldin_score(self.embeddings_normalized, cluster_labels)
        
        # Analyze cluster sizes
        cluster_sizes = Counter(cluster_labels)
        
        results = {
            'algorithm': 'hierarchical',
            'linkage_method': linkage_method,
            'n_clusters': n_clusters,
            'labels': cluster_labels,
            'silhouette_score': silhouette_avg,
            'calinski_harabasz_score': calinski_harabasz,
            'davies_bouldin_score': davies_bouldin,
            'cluster_sizes': dict(cluster_sizes),
            'model': clustering
        }
        
        logger.info(f"Hierarchical clustering completed - Silhouette: {silhouette_avg:.3f}")
        return results
    
    def perform_dbscan_clustering(self, eps: float = 0.3, min_samples: int = 5) -> Dict[str, Any]:
        """Perform DBSCAN clustering"""
        logger.info(f"Performing DBSCAN clustering (eps={eps}, min_samples={min_samples})...")
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        cluster_labels = clustering.fit_predict(self.embeddings_normalized)
        
        # Count clusters (-1 is noise)
        unique_labels = set(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        # Calculate metrics (excluding noise points)
        if n_clusters > 1:
            # Remove noise points for metric calculation
            non_noise_mask = cluster_labels != -1
            if np.sum(non_noise_mask) > 1:
                silhouette_avg = silhouette_score(
                    self.embeddings_normalized[non_noise_mask], 
                    cluster_labels[non_noise_mask]
                )
                calinski_harabasz = calinski_harabasz_score(
                    self.embeddings_normalized[non_noise_mask], 
                    cluster_labels[non_noise_mask]
                )
                davies_bouldin = davies_bouldin_score(
                    self.embeddings_normalized[non_noise_mask], 
                    cluster_labels[non_noise_mask]
                )
            else:
                silhouette_avg = calinski_harabasz = davies_bouldin = 0
        else:
            silhouette_avg = calinski_harabasz = davies_bouldin = 0
        
        # Analyze cluster sizes
        cluster_sizes = Counter(cluster_labels)
        
        results = {
            'algorithm': 'dbscan',
            'eps': eps,
            'min_samples': min_samples,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'labels': cluster_labels,
            'silhouette_score': silhouette_avg,
            'calinski_harabasz_score': calinski_harabasz,
            'davies_bouldin_score': davies_bouldin,
            'cluster_sizes': dict(cluster_sizes),
            'model': clustering
        }
        
        logger.info(f"DBSCAN completed - {n_clusters} clusters, {n_noise} noise points")
        return results
    
    def optimize_dbscan_parameters(self) -> Dict[str, Any]:
        """Find optimal DBSCAN parameters"""
        logger.info("Optimizing DBSCAN parameters...")
        
        eps_range = config.CLUSTERING_ALGORITHMS['dbscan']['eps_range']
        min_samples_range = config.CLUSTERING_ALGORITHMS['dbscan']['min_samples_range']
        
        best_silhouette = -1
        best_params = None
        all_results = []
        
        for eps in tqdm(eps_range, desc="Testing DBSCAN parameters"):
            for min_samples in min_samples_range:
                try:
                    result = self.perform_dbscan_clustering(eps, min_samples)
                    all_results.append(result)
                    
                    # Check if this is the best result so far
                    if (result['n_clusters'] > 1 and 
                        result['silhouette_score'] > best_silhouette and
                        result['n_noise'] < len(self.embeddings) * 0.5):  # Don't allow too much noise
                        
                        best_silhouette = result['silhouette_score']
                        best_params = (eps, min_samples)
                        
                except Exception as e:
                    logger.warning(f"DBSCAN failed with eps={eps}, min_samples={min_samples}: {e}")
        
        return {
            'best_params': best_params,
            'best_silhouette': best_silhouette,
            'all_results': all_results
        }
    
    def analyze_cluster_content(self, cluster_labels: np.ndarray, algorithm_name: str) -> Dict[str, Any]:
        """Analyze the content of each cluster"""
        logger.info(f"Analyzing cluster content for {algorithm_name}...")
        
        cluster_analysis = {}
        unique_clusters = np.unique(cluster_labels)
        
        # Remove noise cluster if present (DBSCAN)
        if -1 in unique_clusters:
            unique_clusters = unique_clusters[unique_clusters != -1]
        
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_posts = self.post_data[cluster_mask]
            
            if len(cluster_posts) < config.MIN_CLUSTER_SIZE:
                continue
            
            # Basic statistics
            analysis = {
                'size': len(cluster_posts),
                'avg_word_count': cluster_posts['word_count'].mean(),
                'avg_content_length': cluster_posts['content_length'].mean(),
                'date_range': {
                    'earliest': cluster_posts['publication_date'].min(),
                    'latest': cluster_posts['publication_date'].max()
                }
            }
            
            # Extract common terms from titles and content
            if self.stop_words:
                # Analyze titles
                titles_text = ' '.join(cluster_posts['title'].fillna('').astype(str))
                title_words = self.extract_keywords(titles_text)
                analysis['common_title_terms'] = title_words[:config.TOP_WORDS_PER_CLUSTER]
                
                # Analyze content (sample to avoid memory issues)
                sample_size = min(50, len(cluster_posts))
                content_sample = cluster_posts['extracted_text'].sample(n=sample_size).fillna('')
                content_text = ' '.join(content_sample.astype(str)[:10000])  # Limit length
                content_words = self.extract_keywords(content_text)
                analysis['common_content_terms'] = content_words[:config.TOP_WORDS_PER_CLUSTER]
            
            # Sample post titles for manual inspection
            sample_titles = cluster_posts['title'].head(10).tolist()
            analysis['sample_titles'] = sample_titles
            
            cluster_analysis[f'cluster_{cluster_id}'] = analysis
        
        return cluster_analysis
    
    def extract_keywords(self, text: str) -> List[Tuple[str, int]]:
        """Extract keywords from text"""
        if not text or not nltk:
            return []
        
        # Clean and tokenize
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = word_tokenize(text)
        
        # Filter words
        keywords = [
            word for word in words 
            if len(word) > 2 
            and word not in self.stop_words
            and word.isalpha()
        ]
        
        # Count and return most common
        word_counts = Counter(keywords)
        return word_counts.most_common()


def load_embeddings_and_data(embeddings_file: str = None, data_dir: str = None) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load embeddings and corresponding post data"""
    data_dir = Path(data_dir or config.OUTPUT_DIR)
    embeddings_file = embeddings_file or str(data_dir / "blog_embeddings.npy")
    
    # Load embeddings
    if not os.path.exists(embeddings_file):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
    
    embeddings = np.load(embeddings_file)
    logger.info(f"Loaded embeddings: {embeddings.shape}")
    
    # Load post data
    posts_file = data_dir / "extracted_posts.csv"
    if not posts_file.exists():
        raise FileNotFoundError(f"Posts data not found: {posts_file}")
    
    post_data = pd.read_csv(posts_file)
    
    # Filter successful extractions to match embeddings
    post_data = post_data[post_data['extraction_success'] == True].copy()
    
    if len(embeddings) != len(post_data):
        logger.warning(f"Embeddings count ({len(embeddings)}) doesn't match posts count ({len(post_data)})")
    
    return embeddings, post_data


def run_comprehensive_analysis(
    embeddings: np.ndarray, 
    post_data: pd.DataFrame,
    output_dir: str = None,
    algorithms: List[str] = None
) -> Dict[str, Any]:
    """Run comprehensive clustering analysis"""
    output_dir = Path(output_dir or config.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    algorithms = algorithms or ['kmeans', 'hierarchical', 'dbscan']
    
    analyzer = ClusteringAnalyzer(embeddings, post_data)
    results = {}
    
    # 1. K-means analysis
    if 'kmeans' in algorithms:
        logger.info("Running K-means analysis...")
        
        # Find optimal clusters
        kmeans_optimization = analyzer.find_optimal_clusters_kmeans()
        results['kmeans_optimization'] = kmeans_optimization
        
        # Run with optimal cluster count (using silhouette score)
        optimal_k = kmeans_optimization['optimal_clusters'].get('silhouette', 15)
        kmeans_result = analyzer.perform_kmeans_clustering(optimal_k)
        
        # Analyze cluster content
        kmeans_content = analyzer.analyze_cluster_content(kmeans_result['labels'], 'kmeans')
        kmeans_result['cluster_analysis'] = kmeans_content
        
        results['kmeans'] = kmeans_result
    
    # 2. Hierarchical clustering
    if 'hierarchical' in algorithms:
        logger.info("Running hierarchical clustering analysis...")
        
        hierarchical_results = {}
        optimal_k = results.get('kmeans_optimization', {}).get('optimal_clusters', {}).get('silhouette', 15)
        
        for linkage_method in config.CLUSTERING_ALGORITHMS['hierarchical']['linkage_methods']:
            hier_result = analyzer.perform_hierarchical_clustering(optimal_k, linkage_method)
            hier_content = analyzer.analyze_cluster_content(hier_result['labels'], f'hierarchical_{linkage_method}')
            hier_result['cluster_analysis'] = hier_content
            hierarchical_results[linkage_method] = hier_result
        
        results['hierarchical'] = hierarchical_results
    
    # 3. DBSCAN analysis
    if 'dbscan' in algorithms:
        logger.info("Running DBSCAN analysis...")
        
        # Optimize parameters
        dbscan_optimization = analyzer.optimize_dbscan_parameters()
        results['dbscan_optimization'] = dbscan_optimization
        
        # Run with best parameters
        if dbscan_optimization['best_params']:
            eps, min_samples = dbscan_optimization['best_params']
            dbscan_result = analyzer.perform_dbscan_clustering(eps, min_samples)
            
            # Analyze cluster content
            dbscan_content = analyzer.analyze_cluster_content(dbscan_result['labels'], 'dbscan')
            dbscan_result['cluster_analysis'] = dbscan_content
            
            results['dbscan'] = dbscan_result
    
    # Save results
    logger.info("Saving clustering results...")
    
    # Create a serializable version of results (remove sklearn objects)
    def make_serializable(obj):
        """Recursively convert numpy types to Python types for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        else:
            return obj
    
    results_serializable = {}
    for algo_name, algo_results in results.items():
        if isinstance(algo_results, dict):
            results_serializable[algo_name] = {}
            for key, value in algo_results.items():
                if key == 'model':
                    continue  # Skip sklearn models
                else:
                    results_serializable[algo_name][key] = make_serializable(value)
    
    # Save results
    results_file = output_dir / "clustering_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=2, default=str)
    
    # Save cluster labels as CSV for each algorithm
    cluster_labels_df = post_data[['post_id', 'title']].copy()
    
    if 'kmeans' in results:
        cluster_labels_df['kmeans_cluster'] = results['kmeans']['labels']
    
    if 'hierarchical' in results:
        for linkage_method, hier_result in results['hierarchical'].items():
            cluster_labels_df[f'hierarchical_{linkage_method}_cluster'] = hier_result['labels']
    
    if 'dbscan' in results:
        cluster_labels_df['dbscan_cluster'] = results['dbscan']['labels']
    
    cluster_labels_df.to_csv(output_dir / "cluster_labels.csv", index=False)
    
    # Save sklearn models separately
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    if 'kmeans' in results:
        joblib.dump(results['kmeans']['model'], models_dir / "kmeans_model.pkl")
    
    if 'hierarchical' in results:
        for linkage_method, hier_result in results['hierarchical'].items():
            joblib.dump(hier_result['model'], models_dir / f"hierarchical_{linkage_method}_model.pkl")
    
    if 'dbscan' in results:
        joblib.dump(results['dbscan']['model'], models_dir / "dbscan_model.pkl")
    
    logger.info(f"Results saved to {output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Perform clustering analysis on blog post embeddings")
    parser.add_argument(
        '--embeddings-file',
        help="Path to embeddings numpy file"
    )
    parser.add_argument(
        '--data-dir',
        default=config.OUTPUT_DIR,
        help="Directory containing data files"
    )
    parser.add_argument(
        '--output-dir',
        default=config.OUTPUT_DIR,
        help="Directory to save clustering results"
    )
    parser.add_argument(
        '--algorithms',
        nargs='+',
        choices=['kmeans', 'hierarchical', 'dbscan'],
        default=['kmeans', 'hierarchical', 'dbscan'],
        help="Clustering algorithms to run"
    )
    
    args = parser.parse_args()
    
    try:
        # Load data
        logger.info("Loading embeddings and post data...")
        embeddings, post_data = load_embeddings_and_data(
            embeddings_file=args.embeddings_file,
            data_dir=args.data_dir
        )
        
        # Run analysis
        logger.info("Starting comprehensive clustering analysis...")
        results = run_comprehensive_analysis(
            embeddings,
            post_data,
            output_dir=args.output_dir,
            algorithms=args.algorithms
        )
        
        # Print summary
        print("\n" + "="*60)
        print("CLUSTERING ANALYSIS SUMMARY")
        print("="*60)
        
        for algo_name, algo_results in results.items():
            if 'optimization' in algo_name:
                continue
                
            print(f"\n{algo_name.upper()}:")
            
            if algo_name == 'hierarchical':
                for linkage_method, hier_result in algo_results.items():
                    print(f"  {linkage_method}: {hier_result['n_clusters']} clusters, "
                          f"silhouette: {hier_result['silhouette_score']:.3f}")
            else:
                if 'n_clusters' in algo_results:
                    print(f"  Clusters: {algo_results['n_clusters']}")
                    print(f"  Silhouette Score: {algo_results['silhouette_score']:.3f}")
                    if 'n_noise' in algo_results:
                        print(f"  Noise Points: {algo_results['n_noise']}")
        
        logger.info("Clustering analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Clustering analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
