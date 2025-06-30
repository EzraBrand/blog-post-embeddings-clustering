#!/usr/bin/env python3
"""
Quick analysis of existing clusters to identify candidates for micro-clustering
"""

import pandas as pd
import numpy as np
from collections import Counter

def main():
    print("🎓 Talmudic Blog Clustering Analysis")
    print("=" * 50)
    
    # Load the cluster labels
    try:
        df = pd.read_csv('processed_data/cluster_labels.csv')
        print(f"✅ Loaded {len(df)} posts with cluster assignments")
    except Exception as e:
        print(f"❌ Error loading cluster data: {e}")
        return
    
    # Analyze K-means clusters
    cluster_counts = df['kmeans_cluster'].value_counts().sort_index()
    print(f"\n📊 K-means Clustering Overview:")
    print(f"   • Total clusters: {len(cluster_counts)}")
    print(f"   • Total posts: {len(df)}")
    print(f"   • Average cluster size: {cluster_counts.mean():.1f}")
    print(f"   • Largest cluster: {cluster_counts.max()} posts")
    print(f"   • Smallest cluster: {cluster_counts.min()} posts")
    
    # Find clusters suitable for subdivision (15+ posts)
    large_clusters = cluster_counts[cluster_counts >= 15].sort_values(ascending=False)
    print(f"\n🎯 Clusters with 15+ posts (good for micro-clustering):")
    print(f"   Found {len(large_clusters)} clusters suitable for subdivision")
    
    for cluster_id, count in large_clusters.items():
        print(f"   • Cluster {cluster_id}: {count} posts")
    
    # Show sample titles from largest clusters
    print(f"\n📚 Sample titles from top 3 largest clusters:")
    for i, (cluster_id, count) in enumerate(large_clusters.head(3).items()):
        print(f"\n--- Cluster {cluster_id} ({count} posts) ---")
        sample_titles = df[df['kmeans_cluster'] == cluster_id]['title'].head(4)
        for j, title in enumerate(sample_titles, 1):
            print(f"   {j}. {title}")
    
    # Analyze content themes by examining titles
    print(f"\n🔍 Thematic Analysis Hints:")
    
    # Look for Talmudic tractate references
    tractate_keywords = ['Yoma', 'Sanhedrin', 'Shabbat', 'Gittin', 'Pesachim', 'Megillah', 'Avodah Zarah']
    for keyword in tractate_keywords:
        matching_posts = df[df['title'].str.contains(keyword, case=False, na=False)]
        if len(matching_posts) > 0:
            clusters = matching_posts['kmeans_cluster'].unique()
            print(f"   • {keyword}: {len(matching_posts)} posts across clusters {clusters}")
    
    print(f"\n✨ Ready for micro-cluster analysis!")
    print(f"   Recommended next step: Focus on clusters {list(large_clusters.head(5).index)}")

if __name__ == "__main__":
    main()
