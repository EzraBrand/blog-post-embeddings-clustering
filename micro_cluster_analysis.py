#!/usr/bin/env python3
"""
Micro-Cluster Analysis for Talmudic Blog Posts
Creates actionable thematic collections for scholarly research
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import re

def load_data():
    """Load all necessary data files"""
    print("📊 Loading data files...")
    
    # Load cluster assignments
    cluster_df = pd.read_csv('processed_data/cluster_labels.csv')
    
    # Load full post content
    posts_df = pd.read_csv('processed_data/extracted_posts.csv')
    
    # Load embeddings
    embeddings = np.load('processed_data/blog_embeddings.npy')
    
    # Merge cluster info with posts
    merged_df = pd.merge(cluster_df, posts_df[['post_id', 'extracted_text', 'word_count']], 
                        on='post_id', how='left')
    
    print(f"✅ Loaded {len(merged_df)} posts with embeddings")
    return merged_df, embeddings

def analyze_cluster_content(df, cluster_id, max_posts=10):
    """Analyze content of a specific cluster to identify themes"""
    cluster_posts = df[df['kmeans_cluster'] == cluster_id].copy()
    
    # Analyze titles for common themes
    titles = cluster_posts['title'].tolist()
    
    # Extract Talmudic tractate references
    tractates = []
    for title in titles:
        # Look for Talmudic citations like "(Yoma 9b)" or "Sanhedrin 91a"
        matches = re.findall(r'\(([A-Za-z]+)\s+\d+[ab]?(?:-\d+[ab]?)?\)', title)
        tractates.extend(matches)
    
    # Extract themes from titles
    themes = {
        'temple_service': any(word in ' '.join(titles).lower() for word in ['temple', 'incense', 'priest', 'sacrifice', 'yom kippur']),
        'aggadah_stories': any(word in ' '.join(titles).lower() for word in ['story', 'stories', 'anecdote', 'tale']),
        'halakha_legal': any(word in ' '.join(titles).lower() for word in ['law', 'legal', 'ritual', 'boundaries', 'modes']),
        'historical_figures': any(word in ' '.join(titles).lower() for word in ['rabbi', 'rabban', 'reish lakish', 'alexander', 'haman']),
        'biblical_commentary': any(word in ' '.join(titles).lower() for word in ['biblical', 'bible', 'ezra', 'esther', 'daniel']),
        'ethics_wisdom': any(word in ' '.join(titles).lower() for word in ['advice', 'guidance', 'wisdom', 'righteous']),
        'digital_humanities': any(word in ' '.join(titles).lower() for word in ['computational', 'digital', 'chatgpt', 'ai', 'algorithm'])
    }
    
    return {
        'posts': cluster_posts,
        'tractates': list(set(tractates)),
        'themes': {k: v for k, v in themes.items() if v},
        'size': len(cluster_posts)
    }

def find_similar_posts_within_cluster(df, embeddings, cluster_id, min_similarity=0.7):
    """Find highly similar posts within a cluster using cosine similarity"""
    cluster_posts = df[df['kmeans_cluster'] == cluster_id].copy()
    cluster_indices = cluster_posts.index.tolist()
    cluster_embeddings = embeddings[cluster_indices]
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(cluster_embeddings)
    
    # Find groups of highly similar posts
    micro_clusters = []
    used_indices = set()
    
    for i, post_idx in enumerate(cluster_indices):
        if i in used_indices:
            continue
            
        # Find posts similar to this one
        similarities = similarity_matrix[i]
        similar_indices = []
        
        for j, sim_score in enumerate(similarities):
            if j not in used_indices and sim_score >= min_similarity:
                similar_indices.append(j)
        
        if len(similar_indices) >= 3:  # At least 3 posts for a micro-cluster
            micro_cluster_posts = cluster_posts.iloc[similar_indices]
            micro_clusters.append({
                'posts': micro_cluster_posts,
                'similarity_scores': [similarities[j] for j in similar_indices],
                'avg_similarity': np.mean([similarities[j] for j in similar_indices])
            })
            used_indices.update(similar_indices)
    
    return micro_clusters

def create_micro_clusters(df, embeddings):
    """Create actionable micro-clusters from the largest clusters"""
    print("\n🎯 Creating Micro-Clusters for Scholarly Research")
    print("=" * 60)
    
    # Focus on largest clusters (15+ posts)
    large_clusters = df['kmeans_cluster'].value_counts()
    large_clusters = large_clusters[large_clusters >= 15].index[:8]  # Top 8 clusters
    
    all_micro_clusters = []
    
    for cluster_id in large_clusters:
        print(f"\n📚 Analyzing Cluster {cluster_id}...")
        
        # Analyze content themes
        analysis = analyze_cluster_content(df, cluster_id)
        
        # Find micro-clusters within this cluster
        micro_clusters = find_similar_posts_within_cluster(df, embeddings, cluster_id, min_similarity=0.6)
        
        if micro_clusters:
            print(f"   Found {len(micro_clusters)} potential micro-clusters")
            
            for i, micro_cluster in enumerate(micro_clusters):
                micro_cluster['parent_cluster'] = cluster_id
                micro_cluster['analysis'] = analysis
                micro_cluster['id'] = f"{cluster_id}.{i+1}"
                all_micro_clusters.append(micro_cluster)
    
    # Sort by quality (average similarity)
    all_micro_clusters.sort(key=lambda x: x['avg_similarity'], reverse=True)
    
    return all_micro_clusters[:15]  # Return top 15 micro-clusters

def generate_scholarly_descriptions(micro_clusters):
    """Generate scholarly descriptions for each micro-cluster"""
    print("\n🎓 Generating Scholarly Descriptions")
    print("=" * 50)
    
    scholarly_clusters = []
    
    for i, mc in enumerate(micro_clusters, 1):
        posts = mc['posts']
        analysis = mc['analysis']
        
        # Determine primary theme
        themes = analysis['themes']
        tractates = analysis['tractates']
        
        # Generate theme title
        if 'temple_service' in themes and tractates:
            theme_title = f"בית המקדש ומשמרת הכהנים - Temple Service and Priestly Duties ({', '.join(tractates[:2])})"
            significance = "Explores the intricate rituals and responsibilities of Temple service"
        elif 'aggadah_stories' in themes:
            theme_title = f"אגדות חז\"ל - Talmudic Narratives and Exemplary Tales"
            significance = "Examines narrative traditions that convey moral and theological teachings"
        elif 'halakha_legal' in themes and tractates:
            theme_title = f"הלכה ומשפט - Legal Principles and Ritual Boundaries ({', '.join(tractates[:2])})"
            significance = "Analyzes legal frameworks and practical halakhic applications"
        elif 'historical_figures' in themes:
            theme_title = f"דמויות בתלמוד - Talmudic Personalities and Historical Encounters"
            significance = "Studies key rabbinic figures and their intellectual contributions"
        elif 'biblical_commentary' in themes:
            theme_title = f"פרשנות התנ\"ך - Talmudic Biblical Hermeneutics"
            significance = "Explores rabbinic interpretation of biblical texts and themes"
        elif 'digital_humanities' in themes:
            theme_title = f"תלמוד ממוחשב - Computational Talmudic Studies"
            significance = "Examines applications of digital methods to traditional texts"
        else:
            # Generic based on tractates
            if tractates:
                theme_title = f"מסכת {tractates[0]} - Studies in Tractate {tractates[0]}"
                significance = f"Focused analysis of themes and passages from Tractate {tractates[0]}"
            else:
                theme_title = f"לימוד תלמוד - Talmudic Studies Collection {i}"
                significance = "Thematically related Talmudic discussions"
        
        # Generate reading sequence
        reading_sequence = []
        for j, (_, post) in enumerate(posts.iterrows(), 1):
            word_count = post.get('word_count', 0)
            if word_count and word_count > 0:
                reading_time = f"~{word_count//200} min read" if word_count < 2000 else f"~{word_count//200} min read (detailed)"
            else:
                reading_time = "~5 min read"
            
            reading_sequence.append({
                'order': j,
                'title': post['title'],
                'post_id': post['post_id'],
                'reading_time': reading_time
            })
        
        scholarly_clusters.append({
            'micro_cluster_id': f"MC-{i:02d}",
            'theme_title': theme_title,
            'significance': significance,
            'post_count': len(posts),
            'avg_similarity': mc['avg_similarity'],
            'primary_tractates': tractates[:3],
            'themes': list(themes.keys()),
            'reading_sequence': reading_sequence,
            'research_applications': [
                "Comparative analysis of rabbinic approaches",
                "Thematic study for academic research",
                "Sequential reading for deep comprehension",
                "Cross-referencing with classical commentaries"
            ]
        })
    
    return scholarly_clusters

def export_research_tools(scholarly_clusters):
    """Export practical research tools"""
    print("\n📋 Generating Research Tools")
    print("=" * 40)
    
    # Create markdown report
    with open('MICRO_CLUSTER_ANALYSIS.md', 'w', encoding='utf-8') as f:
        f.write("# Talmudic Blog Micro-Cluster Analysis\n")
        f.write("## Actionable Research Collections\n\n")
        f.write("*Generated from computational analysis of 539 Talmudic blog posts*\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write(f"This analysis identified **{len(scholarly_clusters)} highly coherent micro-clusters** ")
        f.write("from the original 45 K-means clusters, creating actionable research collections ")
        f.write("for Talmudic scholarship.\n\n")
        
        f.write("### Methodology\n")
        f.write("- **Semantic similarity analysis** using OpenAI embeddings (3072-dim)\n")
        f.write("- **Content thematic analysis** of titles and Talmudic references\n")
        f.write("- **Scholarly curation** respecting traditional learning methodologies\n\n")
        
        for i, cluster in enumerate(scholarly_clusters, 1):
            f.write(f"## {cluster['micro_cluster_id']}: {cluster['theme_title']}\n\n")
            f.write(f"**Research Significance:** {cluster['significance']}\n\n")
            f.write(f"**Collection Size:** {cluster['post_count']} posts | ")
            f.write(f"**Semantic Coherence:** {cluster['avg_similarity']:.3f}\n\n")
            
            if cluster['primary_tractates']:
                f.write(f"**Primary Tractates:** {', '.join(cluster['primary_tractates'])}\n\n")
            
            f.write("### Recommended Reading Sequence\n\n")
            for item in cluster['reading_sequence']:
                f.write(f"{item['order']}. **{item['title']}** ({item['reading_time']})\n")
                f.write(f"   - Post ID: `{item['post_id']}`\n\n")
            
            f.write("### Research Applications\n\n")
            for app in cluster['research_applications']:
                f.write(f"- {app}\n")
            f.write("\n")
        
        f.write("## Cross-Reference Index\n\n")
        f.write("### By Tractate\n\n")
        tractate_index = defaultdict(list)
        for cluster in scholarly_clusters:
            for tractate in cluster['primary_tractates']:
                tractate_index[tractate].append(cluster['micro_cluster_id'])
        
        for tractate, cluster_ids in sorted(tractate_index.items()):
            f.write(f"**{tractate}:** {', '.join(cluster_ids)}\n\n")
    
    # Create CSV for data analysis
    csv_data = []
    for cluster in scholarly_clusters:
        for item in cluster['reading_sequence']:
            csv_data.append({
                'micro_cluster_id': cluster['micro_cluster_id'],
                'theme_title': cluster['theme_title'],
                'post_id': item['post_id'],
                'title': item['title'],
                'reading_order': item['order'],
                'semantic_coherence': cluster['avg_similarity'],
                'primary_tractates': '; '.join(cluster['primary_tractates']),
                'research_significance': cluster['significance']
            })
    
    csv_df = pd.DataFrame(csv_data)
    csv_df.to_csv('micro_clusters_research_tool.csv', index=False)
    
    print(f"✅ Generated MICRO_CLUSTER_ANALYSIS.md")
    print(f"✅ Generated micro_clusters_research_tool.csv")

def main():
    print("🎓 Talmudic Blog Micro-Clustering Analysis")
    print("Digital Humanities for Traditional Jewish Learning")
    print("=" * 60)
    
    # Load data
    df, embeddings = load_data()
    
    # Create micro-clusters
    micro_clusters = create_micro_clusters(df, embeddings)
    
    # Generate scholarly descriptions
    scholarly_clusters = generate_scholarly_descriptions(micro_clusters)
    
    # Export research tools
    export_research_tools(scholarly_clusters)
    
    print(f"\n🎯 Analysis Complete!")
    print(f"   • Created {len(scholarly_clusters)} actionable micro-clusters")
    print(f"   • Average coherence: {np.mean([c['avg_similarity'] for c in scholarly_clusters]):.3f}")
    print(f"   • Total posts organized: {sum(c['post_count'] for c in scholarly_clusters)}")
    print(f"\n📚 Ready for scholarly research and study!")

if __name__ == "__main__":
    main()
