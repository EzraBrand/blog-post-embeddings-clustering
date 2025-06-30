#!/usr/bin/env python3
"""
Refined Micro-Cluster Analysis - Focus on 5-post collections
Creates highly focused, actionable research collections
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
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

def create_tight_micro_clusters(df, embeddings, target_size=5, max_size=7):
    """Create tight micro-clusters of exactly 5 posts (or close to it)"""
    print(f"\n🎯 Creating Tight Micro-Clusters (target size: {target_size} posts)")
    print("=" * 60)
    
    # Focus on largest clusters for subdivision
    large_clusters = df['kmeans_cluster'].value_counts()
    large_clusters = large_clusters[large_clusters >= 10].index  # Only clusters with 10+ posts
    
    all_micro_clusters = []
    
    for cluster_id in large_clusters:
        print(f"\n📚 Subdividing Cluster {cluster_id}...")
        
        cluster_posts = df[df['kmeans_cluster'] == cluster_id].copy()
        cluster_indices = cluster_posts.index.tolist()
        cluster_embeddings = embeddings[cluster_indices]
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(cluster_embeddings)
        
        # Use K-means to subdivide this cluster into smaller groups
        n_posts = len(cluster_posts)
        n_subclusters = max(2, n_posts // target_size)  # Aim for target_size posts per subcluster
        
        if n_posts >= 10:  # Only subdivide if we have enough posts
            kmeans = KMeans(n_clusters=n_subclusters, random_state=42, n_init=10)
            subcluster_labels = kmeans.fit_predict(cluster_embeddings)
            
            # Create micro-clusters from subclusters
            for sub_id in range(n_subclusters):
                sub_indices = [i for i, label in enumerate(subcluster_labels) if label == sub_id]
                
                if len(sub_indices) >= 3 and len(sub_indices) <= max_size:  # Good size range
                    sub_posts = cluster_posts.iloc[sub_indices]
                    
                    # Calculate average similarity within this micro-cluster
                    sub_similarities = []
                    for i in range(len(sub_indices)):
                        for j in range(i+1, len(sub_indices)):
                            sim = similarity_matrix[sub_indices[i]][sub_indices[j]]
                            sub_similarities.append(sim)
                    
                    avg_similarity = np.mean(sub_similarities) if sub_similarities else 0
                    
                    # Only include if similarity is high enough
                    if avg_similarity >= 0.65:  # Lower threshold for smaller clusters
                        all_micro_clusters.append({
                            'posts': sub_posts,
                            'avg_similarity': avg_similarity,
                            'parent_cluster': cluster_id,
                            'subcluster_id': sub_id,
                            'id': f"{cluster_id}.{sub_id}",
                            'size': len(sub_posts)
                        })
                        print(f"   ✅ Created micro-cluster {cluster_id}.{sub_id}: {len(sub_posts)} posts (sim: {avg_similarity:.3f})")
    
    # Also look for naturally tight groups in medium-sized clusters (8-15 posts)
    medium_clusters = df['kmeans_cluster'].value_counts()
    medium_clusters = medium_clusters[(medium_clusters >= 8) & (medium_clusters < 10)].index
    
    for cluster_id in medium_clusters:
        cluster_posts = df[df['kmeans_cluster'] == cluster_id].copy()
        if len(cluster_posts) <= max_size:  # Use the whole cluster if it's already the right size
            cluster_indices = cluster_posts.index.tolist()
            cluster_embeddings = embeddings[cluster_indices]
            
            # Calculate average similarity
            similarity_matrix = cosine_similarity(cluster_embeddings)
            similarities = []
            for i in range(len(cluster_embeddings)):
                for j in range(i+1, len(cluster_embeddings)):
                    similarities.append(similarity_matrix[i][j])
            
            avg_similarity = np.mean(similarities) if similarities else 0
            
            if avg_similarity >= 0.65:
                all_micro_clusters.append({
                    'posts': cluster_posts,
                    'avg_similarity': avg_similarity,
                    'parent_cluster': cluster_id,
                    'subcluster_id': 0,
                    'id': f"{cluster_id}.0",
                    'size': len(cluster_posts)
                })
                print(f"   ✅ Used whole cluster {cluster_id}: {len(cluster_posts)} posts (sim: {avg_similarity:.3f})")
    
    # Sort by quality (similarity) and size preference
    all_micro_clusters.sort(key=lambda x: (x['avg_similarity'], -(abs(x['size'] - target_size))), reverse=True)
    
    # Select best micro-clusters, ensuring variety
    selected_clusters = []
    used_parent_clusters = set()
    
    for mc in all_micro_clusters:
        # Limit to 2 micro-clusters per parent to ensure variety
        parent_count = sum(1 for sc in selected_clusters if sc['parent_cluster'] == mc['parent_cluster'])
        if parent_count < 2:
            selected_clusters.append(mc)
        
        if len(selected_clusters) >= 15:  # Target: 15 micro-clusters
            break
    
    print(f"\n📊 Created {len(selected_clusters)} tight micro-clusters")
    print(f"   Average size: {np.mean([mc['size'] for mc in selected_clusters]):.1f} posts")
    print(f"   Average similarity: {np.mean([mc['avg_similarity'] for mc in selected_clusters]):.3f}")
    
    return selected_clusters

def analyze_cluster_themes(posts):
    """Analyze content themes for a cluster"""
    titles = posts['title'].tolist()
    title_text = ' '.join(titles).lower()
    
    # Extract Talmudic tractate references
    tractates = []
    for title in titles:
        matches = re.findall(r'\(([A-Za-z]+)\s+\d+[ab]?(?:-\d+[ab]?)?\)', title)
        tractates.extend(matches)
    
    # Extract themes
    themes = {
        'temple_service': any(word in title_text for word in ['temple', 'incense', 'priest', 'sacrifice', 'yom kippur', 'service']),
        'aggadah_stories': any(word in title_text for word in ['story', 'stories', 'anecdote', 'tale', 'narrative']),
        'halakha_legal': any(word in title_text for word in ['law', 'legal', 'ritual', 'boundaries', 'modes', 'ruling']),
        'historical_figures': any(word in title_text for word in ['rabbi', 'rabban', 'reish lakish', 'alexander', 'antoninus']),
        'biblical_commentary': any(word in title_text for word in ['biblical', 'bible', 'ezra', 'esther', 'daniel', 'scripture']),
        'ethics_wisdom': any(word in title_text for word in ['advice', 'guidance', 'wisdom', 'righteous', 'moral']),
        'digital_humanities': any(word in title_text for word in ['computational', 'digital', 'chatgpt', 'ai', 'algorithm', 'software']),
        'liturgy_prayer': any(word in title_text for word in ['prayer', 'liturgy', 'blessing', 'tefillin', 'tzitzit']),
        'talmudic_methodology': any(word in title_text for word in ['talmudic', 'methodology', 'hermeneutics', 'interpretation']),
        'social_dynamics': any(word in title_text for word in ['community', 'social', 'hierarchy', 'relationship', 'marriage'])
    }
    
    return {
        'tractates': list(set(tractates)),
        'themes': {k: v for k, v in themes.items() if v}
    }

def generate_focused_descriptions(micro_clusters):
    """Generate scholarly descriptions for tight micro-clusters"""
    print("\n🎓 Generating Focused Scholarly Descriptions")
    print("=" * 50)
    
    scholarly_clusters = []
    
    for i, mc in enumerate(micro_clusters, 1):
        posts = mc['posts']
        analysis = analyze_cluster_themes(posts)
        
        themes = analysis['themes']
        tractates = analysis['tractates']
        
        # Generate precise theme titles for smaller collections
        if 'digital_humanities' in themes:
            theme_title = f"תלמוד דיגיטלי - Digital Talmudic Studies and Computational Methods"
            significance = "Explores applications of technology and digital methods to traditional Jewish texts"
        elif 'temple_service' in themes and tractates:
            primary_tractate = tractates[0] if tractates else "Various"
            theme_title = f"עבודת בית המקדש - Temple Service and Ritual ({primary_tractate})"
            significance = f"Focused study of Temple-era practices and priestly responsibilities in {primary_tractate}"
        elif 'aggadah_stories' in themes:
            theme_title = f"סיפורי אגדה - Talmudic Narratives and Moral Tales"
            significance = "Concentrated collection of aggadic narratives that convey ethical teachings"
        elif 'halakha_legal' in themes and tractates:
            primary_tractate = tractates[0] if tractates else "Various"
            theme_title = f"הלכה ופסיקה - Legal Principles and Practical Rulings ({primary_tractate})"
            significance = f"Focused analysis of halakhic principles and legal reasoning in {primary_tractate}"
        elif 'historical_figures' in themes:
            theme_title = f"דמויות חכמים - Talmudic Sages and Historical Personalities"
            significance = "Intimate study of key rabbinic figures and their contributions"
        elif 'biblical_commentary' in themes:
            theme_title = f"דרוש ופירוש - Talmudic Biblical Interpretation"
            significance = "Concentrated exploration of rabbinic hermeneutical approaches to Scripture"
        elif 'social_dynamics' in themes:
            theme_title = f"חברה ויחסים - Social Dynamics and Community Relations"
            significance = "Focused examination of social structures and interpersonal dynamics in the Talmud"
        elif 'liturgy_prayer' in themes:
            theme_title = f"תפילה ומנהג - Prayer, Liturgy, and Ritual Practice"
            significance = "Concentrated study of liturgical practices and ritual observance"
        else:
            primary_tractate = tractates[0] if tractates else "Various"
            theme_title = f"לימוד מקוטב - Focused Study Collection ({primary_tractate})"
            significance = f"Concentrated thematic exploration of related topics in {primary_tractate}"
        
        # Generate reading sequence optimized for 5-post flow
        reading_sequence = []
        posts_sorted = posts.sort_values('word_count', ascending=True)  # Start with shorter posts
        
        for j, (_, post) in enumerate(posts_sorted.iterrows(), 1):
            word_count = post.get('word_count', 0)
            if word_count and word_count > 0:
                if word_count < 1000:
                    reading_time = f"~{max(3, word_count//200)} min"
                elif word_count < 2500:
                    reading_time = f"~{word_count//200} min"
                else:
                    reading_time = f"~{word_count//200} min (detailed)"
            else:
                reading_time = "~5 min"
            
            reading_sequence.append({
                'order': j,
                'title': post['title'],
                'post_id': post['post_id'],
                'reading_time': reading_time
            })
        
        scholarly_clusters.append({
            'micro_cluster_id': f"FC-{i:02d}",  # FC = Focused Cluster
            'theme_title': theme_title,
            'significance': significance,
            'post_count': len(posts),
            'avg_similarity': mc['avg_similarity'],
            'primary_tractates': tractates[:2],
            'themes': list(themes.keys()),
            'reading_sequence': reading_sequence
        })
    
    return scholarly_clusters

def export_focused_research_tools(scholarly_clusters):
    """Export refined research tools for 5-post collections"""
    print("\n📋 Generating Focused Research Tools")
    print("=" * 40)
    
    # Create refined markdown report
    with open('FOCUSED_MICRO_CLUSTERS.md', 'w', encoding='utf-8') as f:
        f.write("# Focused Talmudic Micro-Clusters\n")
        f.write("## Intimate Research Collections (5-Post Focus)\n\n")
        f.write("*Refined analysis creating highly focused, actionable study collections*\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write(f"This refined analysis identified **{len(scholarly_clusters)} highly focused micro-clusters** ")
        f.write("averaging 5 posts each, creating intimate research collections that are ideal for ")
        f.write("deep scholarly engagement and classroom use.\n\n")
        
        f.write("### Refined Methodology\n")
        f.write("- **Tight similarity thresholds** (0.65+ cosine similarity)\n")
        f.write("- **Optimal collection size** (3-7 posts, targeting 5)\n") 
        f.write("- **Sub-clustering** of large clusters using K-means\n")
        f.write("- **Scholarly curation** for maximum research utility\n\n")
        
        f.write("### Collection Statistics\n")
        avg_size = np.mean([c['post_count'] for c in scholarly_clusters])
        avg_sim = np.mean([c['avg_similarity'] for c in scholarly_clusters])
        f.write(f"- **Average collection size:** {avg_size:.1f} posts\n")
        f.write(f"- **Average semantic coherence:** {avg_sim:.3f}\n")
        f.write(f"- **Total posts organized:** {sum(c['post_count'] for c in scholarly_clusters)}\n")
        f.write(f"- **Optimal for:** Individual study sessions, classroom discussions, focused research\n\n")
        
        for i, cluster in enumerate(scholarly_clusters, 1):
            f.write(f"## {cluster['micro_cluster_id']}: {cluster['theme_title']}\n\n")
            f.write(f"**Research Significance:** {cluster['significance']}\n\n")
            f.write(f"**Collection Size:** {cluster['post_count']} posts | ")
            f.write(f"**Semantic Coherence:** {cluster['avg_similarity']:.3f}\n\n")
            
            if cluster['primary_tractates']:
                f.write(f"**Primary Tractates:** {', '.join(cluster['primary_tractates'])}\n\n")
            
            f.write("### Focused Reading Sequence\n\n")
            total_time = 0
            for item in cluster['reading_sequence']:
                f.write(f"{item['order']}. **{item['title']}** ({item['reading_time']})\n")
                f.write(f"   - Post ID: `{item['post_id']}`\n\n")
                # Extract minutes for total calculation
                time_str = item['reading_time']
                if '~' in time_str and 'min' in time_str:
                    try:
                        mins = int(time_str.split('~')[1].split(' ')[0])
                        total_time += mins
                    except:
                        total_time += 5
            
            f.write(f"**Total estimated reading time:** ~{total_time} minutes\n\n")
            
            f.write("---\n\n")
        
        f.write("## Quick Reference Guide\n\n")
        f.write("### By Study Time\n\n")
        time_sorted = sorted(scholarly_clusters, key=lambda x: x['post_count'])
        f.write("**Quick Study (3-4 posts):**\n")
        for cluster in time_sorted[:5]:
            if cluster['post_count'] <= 4:
                f.write(f"- {cluster['micro_cluster_id']}: {cluster['theme_title']}\n")
        f.write("\n**Standard Study (5-6 posts):**\n")
        for cluster in time_sorted:
            if 5 <= cluster['post_count'] <= 6:
                f.write(f"- {cluster['micro_cluster_id']}: {cluster['theme_title']}\n")
        f.write("\n")
        
        f.write("### By Tractate\n\n")
        tractate_index = defaultdict(list)
        for cluster in scholarly_clusters:
            for tractate in cluster['primary_tractates']:
                tractate_index[tractate].append(cluster['micro_cluster_id'])
        
        for tractate, cluster_ids in sorted(tractate_index.items()):
            f.write(f"**{tractate}:** {', '.join(cluster_ids)}\n")
        f.write("\n")
    
    # Create focused CSV
    csv_data = []
    for cluster in scholarly_clusters:
        for item in cluster['reading_sequence']:
            csv_data.append({
                'micro_cluster_id': cluster['micro_cluster_id'],
                'theme_title': cluster['theme_title'],
                'post_id': item['post_id'],
                'title': item['title'],
                'reading_order': item['order'],
                'collection_size': cluster['post_count'],
                'semantic_coherence': cluster['avg_similarity'],
                'primary_tractates': '; '.join(cluster['primary_tractates']),
                'reading_time': item['reading_time'],
                'research_significance': cluster['significance']
            })
    
    focused_df = pd.DataFrame(csv_data)
    focused_df.to_csv('focused_micro_clusters.csv', index=False)
    
    print(f"✅ Generated FOCUSED_MICRO_CLUSTERS.md")
    print(f"✅ Generated focused_micro_clusters.csv")

def main():
    print("🎓 Focused Talmudic Micro-Clustering Analysis")
    print("Creating Intimate 5-Post Research Collections")
    print("=" * 60)
    
    # Load data
    df, embeddings = load_data()
    
    # Create tight micro-clusters
    micro_clusters = create_tight_micro_clusters(df, embeddings, target_size=5, max_size=7)
    
    # Generate focused descriptions
    scholarly_clusters = generate_focused_descriptions(micro_clusters)
    
    # Export focused research tools
    export_focused_research_tools(scholarly_clusters)
    
    print(f"\n🎯 Focused Analysis Complete!")
    print(f"   • Created {len(scholarly_clusters)} intimate micro-clusters")
    print(f"   • Average collection size: {np.mean([c['post_count'] for c in scholarly_clusters]):.1f} posts")
    print(f"   • Average coherence: {np.mean([c['avg_similarity'] for c in scholarly_clusters]):.3f}")
    print(f"   • Perfect for focused study and classroom use!")

if __name__ == "__main__":
    main()
