#!/usr/bin/env python3
"""
Aggadah-Focused Semantic Index Generator
Creates a multi-layered thematic organization for Talmudic blog posts
with emphasis on aggadic themes and narratives
"""

import pandas as pd
import numpy as np
import re
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import json

def load_blog_data():
    """Load and prepare blog post data"""
    print("📚 Loading blog post data for aggadic analysis...")
    
    # Load the data
    posts_df = pd.read_csv('processed_data/extracted_posts.csv')
    cluster_df = pd.read_csv('processed_data/cluster_labels.csv')
    
    # Merge datasets
    df = pd.merge(cluster_df, posts_df[['post_id', 'extracted_text', 'word_count']], 
                  on='post_id', how='left')
    
    # Exclude meta/index/intro posts that skew the analysis
    meta_posts_to_exclude = [
        "Cataloging My Blogposts: An Organized Breakdown by Category",
        "Introduction to the Talmud", 
        "Guide to Online Resources for Scholarly Jewish Study and Research - 2023",
        "Beyond the Mystique: Correcting Common Misconceptions About the Talmud, and Pathways to Accessibility"
    ]
    
    initial_count = len(df)
    df = df[~df['title'].isin(meta_posts_to_exclude)]
    excluded_count = initial_count - len(df)
    
    print(f"✅ Loaded {len(df)} posts for analysis ({excluded_count} meta posts excluded)")
    return df

def identify_aggadic_themes(df):
    """Identify specific aggadic themes and narratives in the corpus"""
    print("\n🎭 Analyzing aggadic themes and narratives...")
    
    # Extract all titles and content for analysis
    all_text = ' '.join(df['title'].fillna('') + ' ' + df['extracted_text'].fillna(''))
    titles = df['title'].fillna('').tolist()
    
    # Define aggadic theme patterns with Hebrew awareness
    aggadic_patterns = {
        # Narrative Types
        'miracle_stories': [
            'miracle', 'miraculous', 'supernatural', 'divine intervention', 
            'wondrous', 'extraordinary', 'heavenly', 'angel', 'angelic'
        ],
        'sage_narratives': [
            'rabbi', 'rabban', 'rav ', 'reish lakish', 'yochanan', 'akiva', 
            'hillel', 'shammai', 'gamliel', 'sage', 'scholar', 'teacher'
        ],
        'biblical_expansions': [
            'biblical', 'bible', 'genesis', 'exodus', 'moses', 'abraham', 
            'david', 'solomon', 'esther', 'daniel', 'scripture', 'torah'
        ],
        'eschatological_themes': [
            'messiah', 'messianic', 'world to come', 'resurrection', 'redemption',
            'end times', 'future', 'olam haba', 'afterlife', 'judgment'
        ],
        'divine_anthropomorphism': [
            'god', 'divine', 'anthropomorphic', 'hashem', 'holy one', 
            'blessed be he', 'heavenly', 'throne', 'presence'
        ],
        'temple_narratives': [
            'temple', 'priest', 'sacrifice', 'altar', 'incense', 'holy of holies',
            'temple service', 'kohen', 'levite', 'sanctuary'
        ],
        'martyrdom_stories': [
            'martyr', 'persecution', 'death', 'killed', 'execution', 'torture',
            'suffering', 'sacrifice', 'heroic death', 'ten martyrs'
        ],
        'wisdom_tales': [
            'wisdom', 'clever', 'wit', 'riddle', 'parable', 'lesson', 
            'teaching', 'moral', 'ethics', 'values'
        ],
        'historical_encounters': [
            'roman', 'caesar', 'emperor', 'antoninus', 'alexander', 'persian',
            'babylonian', 'historical', 'encounter', 'dialogue'
        ],
        'mystical_themes': [
            'mystical', 'kabbalah', 'merkavah', 'heavenly halls', 'throne',
            'vision', 'revelation', 'secret', 'hidden', 'esoteric'
        ],
        'dream_visions': [
            'dream', 'vision', 'appeared', 'revealed', 'prophecy', 'prophetic',
            'saw in dream', 'nocturnal', 'sleep'
        ],
        'prayer_liturgy': [
            'prayer', 'liturgy', 'blessing', 'psalms', 'hallel', 'shema',
            'amidah', 'kaddish', 'worship', 'devotion'
        ],
        'ethical_teachings': [
            'ethics', 'moral', 'righteousness', 'sin', 'virtue', 'conduct',
            'behavior', 'character', 'piety', 'good deeds'
        ],
        'cosmic_themes': [
            'creation', 'cosmology', 'universe', 'heaven', 'earth', 'cosmic',
            'world', 'bereshit', 'genesis', 'foundations'
        ],
        'death_afterlife': [
            'death', 'afterlife', 'grave', 'burial', 'mourning', 'soul',
            'spirit', 'departed', 'cemetery', 'funeral'
        ],
        'demon_stories': [
            'demon', 'devil', 'satan', 'evil spirit', 'shed', 'mazik',
            'witchcraft', 'magic', 'sorcery', 'supernatural beings'
        ],
        'righteous_women': [
            'righteous women', 'matriarch', 'prophetess', 'female sage',
            'wise woman', 'pious woman', 'virtuous', 'feminine wisdom'
        ],
        'conversion_stories': [
            'convert', 'conversion', 'gentile', 'proselyte', 'ger', 
            'joining', 'acceptance', 'entry into judaism'
        ],
        'community_tales': [
            'community', 'congregation', 'fellowship', 'charity', 'tzedakah',
            'social', 'mutual aid', 'collective', 'public'
        ],
        'legal_narratives': [
            'court case', 'judgment', 'legal decision', 'ruling', 'dispute',
            'litigation', 'testimony', 'witness', 'verdict'
        ],
        'travel_geography': [
            'journey', 'travel', 'babylon', 'israel', 'jerusalem', 'diaspora',
            'exile', 'return', 'pilgrimage', 'geographical'
        ],
        'food_hospitality': [
            'meal', 'feast', 'hospitality', 'eating', 'bread', 'wine',
            'dining', 'guest', 'host', 'food customs'
        ],
        'healing_medicine': [
            'healing', 'medicine', 'cure', 'remedy', 'illness', 'disease',
            'physician', 'medical', 'health', 'recovery'
        ],
        'business_ethics': [
            'business', 'trade', 'merchant', 'honest dealing', 'commerce',
            'market', 'transaction', 'economic', 'livelihood'
        ],
        'family_relationships': [
            'family', 'father', 'mother', 'son', 'daughter', 'marriage',
            'husband', 'wife', 'children', 'domestic', 'household'
        ],
        'repentance_teshuvah': [
            'repentance', 'teshuvah', 'return', 'forgiveness', 'atonement',
            'confession', 'regret', 'change', 'spiritual growth'
        ],
        'study_scholarship': [
            'study', 'learning', 'scholarship', 'academy', 'student', 'teacher',
            'intellectual', 'knowledge', 'wisdom', 'education'
        ],
        'seasonal_calendar': [
            'passover', 'sukkot', 'yom kippur', 'rosh hashanah', 'purim',
            'chanukah', 'sabbath', 'holiday', 'festival', 'season'
        ],
        'divine_justice': [
            'justice', 'punishment', 'reward', 'divine retribution', 'karma',
            'consequences', 'measure for measure', 'celestial court'
        ],
        'prophetic_themes': [
            'prophet', 'prophecy', 'prophetic', 'revelation', 'message',
            'divine communication', 'vision', 'inspiration', 'calling'
        ]
    }
    
    # Analyze posts for each theme
    theme_analysis = {}
    
    for theme, keywords in aggadic_patterns.items():
        matching_posts = []
        theme_scores = []
        
        for idx, row in df.iterrows():
            title = row['title'].lower() if pd.notna(row['title']) else ''
            content = row['extracted_text'].lower() if pd.notna(row['extracted_text']) else ''
            full_text = title + ' ' + content
            
            # Calculate theme score based on keyword matches
            score = sum(1 for keyword in keywords if keyword in full_text)
            
            if score > 0:
                matching_posts.append({
                    'post_id': row['post_id'],
                    'title': row['title'],
                    'score': score,
                    'cluster': row['kmeans_cluster']
                })
                theme_scores.append(score)
        
        if matching_posts:
            theme_analysis[theme] = {
                'posts': sorted(matching_posts, key=lambda x: x['score'], reverse=True),
                'total_posts': len(matching_posts),
                'avg_score': np.mean(theme_scores),
                'top_score': max(theme_scores)
            }
    
    # Sort themes by relevance (number of posts)
    sorted_themes = sorted(theme_analysis.items(), 
                          key=lambda x: x[1]['total_posts'], 
                          reverse=True)
    
    print(f"✅ Identified {len(sorted_themes)} aggadic themes")
    return dict(sorted_themes)

def create_tractate_index(df):
    """Create index by Talmudic tractate"""
    print("\n📖 Creating Talmudic tractate index...")
    
    tractate_patterns = {
        'Sanhedrin': r'\bSanhedrin\b',
        'Shabbat': r'\bShabbat\b',
        'Pesachim': r'\bPesachim\b',
        'Yoma': r'\bYoma\b',
        'Gittin': r'\bGittin\b',
        'Megillah': r'\bMegillah\b',
        'Avodah Zarah': r'\bAvodah Zarah\b',
        'Taanit': r'\bTaanit\b',
        'Eruvin': r'\bEruvin\b',
        'Kiddushin': r'\bKiddushin\b',
        'Bava Batra': r'\bBava Batra\b',
        'Bava Metzia': r'\bBava Metzia\b',
        'Bava Kamma': r'\bBava Kamma\b',
        'Sotah': r'\bSotah\b',
        'Bekhorot': r'\bBekhorot\b',
        'Chullin': r'\bChullin\b',
        'Menachot': r'\bMenachot\b',
        'Sukkah': r'\bSukkah\b',
        'Rosh Hashanah': r'\bRosh Hashanah\b',
        'Moed Katan': r'\bMoed Katan\b',
        'Chagigah': r'\bChagigah\b',
        'Yevamot': r'\bYevamot\b',
        'Ketubot': r'\bKetubot\b',
        'Nedarim': r'\bNedarim\b',
        'Nazir': r'\bNazir\b'
    }
    
    tractate_index = {}
    
    for tractate, pattern in tractate_patterns.items():
        matching_posts = []
        
        for idx, row in df.iterrows():
            title = row['title'] if pd.notna(row['title']) else ''
            content = row['extracted_text'] if pd.notna(row['extracted_text']) else ''
            full_text = title + ' ' + content
            
            if re.search(pattern, full_text, re.IGNORECASE):
                matching_posts.append({
                    'post_id': row['post_id'],
                    'title': row['title'],
                    'cluster': row['kmeans_cluster'],
                    'word_count': row.get('word_count', 0)
                })
        
        if matching_posts:
            tractate_index[tractate] = {
                'posts': sorted(matching_posts, key=lambda x: x.get('word_count', 0), reverse=True),
                'total_posts': len(matching_posts)
            }
    
    print(f"✅ Created tractate index for {len(tractate_index)} tractates")
    return tractate_index

def generate_semantic_index(df, aggadic_themes, tractate_index):
    """Generate the complete semantic index"""
    print("\n🗂️  Generating comprehensive semantic index...")
    
    # Primary categories analysis
    primary_categories = {
        'Aggadah': len([p for theme in aggadic_themes.values() for p in theme['posts']]),
        'Halakha': len([idx for idx, row in df.iterrows() 
                       if any(word in (row['title'] or '').lower() 
                             for word in ['law', 'legal', 'ruling', 'halakha', 'halakhic'])]),
        'Biblical_Commentary': len([idx for idx, row in df.iterrows() 
                                   if any(word in (row['title'] or '').lower() 
                                         for word in ['biblical', 'bible', 'scripture', 'interpretation'])]),
        'Historical_Studies': len([idx for idx, row in df.iterrows() 
                                  if any(word in (row['title'] or '').lower() 
                                        for word in ['historical', 'history', 'period', 'era'])]),
        'Digital_Humanities': len([idx for idx, row in df.iterrows() 
                                  if any(word in (row['title'] or '').lower() 
                                        for word in ['digital', 'computational', 'algorithm', 'ai', 'technology'])])
    }
    
    # Reading time analysis
    reading_times = {}
    for idx, row in df.iterrows():
        word_count = row.get('word_count', 0)
        if word_count > 0:
            reading_time = word_count // 200  # ~200 words per minute
            if reading_time <= 5:
                category = 'Quick Read (≤5 min)'
            elif reading_time <= 15:
                category = 'Standard Read (6-15 min)'
            elif reading_time <= 30:
                category = 'Long Read (16-30 min)'
            else:
                category = 'Deep Study (30+ min)'
            
            if category not in reading_times:
                reading_times[category] = []
            reading_times[category].append({
                'post_id': row['post_id'],
                'title': row['title'],
                'reading_time': reading_time,
                'cluster': row['kmeans_cluster']
            })
    
    semantic_index = {
        'metadata': {
            'total_posts': len(df),
            'generation_date': '2025-06-30',
            'primary_focus': 'Aggadic Narratives and Talmudic Studies'
        },
        'primary_categories': primary_categories,
        'aggadic_themes': aggadic_themes,
        'tractate_index': tractate_index,
        'reading_times': reading_times,
        'cluster_distribution': df['kmeans_cluster'].value_counts().to_dict()
    }
    
    return semantic_index

def export_semantic_index(semantic_index):
    """Export the semantic index in multiple formats"""
    print("\n📤 Exporting semantic index...")
    
    # JSON export for programmatic use
    with open('SEMANTIC_INDEX.json', 'w', encoding='utf-8') as f:
        json.dump(semantic_index, f, ensure_ascii=False, indent=2, default=str)
    
    # Markdown export for human reading
    with open('SEMANTIC_INDEX.md', 'w', encoding='utf-8') as f:
        f.write("# Comprehensive Semantic Index: Talmudic Blog Archive\n\n")
        f.write("*A multi-layered thematic organization of 539 blog posts*\n\n")
        
        # Overview
        f.write("## Overview\n\n")
        f.write(f"**Total Posts Analyzed:** {semantic_index['metadata']['total_posts']}\n")
        f.write(f"**Primary Focus:** {semantic_index['metadata']['primary_focus']}\n")
        f.write(f"**Generation Date:** {semantic_index['metadata']['generation_date']}\n\n")
        
        # Primary Categories
        f.write("## Primary Categories\n\n")
        for category, count in semantic_index['primary_categories'].items():
            f.write(f"- **{category.replace('_', ' ')}:** {count} posts\n")
        f.write("\n")
        
        # Aggadic Themes (Top 25)
        f.write("## Aggadic Themes (Secondary Categories)\n\n")
        f.write("*Ranked by number of relevant posts*\n\n")
        
        top_themes = sorted(semantic_index['aggadic_themes'].items(), 
                           key=lambda x: x[1]['total_posts'], reverse=True)[:25]
        
        for i, (theme, data) in enumerate(top_themes, 1):
            theme_name = theme.replace('_', ' ').title()
            f.write(f"### {i}. {theme_name}\n")
            f.write(f"**Posts:** {data['total_posts']} | **Relevance Score:** {data['avg_score']:.1f}\n\n")
            
            # Show top 5 posts for this theme
            f.write("**Featured Posts:**\n")
            for j, post in enumerate(data['posts'][:5], 1):
                f.write(f"{j}. {post['title']} (Score: {post['score']})\n")
            f.write("\n")
        
        # Tractate Index
        f.write("## Talmudic Tractate Index\n\n")
        for tractate, data in sorted(semantic_index['tractate_index'].items(), 
                                   key=lambda x: x[1]['total_posts'], reverse=True):
            f.write(f"### {tractate} ({data['total_posts']} posts)\n")
            for post in data['posts'][:3]:  # Top 3 posts per tractate
                f.write(f"- {post['title']}\n")
            f.write("\n")
        
        # Reading Time Categories
        f.write("## By Reading Time\n\n")
        for category, posts in semantic_index['reading_times'].items():
            f.write(f"### {category} ({len(posts)} posts)\n")
            # Show a few examples
            for post in posts[:3]:
                f.write(f"- {post['title']} (~{post['reading_time']} min)\n")
            f.write("\n")
    
    print("✅ Generated SEMANTIC_INDEX.json and SEMANTIC_INDEX.md")

def main():
    print("🗂️  Comprehensive Semantic Index Generator")
    print("Aggadah-Focused Multi-Layered Thematic Organization")
    print("=" * 60)
    
    # Load data
    df = load_blog_data()
    
    # Identify aggadic themes
    aggadic_themes = identify_aggadic_themes(df)
    
    # Create tractate index
    tractate_index = create_tractate_index(df)
    
    # Generate comprehensive semantic index
    semantic_index = generate_semantic_index(df, aggadic_themes, tractate_index)
    
    # Export results
    export_semantic_index(semantic_index)
    
    print(f"\n🎯 Semantic Index Complete!")
    print(f"   • {len(aggadic_themes)} aggadic themes identified")
    print(f"   • {len(tractate_index)} tractates indexed")
    print(f"   • Multi-format exports generated")
    print(f"   • Ready for scholarly navigation!")

if __name__ == "__main__":
    main()
