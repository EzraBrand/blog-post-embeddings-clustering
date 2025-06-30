#!/usr/bin/env python3
"""
Quick Data Analysis of Extracted Blog Posts
==========================================

This script provides quick analysis and exploration of the extracted blog post data.
"""

import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from datetime import datetime
from pathlib import Path


def load_extracted_data(data_dir: str = "processed_data") -> pd.DataFrame:
    """Load the extracted posts data"""
    csv_path = Path(data_dir) / "extracted_posts.csv"
    return pd.read_csv(csv_path)


def analyze_content_distribution(df: pd.DataFrame):
    """Analyze content length and word count distributions"""
    # Filter successful extractions
    success_df = df[df['extraction_success'] == True]
    
    print("=== CONTENT DISTRIBUTION ANALYSIS ===")
    print(f"Total posts: {len(df)}")
    print(f"Successful extractions: {len(success_df)}")
    print(f"Success rate: {len(success_df)/len(df):.1%}")
    
    print(f"\nWord Count Statistics:")
    print(f"  Mean: {success_df['word_count'].mean():.0f}")
    print(f"  Median: {success_df['word_count'].median():.0f}")
    print(f"  Min: {success_df['word_count'].min()}")
    print(f"  Max: {success_df['word_count'].max()}")
    print(f"  Std: {success_df['word_count'].std():.0f}")
    
    # Content length categories
    def categorize_length(word_count):
        if word_count < 100:
            return "Very Short"
        elif word_count < 500:
            return "Short"
        elif word_count < 1000:
            return "Medium"
        elif word_count < 2000:
            return "Long"
        else:
            return "Very Long"
    
    success_df['length_category'] = success_df['word_count'].apply(categorize_length)
    category_counts = success_df['length_category'].value_counts()
    
    print(f"\nContent Length Categories:")
    for category, count in category_counts.items():
        pct = count / len(success_df) * 100
        print(f"  {category}: {count} posts ({pct:.1f}%)")


def analyze_temporal_patterns(df: pd.DataFrame):
    """Analyze publication date patterns"""
    success_df = df[df['extraction_success'] == True]
    
    # Convert publication dates
    success_df['pub_date'] = pd.to_datetime(success_df['publication_date'], errors='coerce')
    success_df = success_df.dropna(subset=['pub_date'])
    
    print("\n=== TEMPORAL PATTERNS ===")
    print(f"Date range: {success_df['pub_date'].min().date()} to {success_df['pub_date'].max().date()}")
    
    # Posts per year
    success_df['year'] = success_df['pub_date'].dt.year
    yearly_counts = success_df['year'].value_counts().sort_index()
    
    print(f"\nPosts by Year:")
    for year, count in yearly_counts.items():
        print(f"  {year}: {count} posts")
    
    # Average word count by year
    avg_words_by_year = success_df.groupby('year')['word_count'].mean()
    print(f"\nAverage Word Count by Year:")
    for year, avg_words in avg_words_by_year.items():
        print(f"  {year}: {avg_words:.0f} words")


def analyze_content_themes(df: pd.DataFrame):
    """Analyze content themes based on titles"""
    success_df = df[df['extraction_success'] == True]
    
    print("\n=== CONTENT THEMES (Based on Titles) ===")
    
    # Common words in titles
    all_titles = " ".join(success_df['title'].fillna("")).lower()
    
    # Remove common words and extract meaningful terms
    stop_words = {'the', 'and', 'of', 'in', 'to', 'a', 'an', 'is', 'for', 'on', 'with', 'by', 'from', 'as', 'at', 'vs'}
    words = re.findall(r'\b[a-zA-Z]{3,}\b', all_titles)
    words = [word for word in words if word not in stop_words]
    
    common_words = Counter(words).most_common(20)
    
    print("Most Common Terms in Titles:")
    for word, count in common_words:
        print(f"  {word}: {count}")
    
    # Look for Hebrew/Aramaic terms
    hebrew_pattern = re.compile(r'[\u0590-\u05FF]+')
    titles_with_hebrew = success_df[success_df['title'].str.contains(hebrew_pattern, na=False)]
    
    print(f"\nTitles containing Hebrew: {len(titles_with_hebrew)}/{len(success_df)} ({len(titles_with_hebrew)/len(success_df):.1%})")


def analyze_audience_and_type(df: pd.DataFrame):
    """Analyze audience and post type distribution"""
    success_df = df[df['extraction_success'] == True]
    
    print("\n=== AUDIENCE AND TYPE ANALYSIS ===")
    
    # Audience distribution
    audience_counts = success_df['audience'].value_counts()
    print("Audience Distribution:")
    for audience, count in audience_counts.items():
        pct = count / len(success_df) * 100
        print(f"  {audience}: {count} posts ({pct:.1f}%)")
    
    # Post type distribution
    type_counts = success_df['post_type'].value_counts()
    print("\nPost Type Distribution:")
    for post_type, count in type_counts.items():
        pct = count / len(success_df) * 100
        print(f"  {post_type}: {count} posts ({pct:.1f}%)")


def find_longest_and_shortest_posts(df: pd.DataFrame):
    """Find the longest and shortest posts"""
    success_df = df[df['extraction_success'] == True]
    
    print("\n=== CONTENT EXTREMES ===")
    
    # Longest posts
    longest_posts = success_df.nlargest(5, 'word_count')[['post_id', 'title', 'word_count']]
    print("Longest Posts:")
    for _, post in longest_posts.iterrows():
        print(f"  {post['word_count']} words: {post['title'][:80]}...")
    
    # Shortest posts
    shortest_posts = success_df.nsmallest(5, 'word_count')[['post_id', 'title', 'word_count']]
    print("\nShortest Posts:")
    for _, post in shortest_posts.iterrows():
        print(f"  {post['word_count']} words: {post['title'][:80]}...")


def analyze_extraction_errors(df: pd.DataFrame):
    """Analyze extraction errors"""
    failed_df = df[df['extraction_success'] == False]
    
    print("\n=== EXTRACTION ERRORS ANALYSIS ===")
    print(f"Failed extractions: {len(failed_df)}")
    
    if len(failed_df) > 0:
        print("\nFailed Post IDs:")
        for post_id in failed_df['post_id']:
            print(f"  {post_id}")
        
        # Error patterns (would need to parse extraction_errors column if available)
        print(f"\nThese appear to be administrative/meta posts rather than content.")


def create_summary_stats(df: pd.DataFrame) -> dict:
    """Create summary statistics for the dataset"""
    success_df = df[df['extraction_success'] == True]
    
    stats = {
        'total_posts': len(df),
        'successful_extractions': len(success_df),
        'success_rate': len(success_df) / len(df),
        'total_words': success_df['word_count'].sum(),
        'avg_words': success_df['word_count'].mean(),
        'median_words': success_df['word_count'].median(),
        'total_characters': success_df['content_length'].sum(),
        'posts_over_1000_words': len(success_df[success_df['word_count'] >= 1000]),
        'posts_over_2000_words': len(success_df[success_df['word_count'] >= 2000]),
    }
    
    return stats


def main():
    """Run comprehensive analysis"""
    print("Blog Post Content Analysis")
    print("=" * 50)
    
    # Load data
    try:
        df = load_extracted_data()
    except FileNotFoundError:
        print("Error: Could not find extracted_posts.csv")
        print("Make sure you've run the extraction pipeline first.")
        return
    
    # Run analyses
    analyze_content_distribution(df)
    analyze_temporal_patterns(df)
    analyze_content_themes(df)
    analyze_audience_and_type(df)
    find_longest_and_shortest_posts(df)
    analyze_extraction_errors(df)
    
    # Summary
    stats = create_summary_stats(df)
    
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"Total posts processed: {stats['total_posts']}")
    print(f"Successful extractions: {stats['successful_extractions']} ({stats['success_rate']:.1%})")
    print(f"Total word count: {stats['total_words']:,} words")
    print(f"Total characters: {stats['total_characters']:,} characters")
    print(f"Average post length: {stats['avg_words']:.0f} words")
    print(f"Median post length: {stats['median_words']:.0f} words")
    print(f"Long posts (1000+ words): {stats['posts_over_1000_words']} ({stats['posts_over_1000_words']/stats['successful_extractions']:.1%})")
    print(f"Very long posts (2000+ words): {stats['posts_over_2000_words']} ({stats['posts_over_2000_words']/stats['successful_extractions']:.1%})")
    
    print(f"\nDataset ready for embedding generation and clustering!")


if __name__ == "__main__":
    main()
