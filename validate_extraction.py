#!/usr/bin/env python3
"""
HTML Extraction Testing and Validation Utilities
===============================================

Utility functions for testing the HTML extraction pipeline and validating results.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import re

from html_extractor import ExtractedPost, HTMLContentExtractor


def test_single_extraction(html_file_path: str, verbose: bool = True) -> ExtractedPost:
    """
    Test extraction on a single HTML file for debugging
    
    Args:
        html_file_path: Path to HTML file to test
        verbose: Whether to print detailed output
        
    Returns:
        ExtractedPost object
    """
    extractor = HTMLContentExtractor()
    result = extractor.extract_single_post(Path(html_file_path))
    
    if verbose:
        print(f"=== EXTRACTION TEST: {html_file_path} ===")
        print(f"Post ID: {result.post_id}")
        print(f"Success: {result.extraction_success}")
        print(f"Title: {result.title[:100]}..." if len(result.title) > 100 else result.title)
        print(f"Word Count: {result.word_count}")
        print(f"Content Length: {result.content_length}")
        
        if result.extraction_errors:
            print(f"Errors: {result.extraction_errors}")
        
        if result.extracted_text:
            preview = result.extracted_text[:300] + "..." if len(result.extracted_text) > 300 else result.extracted_text
            print(f"Content Preview:\n{preview}")
    
    return result


def validate_hebrew_content(posts: List[ExtractedPost]) -> Dict[str, int]:
    """
    Validate that Hebrew content is properly preserved
    
    Args:
        posts: List of ExtractedPost objects
        
    Returns:
        Dictionary with Hebrew content statistics
    """
    hebrew_pattern = re.compile(r'[\u0590-\u05FF]')  # Hebrew Unicode range
    
    stats = {
        'total_posts': len(posts),
        'posts_with_hebrew': 0,
        'posts_with_substantial_hebrew': 0,  # >10 Hebrew characters
        'avg_hebrew_chars_per_post': 0
    }
    
    total_hebrew_chars = 0
    
    for post in posts:
        if not post.extraction_success:
            continue
            
        hebrew_chars = len(hebrew_pattern.findall(post.extracted_text))
        total_hebrew_chars += hebrew_chars
        
        if hebrew_chars > 0:
            stats['posts_with_hebrew'] += 1
            
        if hebrew_chars > 10:
            stats['posts_with_substantial_hebrew'] += 1
    
    if posts:
        stats['avg_hebrew_chars_per_post'] = total_hebrew_chars / len(posts)
    
    return stats


def validate_metadata_alignment(posts: List[ExtractedPost], metadata_file: str = "posts.csv") -> Dict[str, int]:
    """
    Validate alignment between extracted posts and metadata
    
    Args:
        posts: List of ExtractedPost objects
        metadata_file: Path to metadata CSV file
        
    Returns:
        Dictionary with alignment statistics
    """
    try:
        metadata_df = pd.read_csv(metadata_file)
        metadata_ids = set(metadata_df['post_id'].astype(str))
        extracted_ids = set(post.post_id for post in posts)
        
        stats = {
            'metadata_posts': len(metadata_ids),
            'extracted_posts': len(extracted_ids),
            'posts_in_both': len(metadata_ids.intersection(extracted_ids)),
            'posts_only_in_metadata': len(metadata_ids - extracted_ids),
            'posts_only_in_extracted': len(extracted_ids - metadata_ids)
        }
        
        return stats
        
    except Exception as e:
        return {'error': str(e)}


def analyze_content_quality(posts: List[ExtractedPost]) -> Dict:
    """
    Analyze the quality of extracted content
    
    Args:
        posts: List of ExtractedPost objects
        
    Returns:
        Dictionary with quality metrics
    """
    successful_posts = [post for post in posts if post.extraction_success]
    
    if not successful_posts:
        return {'error': 'No successful extractions to analyze'}
    
    word_counts = [post.word_count for post in successful_posts]
    content_lengths = [post.content_length for post in successful_posts]
    
    # Quality thresholds
    very_short = sum(1 for wc in word_counts if wc < 50)
    short = sum(1 for wc in word_counts if 50 <= wc < 100)
    medium = sum(1 for wc in word_counts if 100 <= wc < 500)
    long = sum(1 for wc in word_counts if 500 <= wc < 1000)
    very_long = sum(1 for wc in word_counts if wc >= 1000)
    
    stats = {
        'total_successful': len(successful_posts),
        'word_count_distribution': {
            'very_short_(<50)': very_short,
            'short_(50-99)': short,
            'medium_(100-499)': medium,
            'long_(500-999)': long,
            'very_long_(1000+)': very_long
        },
        'word_count_stats': {
            'min': min(word_counts),
            'max': max(word_counts),
            'mean': sum(word_counts) / len(word_counts),
            'median': sorted(word_counts)[len(word_counts) // 2]
        },
        'content_length_stats': {
            'min': min(content_lengths),
            'max': max(content_lengths),
            'mean': sum(content_lengths) / len(content_lengths)
        }
    }
    
    return stats


def find_posts_with_issues(posts: List[ExtractedPost]) -> Dict[str, List[str]]:
    """
    Identify posts with potential extraction issues
    
    Args:
        posts: List of ExtractedPost objects
        
    Returns:
        Dictionary categorizing posts by issue type
    """
    issues = {
        'failed_extraction': [],
        'very_short_content': [],
        'no_title': [],
        'title_body_mismatch': [],
        'potential_encoding_issues': []
    }
    
    for post in posts:
        if not post.extraction_success:
            issues['failed_extraction'].append(post.post_id)
            continue
        
        if post.word_count < 20:
            issues['very_short_content'].append(post.post_id)
        
        if not post.title_text or len(post.title_text.strip()) < 5:
            issues['no_title'].append(post.post_id)
        
        # Check for title/body similarity (might indicate extraction duplication)
        if post.title_text and post.body_text:
            if post.title_text.lower() in post.body_text.lower()[:200]:
                issues['title_body_mismatch'].append(post.post_id)
        
        # Look for potential encoding issues
        if '�' in post.extracted_text or 'â€' in post.extracted_text:
            issues['potential_encoding_issues'].append(post.post_id)
    
    return issues


def sample_posts_for_manual_review(posts: List[ExtractedPost], n: int = 5) -> List[ExtractedPost]:
    """
    Select a sample of posts for manual quality review
    
    Args:
        posts: List of ExtractedPost objects
        n: Number of posts to sample
        
    Returns:
        List of sampled posts
    """
    successful_posts = [post for post in posts if post.extraction_success]
    
    if not successful_posts:
        return []
    
    # Sample posts from different word count ranges for diversity
    sorted_posts = sorted(successful_posts, key=lambda p: p.word_count)
    
    sample_indices = []
    total = len(sorted_posts)
    
    for i in range(n):
        idx = (i * total) // n
        sample_indices.append(idx)
    
    return [sorted_posts[i] for i in sample_indices]


def run_full_validation(data_dir: str = "processed_data") -> Dict:
    """
    Run complete validation suite on extracted data
    
    Args:
        data_dir: Directory containing extracted data
        
    Returns:
        Dictionary with all validation results
    """
    data_path = Path(data_dir)
    
    # Load extracted posts
    try:
        with open(data_path / "extracted_posts.pkl", 'rb') as f:
            posts = pickle.load(f)
    except FileNotFoundError:
        try:
            with open(data_path / "extracted_posts.json", 'r') as f:
                posts_data = json.load(f)
                posts = [ExtractedPost(**post_data) for post_data in posts_data]
        except FileNotFoundError:
            return {'error': 'No extracted data files found'}
    
    print("Running full validation suite...")
    
    # Run all validations
    results = {
        'hebrew_content': validate_hebrew_content(posts),
        'metadata_alignment': validate_metadata_alignment(posts),
        'content_quality': analyze_content_quality(posts),
        'posts_with_issues': find_posts_with_issues(posts),
        'sample_posts': [post.post_id for post in sample_posts_for_manual_review(posts)]
    }
    
    # Save validation results
    output_file = data_path / "validation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"Validation results saved to: {output_file}")
    
    # Print summary
    print("\n=== VALIDATION SUMMARY ===")
    if 'hebrew_content' in results:
        hc = results['hebrew_content']
        print(f"Hebrew content: {hc['posts_with_hebrew']}/{hc['total_posts']} posts contain Hebrew")
    
    if 'content_quality' in results and 'error' not in results['content_quality']:
        cq = results['content_quality']
        print(f"Content quality: {cq['total_successful']} successful extractions")
        print(f"Average word count: {cq['word_count_stats']['mean']:.0f}")
    
    if 'posts_with_issues' in results:
        issues = results['posts_with_issues']
        total_issues = sum(len(posts) for posts in issues.values())
        print(f"Posts with issues: {total_issues}")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            # Test single file
            if len(sys.argv) > 2:
                test_single_extraction(sys.argv[2])
            else:
                print("Usage: python validate_extraction.py test <html_file_path>")
        elif sys.argv[1] == "validate":
            # Run full validation
            run_full_validation()
        else:
            print("Usage: python validate_extraction.py [test <file> | validate]")
    else:
        print("HTML Extraction Validation Utilities")
        print("Usage:")
        print("  python validate_extraction.py test <html_file>  - Test single file")
        print("  python validate_extraction.py validate         - Run full validation")
