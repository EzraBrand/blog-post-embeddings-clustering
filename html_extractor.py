#!/usr/bin/env python3
"""
HTML Content Extraction and Preprocessing Pipeline for Blog Post Clustering
==========================================================================

This script extracts clean text content from HTML blog posts, integrates metadata,
and preprocesses the content for embedding generation and clustering analysis.

Author: GitHub Copilot
Created: June 30, 2025
"""

import os
import re
import sys
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, asdict

import pandas as pd
from bs4 import BeautifulSoup, Comment
import html2text
from tqdm import tqdm


@dataclass
class ExtractedPost:
    """Data structure for extracted blog post content"""
    post_id: str
    title: str
    extracted_text: str
    publication_date: str
    content_length: int
    word_count: int
    title_text: str
    body_text: str
    extraction_success: bool
    extraction_errors: List[str]
    original_file: str
    subtitle: str = ""
    audience: str = ""
    post_type: str = ""


class HTMLContentExtractor:
    """Main class for extracting and preprocessing HTML blog content"""
    
    def __init__(self, posts_dir: str = "posts", metadata_file: str = "posts.csv", 
                 output_dir: str = "processed_data"):
        """
        Initialize the HTML content extractor
        
        Args:
            posts_dir: Directory containing HTML files
            metadata_file: CSV file with post metadata
            output_dir: Directory for output files
        """
        self.posts_dir = Path(posts_dir)
        self.metadata_file = Path(metadata_file)
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize HTML to text converter
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = True
        self.html_converter.ignore_emphasis = False
        self.html_converter.body_width = 0  # No line wrapping
        
        # Load metadata
        self.metadata_df = self._load_metadata()
        
        self.logger.info(f"Initialized extractor for {len(self.metadata_df)} posts")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / "extraction.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load and prepare metadata from CSV file"""
        try:
            df = pd.read_csv(self.metadata_file)
            self.logger.info(f"Loaded metadata for {len(df)} posts")
            
            # Clean and validate data
            df = df[df['is_published'] == True]  # Only published posts
            df = df.dropna(subset=['post_id', 'title'])  # Must have ID and title
            
            self.logger.info(f"After filtering: {len(df)} published posts with valid metadata")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading metadata: {e}")
            raise
    
    def _extract_post_id_from_filename(self, filename: str) -> str:
        """Extract post ID from HTML filename"""
        # Format: {post_id}.{slug}.html -> remove .html extension
        # The post_id in CSV includes the slug, so we need the full name without .html
        return filename.replace('.html', '')
    
    def _clean_html_content(self, html_content: str) -> BeautifulSoup:
        """Clean and prepare HTML content for text extraction"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Remove HTML comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        # Remove or clean specific elements that don't add value
        for element in soup.find_all(['nav', 'header', 'footer', 'aside']):
            element.decompose()
        
        return soup
    
    def _extract_title_from_html(self, soup: BeautifulSoup) -> str:
        """Extract title from HTML structure"""
        # Try different title extraction methods
        title_selectors = ['h1', 'h2', 'title', '.post-title', '.entry-title']
        
        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem:
                title = title_elem.get_text(strip=True)
                if title and len(title) > 5:  # Basic validation
                    return title
        
        return ""
    
    def _extract_text_content(self, soup: BeautifulSoup) -> Tuple[str, str]:
        """
        Extract clean text from HTML soup
        
        Returns:
            Tuple of (title_text, body_text)
        """
        # Extract title
        title_text = self._extract_title_from_html(soup)
        
        # Extract body content
        # Remove title elements to avoid duplication in body
        for title_elem in soup.find_all(['h1', 'h2', 'title']):
            if title_elem.get_text(strip=True) == title_text:
                title_elem.decompose()
        
        # Convert remaining HTML to markdown-style text
        body_html = str(soup)
        body_text = self.html_converter.handle(body_html)
        
        # Clean up the text
        body_text = self._post_process_text(body_text)
        
        return title_text, body_text
    
    def _post_process_text(self, text: str) -> str:
        """Post-process extracted text for better quality"""
        # Remove excessive whitespace while preserving paragraph breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Clean up common artifacts
        text = re.sub(r'\[\d+\]', '', text)  # Remove footnote markers like [1]
        text = re.sub(r'^\s*[\*\-\+]\s*', '', text, flags=re.MULTILINE)  # Clean bullet points
        
        # Remove isolated special characters
        text = re.sub(r'\n[^\w\s]\n', '\n', text)
        
        # Preserve Hebrew text (basic check)
        # Hebrew Unicode range: U+0590 to U+05FF
        
        return text.strip()
    
    def _calculate_content_statistics(self, title: str, body: str) -> Tuple[int, int]:
        """Calculate content statistics"""
        combined_text = f"{title} {body}"
        content_length = len(combined_text)
        
        # Word count (handles both English and Hebrew)
        words = re.findall(r'\b\w+\b', combined_text, re.UNICODE)
        word_count = len(words)
        
        return content_length, word_count
    
    def extract_single_post(self, html_file_path: Path) -> Optional[ExtractedPost]:
        """Extract content from a single HTML file"""
        post_id = self._extract_post_id_from_filename(html_file_path.name)
        errors = []
        
        try:
            # Read HTML file
            with open(html_file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Check if file is too small (likely empty or malformed)
            if len(html_content) < 100:
                errors.append("HTML file too small")
                return self._create_failed_extraction(post_id, html_file_path, errors)
            
            # Clean HTML
            soup = self._clean_html_content(html_content)
            
            # Extract text content
            title_text, body_text = self._extract_text_content(soup)
            
            # Validate extraction
            if not body_text or len(body_text) < 50:
                errors.append("Body text too short or empty")
            
            # Get metadata
            metadata = self._get_post_metadata(post_id)
            if metadata is None:
                errors.append("No metadata found")
                title = title_text  # Use extracted title as fallback
                publication_date = ""
                subtitle = ""
                audience = ""
                post_type = ""
            else:
                title = metadata.get('title', title_text)
                publication_date = metadata.get('post_date', '')
                subtitle = metadata.get('subtitle', '')
                audience = metadata.get('audience', '')
                post_type = metadata.get('type', '')
            
            # Calculate statistics
            content_length, word_count = self._calculate_content_statistics(title_text, body_text)
            
            # Create combined text
            combined_text = f"{title_text}\n\n{body_text}".strip()
            
            return ExtractedPost(
                post_id=post_id,
                title=title,
                extracted_text=combined_text,
                publication_date=publication_date,
                content_length=content_length,
                word_count=word_count,
                title_text=title_text,
                body_text=body_text,
                extraction_success=len(errors) == 0,
                extraction_errors=errors,
                original_file=str(html_file_path),
                subtitle=subtitle,
                audience=audience,
                post_type=post_type
            )
            
        except Exception as e:
            errors.append(f"Extraction error: {str(e)}")
            self.logger.error(f"Error extracting {html_file_path}: {e}")
            return self._create_failed_extraction(post_id, html_file_path, errors)
    
    def _create_failed_extraction(self, post_id: str, html_file_path: Path, 
                                errors: List[str]) -> ExtractedPost:
        """Create an ExtractedPost object for failed extractions"""
        metadata = self._get_post_metadata(post_id)
        title = metadata.get('title', 'Unknown') if metadata else 'Unknown'
        publication_date = metadata.get('post_date', '') if metadata else ''
        
        return ExtractedPost(
            post_id=post_id,
            title=title,
            extracted_text="",
            publication_date=publication_date,
            content_length=0,
            word_count=0,
            title_text="",
            body_text="",
            extraction_success=False,
            extraction_errors=errors,
            original_file=str(html_file_path),
            subtitle="",
            audience="",
            post_type=""
        )
    
    def _get_post_metadata(self, post_id: str) -> Optional[Dict]:
        """Get metadata for a specific post ID"""
        matching_rows = self.metadata_df[self.metadata_df['post_id'] == post_id]
        if len(matching_rows) > 0:
            return matching_rows.iloc[0].to_dict()
        return None
    
    def extract_all_posts(self) -> List[ExtractedPost]:
        """Extract content from all HTML files"""
        html_files = list(self.posts_dir.glob("*.html"))
        self.logger.info(f"Found {len(html_files)} HTML files to process")
        
        extracted_posts = []
        
        # Process files with progress bar
        for html_file in tqdm(html_files, desc="Extracting posts"):
            extracted_post = self.extract_single_post(html_file)
            if extracted_post:
                extracted_posts.append(extracted_post)
        
        self.logger.info(f"Successfully processed {len(extracted_posts)} files")
        
        # Log extraction statistics
        successful = sum(1 for post in extracted_posts if post.extraction_success)
        failed = len(extracted_posts) - successful
        
        self.logger.info(f"Extraction complete: {successful} successful, {failed} failed")
        
        return extracted_posts
    
    def save_extracted_data(self, extracted_posts: List[ExtractedPost]) -> Dict[str, str]:
        """Save extracted data in multiple formats"""
        output_files = {}
        
        # Convert to list of dictionaries
        posts_data = [asdict(post) for post in extracted_posts]
        
        # Save as JSON
        json_file = self.output_dir / "extracted_posts.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(posts_data, f, ensure_ascii=False, indent=2)
        output_files['json'] = str(json_file)
        
        # Save as pickle (preserves exact Python objects)
        pickle_file = self.output_dir / "extracted_posts.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(extracted_posts, f)
        output_files['pickle'] = str(pickle_file)
        
        # Save as CSV
        csv_file = self.output_dir / "extracted_posts.csv"
        df = pd.DataFrame(posts_data)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        output_files['csv'] = str(csv_file)
        
        # Save text files for successful extractions (useful for manual review)
        text_dir = self.output_dir / "text_files"
        text_dir.mkdir(exist_ok=True)
        
        successful_posts = [post for post in extracted_posts if post.extraction_success]
        for post in successful_posts:
            text_file = text_dir / f"{post.post_id}.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(f"Title: {post.title}\n")
                f.write(f"Publication Date: {post.publication_date}\n")
                f.write(f"Word Count: {post.word_count}\n")
                f.write("-" * 50 + "\n\n")
                f.write(post.extracted_text)
        
        output_files['text_dir'] = str(text_dir)
        
        self.logger.info(f"Saved extracted data to: {', '.join(output_files.values())}")
        return output_files
    
    def generate_quality_report(self, extracted_posts: List[ExtractedPost]) -> Dict:
        """Generate a quality report on the extracted content"""
        successful_posts = [post for post in extracted_posts if post.extraction_success]
        failed_posts = [post for post in extracted_posts if not post.extraction_success]
        
        # Basic statistics
        stats = {
            'total_posts': len(extracted_posts),
            'successful_extractions': len(successful_posts),
            'failed_extractions': len(failed_posts),
            'success_rate': len(successful_posts) / len(extracted_posts) if extracted_posts else 0
        }
        
        if successful_posts:
            # Content statistics
            word_counts = [post.word_count for post in successful_posts]
            content_lengths = [post.content_length for post in successful_posts]
            
            stats.update({
                'avg_word_count': sum(word_counts) / len(word_counts),
                'min_word_count': min(word_counts),
                'max_word_count': max(word_counts),
                'avg_content_length': sum(content_lengths) / len(content_lengths),
                'min_content_length': min(content_lengths),
                'max_content_length': max(content_lengths)
            })
            
            # Filter for quality content (reasonable length)
            quality_posts = [post for post in successful_posts if post.word_count >= 100]
            stats['quality_posts'] = len(quality_posts)
            stats['quality_rate'] = len(quality_posts) / len(successful_posts)
        
        # Error analysis
        error_counts = {}
        for post in failed_posts:
            for error in post.extraction_errors:
                error_counts[error] = error_counts.get(error, 0) + 1
        
        stats['common_errors'] = error_counts
        
        # Save report
        report_file = self.output_dir / "quality_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        # Log summary
        self.logger.info("=== EXTRACTION QUALITY REPORT ===")
        self.logger.info(f"Total posts processed: {stats['total_posts']}")
        self.logger.info(f"Successful extractions: {stats['successful_extractions']}")
        self.logger.info(f"Success rate: {stats['success_rate']:.2%}")
        
        if 'avg_word_count' in stats:
            self.logger.info(f"Average word count: {stats['avg_word_count']:.0f}")
            self.logger.info(f"Quality posts (≥100 words): {stats['quality_posts']}")
        
        if error_counts:
            self.logger.info(f"Most common errors: {dict(list(error_counts.items())[:3])}")
        
        return stats


def main():
    """Main execution function"""
    print("HTML Content Extraction and Preprocessing Pipeline")
    print("=" * 55)
    
    # Initialize extractor
    extractor = HTMLContentExtractor()
    
    # Extract all posts
    print("\n1. Extracting content from HTML files...")
    extracted_posts = extractor.extract_all_posts()
    
    # Save extracted data
    print("\n2. Saving extracted data...")
    output_files = extractor.save_extracted_data(extracted_posts)
    
    # Generate quality report
    print("\n3. Generating quality report...")
    quality_stats = extractor.generate_quality_report(extracted_posts)
    
    # Summary
    print("\n" + "=" * 55)
    print("EXTRACTION COMPLETE")
    print("=" * 55)
    print(f"Processed: {quality_stats['total_posts']} posts")
    print(f"Successful: {quality_stats['successful_extractions']} ({quality_stats['success_rate']:.1%})")
    
    if 'quality_posts' in quality_stats:
        print(f"Quality posts: {quality_stats['quality_posts']} (≥100 words)")
    
    print(f"\nOutput files:")
    for format_type, file_path in output_files.items():
        print(f"  {format_type.upper()}: {file_path}")
    
    print(f"\nReady for next phase: Embedding generation!")


if __name__ == "__main__":
    main()
