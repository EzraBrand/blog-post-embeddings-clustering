# Project Outline: Blog Post Clustering with Embeddings

## Current Status (June 30, 2025)

### ✅ PHASE 1 COMPLETE: HTML Content Extraction & Preprocessing

**Extraction Results**:
- **553 HTML files processed** with 97.8% success rate (539 successful extractions)
- **1.25 million words extracted** from scholarly blog posts
- **Perfect metadata alignment** - all HTML files matched with posts.csv entries
- **Hebrew content preserved** - 90.7% of posts contain Hebrew text with proper Unicode handling
- **Academic formatting maintained** - citations, footnotes, and scholarly structure intact

**Dataset Characteristics**:
- **Average post length**: 2,319 words (high-quality academic content)
- **Content distribution**: 81.8% are long-form posts (1000+ words)
- **Temporal range**: 2023-2025 with increasing post length over time
- **Thematic focus**: Talmudic analysis, Jewish studies, biblical commentary
- **Language content**: Mixed Hebrew/English academic writing

**Output Files Generated**:
- `processed_data/extracted_posts.csv` - Primary dataset (539 posts)
- `processed_data/extracted_posts.json` - JSON format for web applications
- `processed_data/text_files/` - Individual text files for manual review
- `processed_data/quality_report.json` - Extraction statistics and validation
- Complete validation and error analysis documentation

**Quality Metrics**:
- 99.8% of successful extractions meet quality thresholds (≥100 words)
- Failed extractions limited to administrative/meta posts (chat, placeholders)
- Hebrew Unicode preservation verified across 489 posts
- Academic citations and references properly maintained

**Ready for Next Phase**: Embedding Generation - Dataset optimized for semantic analysis

## Overview
This project focuses on automatically grouping blog posts using text embeddings and clustering algorithms. The goal is to identify thematically similar content and organize posts into meaningful clusters for better content discovery and analysis.

## 1. Data Preprocessing ✅ COMPLETED

### HTML Content Extraction ✅
- **Parse HTML files** to extract clean text content ✅
  - Strip HTML tags while preserving text structure ✅
  - Remove script and style elements ✅
  - Handle special HTML entities and encoding ✅
  - Preserve meaningful whitespace and paragraph breaks ✅

### Text Cleaning & Preprocessing ✅
- **Content normalization** ✅
  - Remove or standardize special characters ✅
  - Handle different encodings (UTF-8, ASCII, etc.) ✅
  - Normalize whitespace and line breaks ✅
  - Academic-appropriate preprocessing (preserve technical terms, citations) ✅

- **Text preprocessing pipeline** ✅
  - Hebrew/English mixed content handling ✅
  - Academic formatting preservation ✅
  - Quality filtering (minimum content length) ✅
  - Error handling and logging ✅

### Metadata Extraction ✅
- **Extract structured information** ✅
  - Post ID matching with CSV metadata ✅
  - Publication dates and temporal analysis ✅
  - Content statistics (word count, character count) ✅
  - Extraction success validation ✅

### Dataset Structure ✅
- **Created unified data format** ✅
  - 539 successfully extracted posts ✅
  - Clean text content (title + body separation) ✅
  - Complete metadata integration ✅
  - Multiple output formats (CSV, JSON, pickle) ✅
  - Quality validation and error reporting ✅

**Implementation Details**:
- Built robust HTML extraction pipeline using BeautifulSoup4 and html2text
- Implemented comprehensive validation suite with Hebrew content verification
- Generated detailed quality reports and extraction statistics
- Created individual text files for manual review and validation

## 2. Embedding Generation 🔄 NEXT PHASE

### Model Selection
- **Primary options to consider**
  - **OpenAI text-embedding-ada-002**: High quality, API-based, supports multilingual content
  - **Sentence Transformers**: Local models (paraphrase-multilingual-MiniLM-L12-v2 for Hebrew/English)
  - **Academic BERT models**: Specialized for scholarly content
  - **Hebrew BERT**: For Hebrew-heavy content sections

### Embedding Strategy - **Recommended for Academic Content**
- **Content combination approaches**
  - Combined title + body embeddings (recommended for academic coherence)
  - Weighted combination (title weight: 0.2, body weight: 0.8) 
  - Chunked processing for very long posts (36K+ words)
  - Separate Hebrew/English embeddings if needed

### Implementation Considerations - **Updated for Dataset**
- **Batch processing** for 539 posts efficiently
- **Rate limiting** for API-based models
- **Caching** to avoid re-computation during experimentation
- **Memory management** for long academic posts
- **Multilingual support** for Hebrew/English mixed content

### Storage & Management
- **Vector storage options**
  - Local storage (NumPy arrays, HDF5, Parquet) - recommended for dataset size
  - Vector databases (Chroma, Pinecone) for future scalability
  - Simple pickle integration with existing data structure

## 3. Clustering/Grouping

### Algorithm Selection
- **K-Means clustering**
  - Good for spherical clusters
  - Need to determine optimal K
  - Fast and scalable

- **Hierarchical clustering**
  - Creates cluster dendrograms
  - No need to pre-specify cluster count
  - Good for exploring cluster relationships

- **DBSCAN**
  - Handles noise and outliers well
  - Finds clusters of varying densities
  - Automatic cluster count determination

### Clustering Pipeline
- **Preprocessing embeddings**
  - Normalization (L2 norm)
  - Dimensionality reduction (PCA/UMAP) if needed
  - Distance metric selection (cosine, euclidean)

- **Hyperparameter optimization**
  - Elbow method for K-means
  - Silhouette analysis
  - Grid search for DBSCAN parameters

### Validation & Quality Assessment
- **Internal metrics**
  - Silhouette score
  - Calinski-Harabasz index
  - Davies-Bouldin index

- **External validation**
  - Manual inspection of clusters
  - Coherence with known categories/tags
  - Cross-validation with different algorithms

## 4. Analysis & Insights

### Cluster Characterization
- **Generate cluster summaries**
  - Most representative posts per cluster
  - Common keywords/themes
  - Average cluster size and coherence
  - Temporal patterns in clusters

### Topic Modeling Integration
- **Enhance clusters with topic information**
  - LDA or BERTopic for theme extraction
  - Keyword extraction per cluster
  - Topic evolution over time

### Similarity Analysis
- **Inter-cluster relationships**
  - Cluster similarity matrix
  - Hierarchical cluster visualization
  - Bridge posts between clusters

## 5. Output & Results

### Export Formats
- **Structured data outputs**
  - CSV with post ID, cluster label, confidence scores
  - JSON with full metadata and clustering results
  - Database format for integration with existing systems

### Cluster Reports
- **Detailed cluster analysis**
  - Cluster size distribution
  - Representative posts per cluster
  - Cluster coherence metrics
  - Temporal distribution of posts per cluster

### Visualizations
- **2D/3D embeddings visualization**
  - t-SNE or UMAP projections
  - Interactive plots with hover information
  - Color-coded clusters

- **Cluster analysis charts**
  - Cluster size histograms
  - Silhouette plots
  - Dendrogram for hierarchical clustering

### Interactive Results
- **Web dashboard or notebook**
  - Searchable cluster results
  - Post similarity exploration
  - Cluster navigation interface

## Technical Implementation Notes

### Dependencies
```python
# Core libraries
pandas, numpy, scikit-learn
# NLP and embeddings
sentence-transformers, openai, transformers
# Clustering and visualization
umap-learn, matplotlib, seaborn, plotly
# Web scraping and HTML parsing
beautifulsoup4, lxml, html2text
```

### Performance Considerations
- **Memory management** for large datasets
- **Batch processing** for embedding generation
- **Incremental clustering** for new posts
- **Caching strategies** for expensive operations

### Evaluation Criteria
- **Cluster quality metrics**
- **Processing time benchmarks**
- **Memory usage optimization**
- **Scalability testing**

## Next Steps

### CURRENT PRIORITY: Phase 2 - Embedding Generation 🔄
1. **Model Selection** - Choose embedding model optimized for multilingual academic content
2. **Embedding Pipeline** - Process 539 quality posts with chosen model
3. **Vector Storage** - Store embeddings with metadata for clustering analysis
4. **Quality Validation** - Verify embedding quality with sample similarity tests

### SUBSEQUENT PHASES:
4. **Clustering Implementation** - Apply multiple algorithms (K-means, hierarchical, DBSCAN)
5. **Cluster Analysis** - Generate themes, topics, and content insights
6. **Visualization** - Create interactive cluster visualizations and analysis dashboard
7. **Academic Applications** - Develop tools for content discovery and thematic analysis

### TECHNICAL ASSETS AVAILABLE:
- **Clean Dataset**: 539 posts, 1.25M words, validated and processed
- **Extraction Pipeline**: `html_extractor.py` - reusable for new content
- **Validation Suite**: `validate_extraction.py` - quality assurance tools
- **Analysis Tools**: `analyze_extracted_data.py` - dataset exploration
- **Documentation**: Complete extraction methodology and results

### DATA LOCATION: 
- **Primary Dataset**: `/workspaces/blog-post-embeddings-clustering/processed_data/extracted_posts.csv`
- **Individual Posts**: `/workspaces/blog-post-embeddings-clustering/processed_data/text_files/`
- **Quality Reports**: `/workspaces/blog-post-embeddings-clustering/processed_data/validation_results.json`

**STATUS**: Phase 1 Complete ✅ | Ready for Phase 2: Embedding Generation 🔄
