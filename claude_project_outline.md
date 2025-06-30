# Project Outline: Blog Post Clustering with Embeddings

## Current Status (June 30, 2025)
**Data Available**: The blog.zip archive has been extracted and examined. Contains:
- **553 blog posts** in HTML format (posts.csv shows metadata)
- **Focus**: Scholarly content on Jewish studies, Talmudic analysis, biblical commentary, and research methodology
- **Content Range**: From personal genealogy research to comprehensive academic guides (40+ page resources)
- **Format**: Rich HTML with embedded links, citations, footnotes, YouTube embeds, and PDF attachments
- **Quality**: High-quality academic content with extensive references and scholarly approach

**Files Examined**:
- `posts.csv`: Metadata for all 553 posts (titles, dates, IDs)
- Sample HTML files showing diverse but coherent academic content
- Email delivery/engagement CSV files (can be ignored for clustering)

**Ready for Next Steps**: Data preprocessing and HTML content extraction

## Overview
This project focuses on automatically grouping blog posts using text embeddings and clustering algorithms. The goal is to identify thematically similar content and organize posts into meaningful clusters for better content discovery and analysis.

## 1. Data Preprocessing

### HTML Content Extraction
- **Parse HTML files** to extract clean text content
  - Strip HTML tags while preserving text structure
  - Remove script and style elements
  - Handle special HTML entities and encoding
  - Preserve meaningful whitespace and paragraph breaks

### Text Cleaning & Preprocessing
- **Content normalization**
  - Remove or standardize special characters
  - Handle different encodings (UTF-8, ASCII, etc.)
  - Normalize whitespace and line breaks
  - Convert to lowercase (optional, depending on model)

- **Text preprocessing pipeline**
  - Remove stopwords (language-specific)
  - Handle punctuation appropriately
  - Consider stemming/lemmatization
  - Filter out very short or empty posts

### Metadata Extraction
- **Extract structured information**
  - Post title (from `<title>` tag or `<h1>`)
  - Publication date (from meta tags or content)
  - Author information
  - Tags/categories (from meta tags or content)
  - URL structure analysis for additional context

### Dataset Structure
- **Create unified data format**
  - Unique post ID (hash or sequential)
  - Clean text content (title + body)
  - Extracted metadata
  - Original file path/reference
  - Content length and basic statistics

## 2. Embedding Generation

### Model Selection
- **Primary options to consider**
  - **OpenAI text-embedding-ada-002**: High quality, API-based
  - **Sentence Transformers**: Local models (all-MiniLM-L6-v2, all-mpnet-base-v2)
  - **Local alternatives**: BGE, E5, or custom fine-tuned models

### Embedding Strategy
- **Content combination approaches**
  - Title-only embeddings
  - Body-only embeddings
  - Weighted combination (title weight: 0.3, body weight: 0.7)
  - Separate embeddings with concatenation

### Implementation Considerations
- **Batch processing** for efficiency
- **Rate limiting** for API-based models
- **Caching** to avoid re-computation
- **Dimensionality** consideration (512, 768, 1536 dimensions)

### Storage & Management
- **Vector storage options**
  - Local storage (NumPy arrays, HDF5, Parquet)
  - Vector databases (Chroma, Pinecone, Weaviate)
  - Simple JSON/pickle for small datasets

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
**IMMEDIATE PRIORITY**: 
1. **HTML Content Extraction** - Parse the 553 HTML files in `/posts/` directory to extract clean text
2. **Metadata Integration** - Combine content with metadata from `posts.csv` 
3. **Content Analysis** - Given the academic Jewish studies focus, consider domain-specific preprocessing

**Subsequent Steps**:
4. Choose and configure embedding model (consider models trained on academic text)
5. Develop clustering pipeline with multiple algorithms
6. Create evaluation and visualization framework
7. Generate comprehensive results and reports

**Data Location**: All blog post HTML files are in `/workspaces/blog-post-embeddings-clustering/posts/`
