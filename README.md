# Blog Post Embeddings & Clustering Analysis

> Comprehensive semantic analysis and clustering of 539 scholarly blog posts using OpenAI embeddings and advanced machine learning techniques.

## 🎯 Project Overview

This project implements a complete pipeline for analyzing and clustering scholarly blog posts from academic content. Using state-of-the-art embedding models and multiple clustering algorithms, it identifies thematic patterns and relationships within academic discussions on Talmudic studies, Jewish scholarship, and biblical commentary.

## ✅ Current Status: Phase 2 Complete

- **539 blog posts** successfully processed and analyzed
- **1.25 million words** of scholarly content extracted and clustered
- **Production-ready pipeline** with OpenAI integration
- **21 comprehensive visualizations** generated
- **Multi-algorithm clustering** implemented and tested

## 🚀 Quick Start

```bash
# Generate real embeddings with OpenAI API
python run_phase2.py --chunk-long-posts

# Or run with mock embeddings for testing
python run_phase2.py --mock-embeddings
```

## 📋 Key Features

- **Smart Text Processing**: Handles Hebrew/English mixed content with intelligent chunking
- **Multiple Clustering Algorithms**: K-means, Hierarchical, and DBSCAN with optimization
- **Advanced Visualizations**: t-SNE, UMAP, PCA plots plus interactive analysis
- **Production Infrastructure**: Comprehensive error handling, logging, and testing
- **Academic Focus**: Specialized for scholarly content analysis and topic modeling

## 📁 Project Structure

```
├── generate_embeddings.py      # OpenAI embedding generation
├── clustering_analysis.py      # Multi-algorithm clustering
├── visualize_clusters.py      # Comprehensive visualization suite
├── run_phase2.py              # End-to-end pipeline
├── config.py                  # Configuration management
├── processed_data/            # Generated data and results
│   ├── blog_embeddings.npy   # 3072-dimensional embeddings
│   ├── clustering_results.json # Complete analysis results
│   └── plots/                # 21 visualization files
└── PHASE2_README.md          # Detailed technical documentation
```

## 📊 Results & Analysis

The pipeline generates comprehensive analysis including:
- **Semantic embeddings** for all 539 posts using OpenAI's text-embedding-3-large
- **Optimized clustering** with parameter tuning and evaluation metrics
- **Interactive visualizations** for cluster exploration and analysis
- **Content analysis** with keyword extraction and topic identification

## 🔗 Documentation

- **[PHASE2_README.md](PHASE2_README.md)** - Complete technical documentation
- **[claude_project_outline.md](claude_project_outline.md)** - Full project roadmap and status

---

**Status**: Phase 2 Complete ✅ | Ready for Production Use 🚀
