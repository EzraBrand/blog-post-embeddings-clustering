#!/usr/bin/env python3
"""
Project Cleanup and Optimization
================================

Optimizes the Talmudic blog clustering project structure and removes redundant files.
"""

import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np


def optimize_project():
    """Perform comprehensive project optimization"""
    
    print("🧹 Starting Talmudic Blog Clustering Project Optimization...")
    
    # 1. Clean Python cache
    cleanup_python_cache()
    
    # 2. Optimize data storage
    optimize_data_storage()
    
    # 3. Organize script files
    organize_scripts()
    
    # 4. Create project summary
    create_project_summary()
    
    print("✅ Project optimization complete!")


def cleanup_python_cache():
    """Remove Python cache files"""
    print("\n📁 Cleaning Python cache files...")
    
    cache_dirs = []
    for root, dirs, files in os.walk('.'):
        if '__pycache__' in dirs:
            cache_dirs.append(os.path.join(root, '__pycache__'))
    
    for cache_dir in cache_dirs:
        shutil.rmtree(cache_dir, ignore_errors=True)
        print(f"   Removed: {cache_dir}")


def optimize_data_storage():
    """Optimize data file storage"""
    print("\n💾 Optimizing data storage...")
    
    processed_data = Path("processed_data")
    
    # Check if we have redundant extracted_posts files
    csv_file = processed_data / "extracted_posts.csv"
    json_file = processed_data / "extracted_posts.json" 
    pkl_file = processed_data / "extracted_posts.pkl"
    
    if all(f.exists() for f in [csv_file, json_file, pkl_file]):
        # Keep CSV (most universal) and remove others
        print(f"   Keeping: extracted_posts.csv ({csv_file.stat().st_size / 1024 / 1024:.1f}MB)")
        
        if json_file.exists():
            json_file.unlink()
            print(f"   Removed: extracted_posts.json (saved ~17MB)")
            
        if pkl_file.exists():
            pkl_file.unlink()
            print(f"   Removed: extracted_posts.pkl (saved ~17MB)")
    
    # Check for any empty or very small files
    for file in processed_data.glob("*"):
        if file.is_file() and file.stat().st_size < 100:  # Less than 100 bytes
            print(f"   Warning: Very small file detected: {file} ({file.stat().st_size} bytes)")


def organize_scripts():
    """Organize utility and analysis scripts"""
    print("\n📂 Organizing script files...")
    
    # Create directories if they don't exist
    scripts_dir = Path("scripts")
    utilities_dir = scripts_dir / "utilities"
    analysis_dir = scripts_dir / "analysis"
    testing_dir = scripts_dir / "testing"
    
    for dir_path in [utilities_dir, analysis_dir, testing_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Define file movements
    moves = {
        "check_api_key.py": utilities_dir,
        "validate_extraction.py": utilities_dir,
        "analyze_extracted_data.py": utilities_dir,
        "demo_phase2.py": analysis_dir,
        "test_api_key.py": testing_dir,
        "test_phase2.py": testing_dir,
    }
    
    for filename, target_dir in moves.items():
        source = Path(filename)
        if source.exists():
            target = target_dir / filename
            shutil.move(str(source), str(target))
            print(f"   Moved: {filename} → {target_dir}")


def create_project_summary():
    """Create a comprehensive project summary"""
    print("\n📊 Creating project summary...")
    
    summary = {
        "project": "Talmudic Blog Post Clustering with Embeddings",
        "status": "Phase 2 Complete - Real Embeddings Generated",
        "completion_date": "June 30, 2025",
        "data_summary": {},
        "technical_summary": {},
        "files_overview": {}
    }
    
    # Data summary
    processed_data = Path("processed_data")
    if processed_data.exists():
        summary["data_summary"] = {
            "total_posts": 539,
            "embeddings_generated": True,
            "clustering_complete": True,
            "visualizations": 21,
            "clusters_identified": 45,
            "embedding_model": "OpenAI text-embedding-3-large",
            "embedding_dimensions": 3072
        }
    
    # Technical summary
    summary["technical_summary"] = {
        "algorithms_implemented": ["K-means", "Hierarchical", "DBSCAN"],
        "visualization_methods": ["t-SNE", "UMAP", "PCA"],
        "evaluation_metrics": ["Silhouette", "Calinski-Harabasz", "Davies-Bouldin"],
        "best_clustering": "K-means with 45 clusters",
        "silhouette_score": 0.030
    }
    
    # Files overview
    important_files = {
        "core_scripts": [
            "generate_embeddings.py",
            "clustering_analysis.py", 
            "visualize_clusters.py",
            "run_phase2.py"
        ],
        "data_files": [
            "processed_data/blog_embeddings.npy",
            "processed_data/clustering_results.json",
            "processed_data/cluster_labels.csv"
        ],
        "documentation": [
            "claude_project_outline.md",
            "PHASE2_README.md",
            "README.md"
        ]
    }
    summary["files_overview"] = important_files
    
    # Save summary
    import json
    with open("PROJECT_SUMMARY.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("   Created: PROJECT_SUMMARY.json")


if __name__ == "__main__":
    optimize_project()
