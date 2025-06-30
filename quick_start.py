#!/usr/bin/env python3
"""
Quick Start - Talmudic Blog Clustering Explorer
==============================================

Quick script to explore the clustering results and visualizations.
"""

import sys
from pathlib import Path
import pandas as pd
import webbrowser
import subprocess


def main():
    """Quick start exploration of clustering results"""
    
    print("🎓 Talmudic Blog Clustering - Quick Start Explorer")
    print("=" * 55)
    
    # Check if results exist
    if not check_results_exist():
        print("❌ Clustering results not found. Please run Phase 2 first.")
        print("   Command: python run_phase2.py")
        return
    
    print("✅ Clustering results found!")
    
    # Show quick statistics
    show_quick_stats()
    
    # Interactive menu
    while True:
        print("\n📊 What would you like to explore?")
        print("1. View cluster assignments")
        print("2. Open interactive visualizations")
        print("3. Show clustering statistics")
        print("4. List all available plots")
        print("5. Open project documentation")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            view_cluster_assignments()
        elif choice == "2":
            open_interactive_visualizations()
        elif choice == "3":
            show_clustering_statistics()
        elif choice == "4":
            list_available_plots()
        elif choice == "5":
            open_documentation()
        elif choice == "6":
            print("👋 Happy exploring your Talmudic clusters!")
            break
        else:
            print("❌ Invalid choice. Please try again.")


def check_results_exist():
    """Check if clustering results exist"""
    required_files = [
        "processed_data/cluster_labels.csv",
        "processed_data/clustering_results.json",
        "processed_data/blog_embeddings.npy"
    ]
    
    return all(Path(f).exists() for f in required_files)


def show_quick_stats():
    """Show quick clustering statistics"""
    try:
        df = pd.read_csv("processed_data/cluster_labels.csv")
        print(f"\n📈 Quick Statistics:")
        print(f"   • Total posts: {len(df)}")
        print(f"   • K-means clusters: {df['kmeans_cluster'].nunique()}")
        print(f"   • Largest cluster: {df['kmeans_cluster'].value_counts().max()} posts")
        print(f"   • Average cluster size: {len(df) / df['kmeans_cluster'].nunique():.1f} posts")
    except Exception as e:
        print(f"   Error loading statistics: {e}")


def view_cluster_assignments():
    """View sample cluster assignments"""
    try:
        df = pd.read_csv("processed_data/cluster_labels.csv")
        
        print("\n📋 Sample Cluster Assignments:")
        print("-" * 80)
        
        # Show a few examples from different clusters
        for cluster_id in sorted(df['kmeans_cluster'].unique())[:5]:
            cluster_posts = df[df['kmeans_cluster'] == cluster_id]
            print(f"\n🔹 Cluster {cluster_id} ({len(cluster_posts)} posts):")
            
            for _, post in cluster_posts.head(2).iterrows():
                title = post['title'][:60] + "..." if len(post['title']) > 60 else post['title']
                print(f"   • {title}")
        
        print(f"\n... and {df['kmeans_cluster'].nunique() - 5} more clusters")
        
    except Exception as e:
        print(f"❌ Error loading cluster assignments: {e}")


def open_interactive_visualizations():
    """Open interactive HTML visualizations"""
    plots_dir = Path("processed_data/plots")
    
    if not plots_dir.exists():
        print("❌ Plots directory not found.")
        return
    
    html_files = list(plots_dir.glob("*.html"))
    
    if not html_files:
        print("❌ No interactive visualizations found.")
        return
    
    print(f"\n🎨 Found {len(html_files)} interactive visualizations:")
    for i, html_file in enumerate(html_files, 1):
        print(f"   {i}. {html_file.name}")
    
    try:
        choice = int(input(f"\nWhich visualization would you like to open? (1-{len(html_files)}): "))
        
        if 1 <= choice <= len(html_files):
            selected_file = html_files[choice - 1]
            file_path = selected_file.absolute()
            webbrowser.open(f"file://{file_path}")
            print(f"🌐 Opening {selected_file.name} in your browser...")
        else:
            print("❌ Invalid choice.")
            
    except ValueError:
        print("❌ Please enter a valid number.")
    except Exception as e:
        print(f"❌ Error opening visualization: {e}")


def show_clustering_statistics():
    """Show detailed clustering statistics"""
    try:
        import json
        
        with open("processed_data/clustering_results.json", "r") as f:
            results = json.load(f)
        
        print("\n📊 Detailed Clustering Statistics:")
        print("=" * 50)
        
        # K-means results
        if "kmeans" in results:
            kmeans = results["kmeans"]
            print(f"🔹 K-means Clustering:")
            print(f"   • Clusters: {kmeans['n_clusters']}")
            print(f"   • Silhouette Score: {kmeans['silhouette_score']:.4f}")
            print(f"   • Calinski-Harabasz: {kmeans['calinski_harabasz_score']:.2f}")
            print(f"   • Davies-Bouldin: {kmeans['davies_bouldin_score']:.4f}")
        
        # Optimization results
        if "kmeans_optimization" in results:
            opt = results["kmeans_optimization"]
            optimal = opt["optimal_clusters"]
            print(f"\n🔹 Optimization Results:")
            print(f"   • Elbow method optimal: {optimal['elbow']} clusters")
            print(f"   • Silhouette optimal: {optimal['silhouette']} clusters")
            print(f"   • Calinski-Harabasz optimal: {optimal['calinski_harabasz']} clusters")
        
    except Exception as e:
        print(f"❌ Error loading clustering statistics: {e}")


def list_available_plots():
    """List all available plots"""
    plots_dir = Path("processed_data/plots")
    
    if not plots_dir.exists():
        print("❌ Plots directory not found.")
        return
    
    plot_files = list(plots_dir.glob("*"))
    
    if not plot_files:
        print("❌ No plots found.")
        return
    
    print(f"\n🎨 Available Visualizations ({len(plot_files)} files):")
    print("-" * 60)
    
    # Categorize plots
    categories = {
        "Interactive": [f for f in plot_files if f.suffix == ".html"],
        "Scatter Plots": [f for f in plot_files if "scatter" in f.name and f.suffix == ".png"],
        "Optimization": [f for f in plot_files if "optimization" in f.name],
        "Analysis": [f for f in plot_files if f.suffix == ".png" and "scatter" not in f.name and "optimization" not in f.name]
    }
    
    for category, files in categories.items():
        if files:
            print(f"\n🔹 {category}:")
            for f in sorted(files):
                print(f"   • {f.name}")


def open_documentation():
    """Open project documentation"""
    docs = [
        ("Project Outline", "claude_project_outline.md"),
        ("Phase 2 README", "PHASE2_README.md"),
        ("Main README", "README.md")
    ]
    
    print("\n📚 Available Documentation:")
    for i, (name, filename) in enumerate(docs, 1):
        if Path(filename).exists():
            print(f"   {i}. {name}")
    
    print("\n💡 Tip: Open these files in your text editor or markdown viewer.")


if __name__ == "__main__":
    main()
