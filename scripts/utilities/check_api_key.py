#!/usr/bin/env python3
"""
Quick API Key Validation
"""

import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
if api_key and api_key.startswith('sk-'):
    print(f"✅ OpenAI API key configured: {api_key[:15]}...{api_key[-15:]}")
    print("✅ Key format appears valid (starts with 'sk-')")
    print("\n🚀 Ready to run Phase 2 with real embeddings!")
    print("\nNext steps:")
    print("1. python generate_embeddings.py --chunk-long-posts")
    print("2. python clustering_analysis.py") 
    print("3. python visualize_clusters.py")
    print("\nOr run the complete pipeline:")
    print("python run_phase2.py --chunk-long-posts")
else:
    print("❌ Invalid or missing OpenAI API key")
