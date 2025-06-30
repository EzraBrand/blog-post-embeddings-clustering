#!/usr/bin/env python3
"""
Test OpenAI API Key Configuration
"""

import os
import sys
from pathlib import Path

# Load environment from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded .env file")
except ImportError:
    print("⚠️  python-dotenv not available, using manual loading")
    # Manual loading
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
        print("✅ Manually loaded .env file")

# Test API key
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("❌ No OPENAI_API_KEY found in environment")
    sys.exit(1)

print(f"✅ API Key loaded: {api_key[:15]}...{api_key[-15:]}")

# Test OpenAI connection
try:
    from openai import OpenAI
    
    client = OpenAI(api_key=api_key)
    
    # Test with a simple API call
    models = client.models.list()
    print(f"✅ OpenAI API connection successful!")
    print(f"📊 Available models: {len(models.data)}")
    
    # Check if our target embedding model is available
    embedding_models = [m for m in models.data if 'embedding' in m.id.lower()]
    print(f"🔍 Embedding models available: {len(embedding_models)}")
    
    target_model = "text-embedding-3-large"
    if any(m.id == target_model for m in models.data):
        print(f"✅ Target model '{target_model}' is available")
    else:
        print(f"⚠️  Target model '{target_model}' not found")
        print("Available embedding models:", [m.id for m in embedding_models])

except Exception as e:
    print(f"❌ OpenAI API test failed: {e}")
    sys.exit(1)

print("\n🎉 OpenAI API configuration is working correctly!")
print("You can now run the embedding generation with:")
print("python generate_embeddings.py --chunk-long-posts")
