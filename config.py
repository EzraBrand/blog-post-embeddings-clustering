# HTML Content Extraction Configuration
# ====================================

# Directory and file paths
POSTS_DIR = "posts"
METADATA_FILE = "posts.csv" 
OUTPUT_DIR = "processed_data"

# Content extraction settings
MIN_CONTENT_LENGTH = 50  # Minimum characters for valid body text
MIN_WORD_COUNT = 10      # Minimum word count for quality content
QUALITY_WORD_THRESHOLD = 100  # Word count threshold for "quality" posts

# HTML processing settings
IGNORE_LINKS = False     # Whether to remove links from extracted text
IGNORE_IMAGES = True     # Whether to remove images
IGNORE_EMPHASIS = False  # Whether to preserve bold/italic formatting
BODY_WIDTH = 0          # Line wrapping (0 = no wrapping)

# Text cleaning settings
PRESERVE_FOOTNOTES = True    # Keep footnote markers like [1]
CLEAN_BULLET_POINTS = True   # Remove bullet point markers
REMOVE_ISOLATED_CHARS = True # Remove isolated special characters

# Logging settings
LOG_LEVEL = "INFO"
LOG_TO_FILE = True
LOG_TO_CONSOLE = True

# Output formats
SAVE_JSON = True
SAVE_CSV = True  
SAVE_PICKLE = True
SAVE_TEXT_FILES = True  # Individual text files for manual review

# Validation settings
VALIDATE_HEBREW_CONTENT = True  # Check for Hebrew text preservation
CHECK_METADATA_ALIGNMENT = True # Verify posts match metadata entries


# ====================================
# Phase 2: Embedding and Clustering Configuration
# ====================================

# Embedding settings
EMBEDDING_MODEL = "text-embedding-3-large"  # OpenAI model for high-quality embeddings
EMBEDDING_DIMENSIONS = 3072  # Dimensions for text-embedding-3-large
MAX_CHUNK_SIZE = 8000  # Maximum characters per chunk for very long posts
CHUNK_OVERLAP = 200    # Character overlap between chunks
EMBEDDING_BATCH_SIZE = 100  # Number of texts to process in one API call

# OpenAI API settings
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"  # Environment variable name for API key
OPENAI_MAX_RETRIES = 3
OPENAI_RETRY_DELAY = 1  # Seconds to wait between retries

# Clustering algorithms and parameters
CLUSTERING_ALGORITHMS = {
    'kmeans': {
        'min_clusters': 5,
        'max_clusters': 50,
        'step': 5
    },
    'hierarchical': {
        'min_clusters': 5,
        'max_clusters': 30,
        'linkage_methods': ['ward', 'complete', 'average']
    },
    'dbscan': {
        'eps_range': [0.1, 0.2, 0.3, 0.4, 0.5],
        'min_samples_range': [3, 5, 10, 15]
    }
}

# Optimal cluster number detection
ELBOW_METHOD = True
SILHOUETTE_ANALYSIS = True
CALINSKI_HARABASZ_SCORE = True

# Dimensionality reduction for visualization
REDUCTION_METHODS = ['tsne', 'umap', 'pca']
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
PCA_N_COMPONENTS = 50  # For initial dimensionality reduction before t-SNE/UMAP

# Visualization settings
FIGURE_SIZE = (12, 8)
DPI = 300
SAVE_PLOTS = True
PLOT_FORMAT = 'png'

# Analysis settings
GENERATE_CLUSTER_SUMMARIES = True
TOP_WORDS_PER_CLUSTER = 20
MIN_CLUSTER_SIZE = 3  # Minimum posts per cluster for analysis
TOPIC_MODELING = True  # Generate topic summaries for clusters
