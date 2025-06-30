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
