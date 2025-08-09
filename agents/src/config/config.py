import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    OUTPUT_DIR = "output"
    THEOREMS_PATH = os.path.join("data", "easy_20.json")
    CONTEXT_LEARNING_PATH = "data/context_learning"
    CHROMA_DB_PATH = "data/rag/chroma_db"
    MANIM_DOCS_PATH = "data/rag/manim_docs"
    EMBEDDING_MODEL = "hf:ibm-granite/granite-embedding-30m-english"
    
    # Kokoro TTS configurations
    KOKORO_MODEL_PATH = os.getenv('KOKORO_MODEL_PATH')
    KOKORO_VOICES_PATH = os.getenv('KOKORO_VOICES_PATH')
    KOKORO_DEFAULT_VOICE = os.getenv('KOKORO_DEFAULT_VOICE')
    KOKORO_DEFAULT_SPEED = float(os.getenv('KOKORO_DEFAULT_SPEED', '1.0'))
    KOKORO_DEFAULT_LANG = os.getenv('KOKORO_DEFAULT_LANG') 

    # AWS/S3 configuration (shared with backend)
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_SESSION_TOKEN = os.getenv('AWS_SESSION_TOKEN')
    AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')

    # Primary outputs bucket and optional base prefix (e.g., "projects/")
    S3_BUCKET = os.getenv('S3_BUCKET')
    S3_BASE_PREFIX = os.getenv('S3_BASE_PREFIX', '')

    # Optional public base URL (e.g., CloudFront distribution). If set, URLs will use this base
    S3_PUBLIC_BASE_URL = os.getenv('S3_PUBLIC_BASE_URL')

    # Optional KMS key for server-side encryption
    S3_KMS_KEY_ID = os.getenv('S3_KMS_KEY_ID')

    # Transfer tuning and behavior
    S3_UPLOAD_ON_WRITE = os.getenv('S3_UPLOAD_ON_WRITE', 'false').lower() == 'true'
    S3_MAX_CONCURRENCY = int(os.getenv('S3_MAX_CONCURRENCY', '8'))
    S3_MULTIPART_THRESHOLD_MB = int(os.getenv('S3_MULTIPART_THRESHOLD_MB', '64'))

    # Optional custom endpoint (e.g., MinIO) and path-style toggle
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')
    S3_FORCE_PATH_STYLE = os.getenv('S3_FORCE_PATH_STYLE', 'false').lower() == 'true'

    # Default presigned URL expiration (seconds)
    S3_PRESIGN_EXPIRATION = int(os.getenv('S3_PRESIGN_EXPIRATION', '3600'))