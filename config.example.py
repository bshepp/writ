"""
Configuration template for Writ projects.

Rename to config.py and customize for your domain.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
RESOURCES_DIR = PROJECT_ROOT / "resources"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"


def ensure_dirs():
    """Create project directories if they don't exist."""
    DATA_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)


# Database
NEO4J_URI = os.getenv("NEO4J_URI", None)
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", None)

# Document processing
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# AI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NLP_MODE = os.getenv("NLP_MODE", "auto").lower()  # auto | openai | langchain | basic

# CORS / API
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")
DEBUG = os.getenv("DEBUG", "true").lower() == "true"

# ---------------------------------------------------------------------------
# DOMAIN-SPECIFIC: Customize entity and relationship types for your regulation
# ---------------------------------------------------------------------------
# See schema.example.yaml for full schema; these are used by ingestion/query.

ENTITY_TYPES = [
    "Regulation",
    "Requirement",
    "Procedure",
    "Personnel",
    "Control",
    "Timeline",
    "Document",
    "Location",
]

RELATIONSHIP_TYPES = [
    "REQUIRES",
    "APPLIES_TO",
    "IMPLEMENTS",
    "HAS_DEADLINE",
    "MENTIONED_IN",
    "REFERENCES",
    "PERFORMED_BY",
    "SUPERSEDES",
]
