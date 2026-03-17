"""Configuration for the literature domain."""

# Knowledge cutoff: only papers published before this date
CUTOFF_DATE = "2023-12-31"

# arXiv categories relevant to ML architecture research
CATEGORIES = ["cs.LG", "cs.CL", "cs.AI", "cs.NE", "stat.ML"]

# Embedding model for paper similarity search (trained on scientific papers)
EMBEDDING_MODEL = "allenai-specter"

# Chroma vector DB path
CHROMA_PATH = "./data/chroma_db"

# Default number of results for similarity search
DEFAULT_QUERY_RESULTS = 10

# Collection name in Chroma
COLLECTION_NAME = "arxiv_papers"
