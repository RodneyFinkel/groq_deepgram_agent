CHUNK_SIZE_INGEST = 1000
CHUNK_OVERLAP_INGEST = 200
CHUNK_SIZE_LLM = 2000
CHUNK_OVERLAP_LLM = 0

SEMANTIC_SIMILARITY_THRESHOLD = 0.6  # Default threshold for grouping sentences (0-1; lower = more chunks)
CHUNKING_TYPE = 'semantic'  # 'semantic' or 'fixed'