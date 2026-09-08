from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class EmbeddingIndexer:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        """
        Initializes the multilingual dense embedding model.
        The fallback model is used to avoid environment OOM/download timeout issues.
        """
        self.model_name = model_name
        # Prefer an already cached model. This keeps normal API requests from
        # making a metadata call to Hugging Face and still allows first-time
        # setup to download the model when it is genuinely absent.
        try:
            self.model = SentenceTransformer(model_name, local_files_only=True)
        except OSError:
            self.model = SentenceTransformer(model_name)
        
    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """
        Encodes a list of document strings into dense vectors.
        """
        # For mpnet-base-v2, no special prefixes are needed.
        # It's good practice to ensure they are strings and not empty.
        docs = [str(d).strip() if str(d).strip() else " " for d in documents]
        embeddings = self.model.encode(docs, show_progress_bar=True, convert_to_numpy=True)
        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encodes a single query string into a dense vector.
        """
        query = str(query).strip() if str(query).strip() else " "
        embedding = self.model.encode(query, convert_to_numpy=True)
        return embedding.astype(np.float32)
