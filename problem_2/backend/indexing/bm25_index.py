import re
from typing import List
from rank_bm25 import BM25Okapi

class HinglishTokenizer:
    def __init__(self):
        # A basic set of English stopwords that also commonly appear in Hinglish chat.
        self.stop_words = {
            "a", "an", "and", "the", "in", "on", "at", "to", "for", "of",
            "is", "are", "was", "were", "it", "this", "that", "with", "from"
        }

    def tokenize(self, text: str) -> List[str]:
        # Lowercase
        text = str(text).lower()
        # Preserve useful alphanumeric tokens (including numbers)
        # We replace non-alphanumeric with spaces, so punctuation is removed
        # but words and numbers stay intact.
        text = re.sub(r'[^\w\s]', ' ', text)
        
        tokens = text.split()
        # Filter out purely stop words (do not destructively stem!)
        tokens = [t for t in tokens if t not in self.stop_words]
        
        # If filtering removed everything, just return the raw split to avoid empty documents
        if not tokens and text.split():
            return text.split()
            
        return tokens

class BM25Indexer:
    def __init__(self):
        self.tokenizer = HinglishTokenizer()
        self.bm25 = None
        self.tokenized_corpus = []

    def build(self, documents: List[str]):
        """
        Builds the BM25 index over a list of document strings.
        """
        self.tokenized_corpus = [self.tokenizer.tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def get_scores(self, query: str) -> List[float]:
        """
        Scores the query against the built BM25 index.
        """
        if self.bm25 is None:
            raise ValueError("BM25 index has not been built or loaded yet.")
            
        tokenized_query = self.tokenizer.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        return scores.tolist()
