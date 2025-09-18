import abc
from typing import Callable, Literal, Optional, Dict, Any, List

from pathlib import Path

from ..db import PostgresDB
from ..utils import CachedBM25
from ..model.embed import PreTrainedEmbedder

class BaseRanker(abc.ABC):
    @abc.abstractmethod
    def rank(self, query: str, top_k: int = 5) -> list:
        pass
    
    @abc.abstractmethod
    def add_document(self, document_id: str, document: str) -> None:
        pass
    
    def add_documents(self, document_ids: List[str], documents: List[str]) -> None:
        for doc_id, doc in zip(document_ids, documents):
            self.add_document(doc_id, doc)

class PgRanker(BaseRanker):
    def __init__(self,
                 db: PostgresDB,
                 embedding_fn: PreTrainedEmbedder,
                 table: str,
                 similarity_metric: Literal["cosine", "euclidean", "inner_product"] = "cosine"):
        self.db = db
        self.table = table
        self.similarity_metric = similarity_metric
        if similarity_metric not in ["cosine", "euclidean", "inner_product"]:
            raise ValueError(f"Unsupported similarity metric: {similarity_metric}")
        
    def rank(self, query: str, top_k: int = 5) -> list:
        query_embedding = self.embedding_fn(query).detach().cpu().numpy().tolist()
        
        results = self.db.vector_search(
            table=self.table,
            query_embedding=query_embedding,
            top_k=top_k,
            metric=self.similarity_metric
        )
        return results
    
    def add_document(self, document_id: str, document: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        embedding = self.embedding_fn(document).detach().cpu().numpy().tolist()
        self.db.upsert_embedding(
            table=self.table,
            record_id=document_id,
            embedding=embedding,
            metadata=None
        )

class BM25Ranker(BaseRanker):
    def __init__(self,
                 corpus_loader: Callable[[], list[str]],
                 corpus_adder: Callable[[List[str]], None],
                 bm25_instance: Optional[CachedBM25] = None,
                 bm25_cache_path: Optional[str] = None
                 ):
        self._loader = corpus_loader
        self._adder = corpus_adder
        if bm25_instance is None and bm25_cache_path is None:
            raise ValueError("Either bm25_instance or bm25_cache_path must be provided.")
        elif bm25_instance is None:
            try:
                bm25_instance = CachedBM25.load(Path(bm25_cache_path))
            except FileNotFoundError:
                corpus = self._loader()
                tokenized_corpus = [self._tokenize(doc) for doc in corpus]
                bm25_instance = CachedBM25(Path(bm25_cache_path), tokenized_corpus)
        self.bm25 = bm25_instance
    
    def _tokenize(self, text: str) -> list[str]:
        return text.split()  # TODO: add tokenizer

    def rank(self, query: str, top_k: int = 5) -> list:
        results = self.bm25.get_scores(self._tokenize(query))
        return results
    
    def add_document(self, document_id: str, document: str) -> None:
        return self.add_documents([document_id], [document])
    
    def add_documents(self, document_ids: List[str], documents: List[str]) -> None:
        self._adder(documents)
        new_corpus = self._loader()
        tokenized_corpus = [self._tokenize(doc) for doc in new_corpus]

        self.bm25 = CachedBM25(self.bm25._cache_path, corpus=tokenized_corpus) # TODO: pass other bm25-specific hyperparams

class HybridRanker(BaseRanker):
    def __init__(self,
                 pg_ranker: PgRanker,
                 bm25_ranker: BM25Ranker,
                 alpha: float = 0.5):
        if not (0 <= alpha <= 1):
            raise ValueError("Alpha must be between 0 and 1.")
        self.pg_ranker = pg_ranker
        self.bm25_ranker = bm25_ranker
        self.alpha = alpha
    
    def rank(self, query: str, top_k: int = 5) -> list:
        pg_results = self.pg_ranker.rank(query, top_k=top_k*2)  # get more results to merge
        bm25_results = self.bm25_ranker.rank(query, top_k=top_k*2)
        
        # Merge results based on scores
        combined_scores = {}
        for doc_id, score in pg_results:
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + self.alpha * score
        for doc_id, score in bm25_results:
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + (1 - self.alpha) * score
        
        # Sort and return top_k
        ranked_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return ranked_docs
    
    def add_document(self, document_id: str, document: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.pg_ranker.add_document(document_id, document, metadata)
        self.bm25_ranker.add_document(document)
    
    def add_documents(self, document_ids: List[str], documents: List[str]) -> None:
        self.bm25_ranker.add_documents(documents)
        for i, doc in enumerate(documents):
            self.pg_ranker.add_document(document_ids[i], doc, None)

# TODO: rerankers