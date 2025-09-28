import abc
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Literal, Optional, Dict, Any, List, Tuple
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import select

from pathlib import Path

from ..utils import CachedBM25
from ..model.embed import PreTrainedEmbedder
from ..model.prompts import Question
from ..db.model import Document, DocumentChunk, Candidate
from ..db.similarity import sim_

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
                 db: Session,
                 embedding_fn: PreTrainedEmbedder,
                 table: Literal["documents", "document_chunks"] = "documents",
                 similarity_metric: Literal["cosine", "euclidean", "inner_product"] = "cosine"):
        self.db = db
        self.embedding_fn = embedding_fn
        self.table = table
        self.similarity_metric = similarity_metric
        if similarity_metric not in ["cosine", "euclidean", "inner_product"]:
            raise ValueError(f"Unsupported similarity metric: {similarity_metric}")

    def rank(self, query: str, top_k: int = 5) -> List[Tuple[BaseModel, float]]:
        """
        Rank documents or chunks by vector similarity to query.

        Returns:
            List of tuples (document/chunk, similarity_score)
        """
        # Generate query embedding
        query_embedding = self.embedding_fn(query).detach().cpu().numpy().tolist()

        if self.table == "documents":
            model_cls = Document
            embedding_column = Document.embedding
        else:  # document_chunks
            model_cls = DocumentChunk
            embedding_column = DocumentChunk.embedding

        # Create similarity expression
        similarity_expr = sim_(embedding_column, query_embedding, metric=self.similarity_metric)

        # Query with similarity filtering and ordering
        stmt = (
            select(model_cls, similarity_expr.label('similarity_score'))
            .where(embedding_column.is_not(None))  # Only rows with embeddings
            .order_by(similarity_expr.asc())  # Lower distance = higher similarity
            .limit(top_k)
        )

        results = []
        for row in self.db.exec(stmt):
            document_or_chunk = row[0]  # The model instance
            score = row[1]  # The similarity score
            results.append((document_or_chunk, score))

        return results

    def add_document(self, document_id: str, document: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add or update document embedding in the database.

        This method generates embeddings and stores them directly in the Document/DocumentChunk table,
        eliminating the need for a separate embeddings table.
        """
        embedding = self.embedding_fn(document).detach().cpu().numpy().tolist()

        if self.table == "documents":
            # Update existing document or create new one
            stmt = select(Document).where(Document.id == document_id)
            existing_doc = self.db.exec(stmt).first()

            if existing_doc:
                existing_doc.embedding = embedding
                if not existing_doc.content:
                    existing_doc.content = document
                self.db.merge(existing_doc)
            else:
                new_doc = Document(id=document_id, content=document, embedding=embedding)
                self.db.add(new_doc)

        else:  # document_chunks
            # For chunks, we need document_id and chunk index
            # This assumes document_id format like "doc_id:chunk_idx"
            if ":" in document_id:
                doc_id, chunk_idx_str = document_id.rsplit(":", 1)
                chunk_idx = int(chunk_idx_str)
            else:
                raise ValueError("For chunks, document_id must be in format 'doc_id:chunk_idx'")

            stmt = select(DocumentChunk).where(
                DocumentChunk.document_id == doc_id,
                DocumentChunk.idx == chunk_idx
            )
            existing_chunk = self.db.exec(stmt).first()

            if existing_chunk:
                existing_chunk.embedding = embedding
                if not existing_chunk.text:
                    existing_chunk.text = document
                self.db.merge(existing_chunk)
            else:
                new_chunk = DocumentChunk(
                    document_id=doc_id,
                    idx=chunk_idx,
                    text=document,
                    embedding=embedding,
                    metadata=metadata
                )
                self.db.add(new_chunk)

        self.db.commit()

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
        self.bm25_ranker.add_document(document_id, document)

    def add_documents(self, document_ids: List[str], documents: List[str]) -> None:
        self.bm25_ranker.add_documents(documents)
        for i, doc in enumerate(documents):
            self.pg_ranker.add_document(document_ids[i], doc, None)

class BaseReranker(abc.ABC):
    @abc.abstractmethod
    def rerank(self, query: str, documents: List[str], top_k: int = -1) -> list:
        pass

class ScoringReranker(BaseReranker):
    def __init__(self, scoring_fn: Callable[[str, str], float],
                 parallelize: bool = False):
        self.scoring_fn = scoring_fn
        self.parallelize = parallelize

    def rerank(self, query: str, ids: List[str], documents: List[str], top_k: int = -1) -> list:
        assert len(documents) == len(ids), "IDs and documents must have the same length."
        if self.parallelize:
            with ProcessPoolExecutor() as executor:
                scores = list(executor.map(lambda did, doc: (did, self.scoring_fn(query, doc)), ids, documents))
        else:
            scores = [(did, self.scoring_fn(query, doc)) for did, doc in zip(ids, documents)]
        ranked_docs = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
        return ranked_docs

class QuestionnaireScorer(Callable):
    def __init__(
        self,
        questionnaire: List[Question],
        scoring_fn: Callable[[str, List[str]], float],
        score_weight_map: Dict[str, float],
        score_value_map: Dict[str, float],
    ):
        self.questionnaire = questionnaire
        self.scoring_fn = scoring_fn
        self.score_weight_map = score_weight_map
        self.score_value_map = score_value_map

    def __call__(self, query: str, document: str) -> float:
        """
        Scores a document based on the questionnaire.
        query is ignored.
        document is what is compared agains the questionnaire.
        """
        total_score = 0.0
        total_weight = 0.0
        scores = self.scoring_fn(document, [q.criterion for q in self.questionnaire])
        for question, score in zip(self.questionnaire, scores):
            weight = self.score_weight_map.get(question.importance, 1.0)
            value = self.score_value_map.get(score, 0.0)
            total_score += weight * value
            total_weight += weight

        if total_weight == 0:
            return 0.0
        return total_score / total_weight