from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .model import (
    Candidate,
    CandidateFitness,
    Document,
    Job,
    JobCandidateScore,
    JobIdealCandidate,
    Questionnaire,
)


class BaseDb(ABC):
    """
    Abstract interface for HR Assistant database backends.

    Goals:
    - Provide flexible CRUD/query access across all tables.
    - Handle list/dict fields appropriately (arrays/JSON).
    - Offer optional vector support (store/search embeddings).
    """

    # ---- Generic helpers
    @abstractmethod
    def close(self) -> None:
        """Close any open resources/connections."""

    @abstractmethod
    def ping(self) -> bool:
        """Return True if the DB is reachable/healthy."""

    @abstractmethod
    def query(
        self,
        table: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        order_by: Optional[Iterable[str]] = None,
        columns: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generic query over a table.
        - filters: dict of column -> exact match, JSON containment for JSON columns.
        - order_by: iterable of column names, prefix with '-' for DESC.
        - columns: subset of columns to return; defaults to all.
        """

    @abstractmethod
    def modify(
        self,
        table: str,
        key: Dict[str, Any],
        changes: Dict[str, Any],
    ) -> bool:
        """Generic partial update.

        Parameters:
            table: table name.
            key: dict mapping primary key column(s) to value(s).
            changes: dict of columns -> new values (None values are ignored).

        Returns True if a row was updated, False if no matching row.
        Implementations must validate input and protect against SQL injection.
        """

    # ---- Documents
    @abstractmethod
    def upsert_document(self, doc: Document) -> None:
        pass

    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[Document]:
        pass

    @abstractmethod
    def list_documents(self, limit: Optional[int] = None, offset: int = 0) -> List[Document]:
        pass

    @abstractmethod
    def delete_document(self, doc_id: str) -> bool:
        pass

    # ---- Jobs
    @abstractmethod
    def upsert_job(self, job: Job) -> None:
        pass

    @abstractmethod
    def get_job(self, job_id: str) -> Optional[Job]:
        pass

    @abstractmethod
    def list_jobs(self, limit: Optional[int] = None, offset: int = 0) -> List[Job]:
        pass

    @abstractmethod
    def delete_job(self, job_id: str) -> bool:
        pass

    # ---- Job Ideal Candidates
    @abstractmethod
    def upsert_job_ideal_candidate(self, ideal_candidate: JobIdealCandidate) -> None:
        pass

    @abstractmethod
    def get_job_ideal_candidate(self, job_id: str) -> Optional[JobIdealCandidate]:
        pass

    @abstractmethod
    def list_job_ideal_candidates(self, limit: Optional[int] = None, offset: int = 0) -> List[JobIdealCandidate]:
        pass

    @abstractmethod
    def delete_job_ideal_candidate(self, job_id: str) -> bool:
        pass

    # ---- Candidates
    @abstractmethod
    def upsert_candidate(self, candidate: Candidate) -> None:
        pass

    @abstractmethod
    def get_candidate(self, candidate_id: str) -> Optional[Candidate]:
        pass

    @abstractmethod
    def list_candidates(self, limit: Optional[int] = None, offset: int = 0) -> List[Candidate]:
        pass

    @abstractmethod
    def delete_candidate(self, candidate_id: str) -> bool:
        pass

    # ---- Questionnaire
    @abstractmethod
    def upsert_questionnaire(self, questionnaire: Questionnaire) -> None:
        pass

    @abstractmethod
    def get_questionnaire(self, job_id: str) -> Optional[Questionnaire]:
        pass

    @abstractmethod
    def list_questionnaires(
        self, job_id: Optional[str] = None, limit: Optional[int] = None, offset: int = 0
    ) -> List[Questionnaire]:
        pass

    # ---- Job Candidate Scores
    @abstractmethod
    def upsert_job_candidate_score(self, score: JobCandidateScore) -> None:
        pass

    @abstractmethod
    def get_job_candidate_score(self, job_id: str, candidate_id: str) -> Optional[JobCandidateScore]:
        pass

    @abstractmethod
    def list_job_candidate_scores(
        self, job_id: str, limit: Optional[int] = None, offset: int = 0
    ) -> List[JobCandidateScore]:
        pass

    @abstractmethod
    def delete_job_candidate_scores(self, job_id: str) -> int:
        """
        Delete all candidate scores for a specific job.

        Returns:
            Number of rows deleted
        """

    # ---- Candidate Fitness
    @abstractmethod
    def upsert_candidate_fitness(self, fitness: CandidateFitness) -> None:
        pass

    @abstractmethod
    def get_candidate_fitness(
        self, candidate_id: str, job_id: str, questionnaire_id: str
    ) -> Optional[CandidateFitness]:
        pass

    # ---- Vector support
    @abstractmethod
    def upsert_embedding(
        self,
        table: str,
        record_id: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store or update an embedding for a given record."""

    @abstractmethod
    def vector_search(
        self,
        table: str,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        metric: str = "cosine",
    ) -> List[Tuple[str, float]]:
        """
        Return a list of (record_id, score), higher score means more similar for cosine.
        Implementations may use pgvector, external vector DBs, or in-memory fallback.
        """

    # ---- Batch operations
    @abstractmethod
    def batch_upsert(
        self,
        table: str,
        records: List[Dict[str, Any]],
        conflict_columns: Optional[List[str]] = None
    ) -> None:
        """
        Batch insert/update records into a table.

        Parameters:
            table: table name
            records: list of record dictionaries
            conflict_columns: columns to use for conflict resolution (ON CONFLICT)
        """

    @abstractmethod
    def batch_modify(
        self,
        table: str,
        updates: List[Tuple[Dict[str, Any], Dict[str, Any]]]
    ) -> int:
        """
        Batch modify records in a table.

        Parameters:
            table: table name
            updates: list of (key_dict, changes_dict) tuples

        Returns:
            Number of rows updated
        """

    @abstractmethod
    def batch_delete(
        self,
        table: str,
        keys: List[Dict[str, Any]]
    ) -> int:
        """
        Batch delete records from a table.

        Parameters:
            table: table name
            keys: list of key dictionaries identifying records to delete

        Returns:
            Number of rows deleted
        """


__all__ = ["BaseDb"]
