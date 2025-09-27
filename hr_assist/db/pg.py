
import os
import re
from typing import Literal
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .base import BaseDb
from .model import Candidate, CandidateFitness, Document, DocumentChunk, Job, JobIdealCandidate, Questionnaire, QuestionnaireItem


def _as_array(val: Optional[List[Any]]) -> Optional[List[Any]]:
    return None if val is None else list(val)


TABLE_MODEL_MAP = {
    "documents": Document,
    "document_chunks": DocumentChunk,
    "jobs": Job,
    "job_ideal_candidates": JobIdealCandidate,
    "candidates": Candidate,
    "questionnaires": Questionnaire,
    "candidate_fitness": CandidateFitness,
}

class PostgresDB(BaseDb):
    def __init__(
        self,
        dsn: Optional[str] = None,
        autocommit: bool = True,
    ) -> None:
        # Lazy import psycopg to avoid hard dependency at import time
        import importlib

        try:
            self._psycopg = importlib.import_module("psycopg")
            rows_mod = importlib.import_module("psycopg.rows")
            types_json_mod = importlib.import_module("psycopg.types.json")
        except ModuleNotFoundError as e:  # pragma: no cover
            raise RuntimeError("psycopg is required for PostgresDB") from e

        self._dict_row = getattr(rows_mod, "dict_row")
        self._Jsonb = getattr(types_json_mod, "Jsonb")

        if not dsn:
            self._db_user = os.getenv("DATABASE_USER") or "postgres"
            self._db_password = os.getenv("DATABASE_PASSWORD") or "password"
            self._db_host = os.getenv("DATABASE_HOST") or "localhost"
            self._db_name = os.getenv("DATABASE_DB") or "hr_assistant"
            self._dsn = f"postgresql://{self._db_user}:{self._db_password}@{self._db_host}/{self._db_name}"
        else:
            self._dsn = dsn

        import psycopg
        self._psycopg = psycopg
        self._conn = self._psycopg.connect(self._dsn, autocommit=autocommit, row_factory=self._dict_row)

    def close(self) -> None:
        self._conn.close()

    def ping(self) -> bool:
        try:
            with self._conn.cursor() as cur:
                cur.execute("SELECT 1")
                _ = cur.fetchone()
            return True
        except Exception:
            return False

    @contextmanager
    def _cursor(self):
        with self._conn.cursor() as cur:
            yield cur

    # ---- Generic query
    def query(
        self,
        table: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        order_by: Optional[Iterable[str]] = None,
        columns: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:
        cols = ", ".join(columns) if columns else "*"
        sql = [f"SELECT {cols} FROM {table}"]
        params: List[Any] = []

        if filters:
            where_clauses = []
            for k, v in filters.items():
                if isinstance(v, dict):
                    where_clauses.append(f"{k} @> %s")
                    params.append(self._Jsonb(v))
                else:
                    where_clauses.append(f"{k} = %s")
                    params.append(v)
            sql.append("WHERE " + " AND ".join(where_clauses))

        if order_by:
            parts = []
            for ob in order_by:
                if ob.startswith("-"):
                    parts.append(f"{ob[1:]} DESC")
                else:
                    parts.append(f"{ob} ASC")
            sql.append("ORDER BY " + ", ".join(parts))

        if limit is not None:
            sql.append("LIMIT %s")
            params.append(limit)
        if offset:
            sql.append("OFFSET %s")
            params.append(offset)

        query_str = " ".join(sql)
        with self._cursor() as cur:
            cur.execute(query_str, params)
            rows = cur.fetchall() or []
        return rows

    def modify(self, table: str, key: Dict[str, Any], changes: Dict[str, Any]) -> bool:
        def _is_safe_ident(s: str) -> bool:
            return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", s or ""))

        if not table or not isinstance(key, dict) or not key:
            raise ValueError("modify requires a table and a non-empty key dict")
        if not isinstance(changes, dict) or not changes:
            return False
        if not _is_safe_ident(table):
            raise ValueError("invalid table name")
        for k in list(key.keys()) + list(changes.keys()):
            if not _is_safe_ident(k):
                raise ValueError("invalid column name")
        # Filter out None values so we don't overwrite with NULL inadvertently
        filtered = {k: v for k, v in changes.items() if v is not None}
        if not filtered:
            return False
        set_clauses: List[str] = []
        params: List[Any] = []
        for col, val in filtered.items():
            # Heuristic: wrap dicts/lists as JSONB
            if isinstance(val, (dict, list)):
                set_clauses.append(f"{col} = %s")
                params.append(self._Jsonb(val))
            else:
                set_clauses.append(f"{col} = %s")
                params.append(val)
        where_clauses: List[str] = []
        for kcol, kval in key.items():
            where_clauses.append(f"{kcol} = %s")
            params.append(kval)
        sql = f"UPDATE {table} SET " + ", ".join(set_clauses) + " WHERE " + " AND ".join(where_clauses)
        try:
            with self._cursor() as cur:
                cur.execute(sql, params)
                return cur.rowcount > 0
        except Exception as e:
            # For safety, don't leak SQL. Re-raise a sanitized error.
            raise RuntimeError(f"Failed to modify {table}") from e

    # ---- Documents
    def upsert_document(self, doc: Document) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (id, content) VALUES (%s, %s)
                ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content
                """,
                [doc.id, doc.content],
            )
            # Replace chunks
            cur.execute("DELETE FROM document_chunks WHERE document_id = %s", [doc.id])
            if doc.chunks:
                for ch in doc.chunks:
                    cur.execute(
                        """
                        INSERT INTO document_chunks (document_id, idx, metadata, toc_items, tables, images, graphics, text)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        [
                            ch.document_id,
                            ch.idx,
                            self._Jsonb(ch.metadata) if ch.metadata is not None else None,
                            self._Jsonb(ch.toc_items) if ch.toc_items is not None else None,
                            self._Jsonb(ch.tables) if ch.tables is not None else None,
                            self._Jsonb(ch.images) if ch.images is not None else None,
                            self._Jsonb(ch.graphics) if ch.graphics is not None else None,
                            ch.text,
                        ],
                    )

    def get_document(self, doc_id: str) -> Optional[Document]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM documents WHERE id = %s", [doc_id])
            row = cur.fetchone()
            if not row:
                return None
            cur.execute(
                "SELECT document_id, idx, metadata, toc_items, tables, images, graphics, text FROM document_chunks WHERE document_id = %s ORDER BY idx ASC",
                [doc_id],
            )
            chunk_rows = cur.fetchall() or []
        chunks = [DocumentChunk(**cr) for cr in chunk_rows]
        return Document(id=row["id"], content=row.get("content"), chunks=chunks)

    def list_documents(self, limit: Optional[int] = None, offset: int = 0) -> List[Document]:
        rows = self.query("documents", limit=limit, offset=offset, order_by=["id"])
        return [Document(id=r["id"], content=r.get("content"), chunks=None) for r in rows]

    def delete_document(self, doc_id: str) -> bool:
        with self._cursor() as cur:
            cur.execute("DELETE FROM documents WHERE id = %s", [doc_id])
            return cur.rowcount > 0

    # ---- Jobs
    def upsert_job(self, job: Job) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO jobs (id, company_name, job_title, job_description)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    company_name = EXCLUDED.company_name,
                    job_title = EXCLUDED.job_title,
                    job_description = EXCLUDED.job_description
                """,
                [job.id, job.company_name, job.job_title, job.job_description],
            )

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM jobs WHERE id = %s", [job_id])
            row = cur.fetchone()
        return Job(**row) if row else None

    def list_jobs(self, limit: Optional[int] = None, offset: int = 0) -> List[Job]:
        rows = self.query("jobs", limit=limit, offset=offset, order_by=["id"])
        return [Job(**r) for r in rows]

    def delete_job(self, job_id: str) -> bool:
        with self._cursor() as cur:
            cur.execute("DELETE FROM jobs WHERE id = %s", [job_id])
            return cur.rowcount > 0

    # ---- Job Ideal Candidates
    def upsert_job_ideal_candidate(self, ideal_candidate: JobIdealCandidate) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO job_ideal_candidates (job_id, ideal_candidate_resume)
                VALUES (%s, %s)
                ON CONFLICT (job_id) DO UPDATE SET
                    ideal_candidate_resume = EXCLUDED.ideal_candidate_resume
                """,
                [ideal_candidate.job_id, ideal_candidate.ideal_candidate_resume],
            )

    def get_job_ideal_candidate(self, job_id: str) -> Optional[JobIdealCandidate]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM job_ideal_candidates WHERE job_id = %s", [job_id])
            row = cur.fetchone()
        return JobIdealCandidate(**row) if row else None

    def list_job_ideal_candidates(self, limit: Optional[int] = None, offset: int = 0) -> List[JobIdealCandidate]:
        rows = self.query("job_ideal_candidates", limit=limit, offset=offset, order_by=["job_id"])
        return [JobIdealCandidate(**r) for r in rows]

    def delete_job_ideal_candidate(self, job_id: str) -> bool:
        with self._cursor() as cur:
            cur.execute("DELETE FROM job_ideal_candidates WHERE job_id = %s", [job_id])
            return cur.rowcount > 0

    # ---- Candidates
    def upsert_candidate(self, candidate: Candidate) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO candidates (id, candidate_name, candidate_cv_id)
                VALUES (%s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    candidate_name = EXCLUDED.candidate_name,
                    candidate_cv_id = EXCLUDED.candidate_cv_id
                """,
                [candidate.id, candidate.candidate_name, candidate.candidate_cv_id],
            )

    def get_candidate(self, candidate_id: str) -> Optional[Candidate]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM candidates WHERE id = %s", [candidate_id])
            row = cur.fetchone()
        return Candidate(**row) if row else None

    def list_candidates(self, limit: Optional[int] = None, offset: int = 0) -> List[Candidate]:
        rows = self.query("candidates", limit=limit, offset=offset, order_by=["id"])
        return [Candidate(**r) for r in rows]

    def delete_candidate(self, candidate_id: str) -> bool:
        with self._cursor() as cur:
            cur.execute("DELETE FROM candidates WHERE id = %s", [candidate_id])
            return cur.rowcount > 0

    # ---- Questionnaire
    def upsert_questionnaire(self, questionnaire: Questionnaire) -> None:
        payload = {"questionnaire": [qi.model_dump() for qi in questionnaire.questionnaire]}
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO questionnaires (job_id, questionnaire)
                VALUES (%s, %s)
                ON CONFLICT (job_id) DO UPDATE SET
                    questionnaire = EXCLUDED.questionnaire
                """,
                [questionnaire.job_id, self._Jsonb(payload)],
            )

    def get_questionnaire(self, job_id: str) -> Optional[Questionnaire]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM questionnaires WHERE job_id = %s", [job_id])
            row = cur.fetchone()
        if not row:
            return None
        q_items = [QuestionnaireItem(**qi) for qi in (row["questionnaire"] or {}).get("questionnaire", [])]
        return Questionnaire(job_id=row["job_id"], questionnaire=q_items)

    def list_questionnaires(
        self, job_id: Optional[str] = None, limit: Optional[int] = None, offset: int = 0
    ) -> List[Questionnaire]:
        filters = {"job_id": job_id} if job_id else None
        rows = self.query("questionnaires", filters=filters, limit=limit, offset=offset, order_by=["job_id"])
        items: List[Questionnaire] = []
        for r in rows:
            q_items = [QuestionnaireItem(**qi) for qi in (r["questionnaire"] or {}).get("questionnaire", [])]
            items.append(Questionnaire(job_id=r["job_id"], questionnaire=q_items))
        return items

    # ---- Candidate Fitness
    def upsert_candidate_fitness(self, fitness: CandidateFitness) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO candidate_fitness (candidate_id, job_id, questionnaire_id, scores)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (candidate_id, job_id, questionnaire_id) DO UPDATE SET
                    scores = EXCLUDED.scores
                """,
                [fitness.candidate_id, fitness.job_id, fitness.questionnaire_id, fitness.scores],
            )

    def get_candidate_fitness(self, candidate_id: str, job_id: str, questionnaire_id: str):
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT * FROM candidate_fitness WHERE candidate_id = %s AND job_id = %s AND questionnaire_id = %s
                """,
                [candidate_id, job_id, questionnaire_id],
            )
            row = cur.fetchone()
        return CandidateFitness(**row) if row else None

    # ---- Vectors
    def upsert_embedding(
        self,
        table: str,
        record_id: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO embeddings (table_name, record_id, embedding, metadata)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (table_name, record_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata
                """,
                [table, record_id, embedding, self._Jsonb(metadata) if metadata is not None else None],
            )

    def _get_embedding_op(self, metric: Literal["cosine", "euclidean", "inner_product"]) -> str:
        if metric == "cosine":
            return "<=>"
        elif metric == "euclidean":
            return "<->"
        elif metric == "inner_product":
            return "<#>"
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")

    def vector_search(
        self,
        table: str,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        metric: Literal["cosine", "euclidean", "inner_product"] = "cosine",
    ) -> List[Tuple[str, float]]:
        op = self._get_embedding_op(metric)
        sql = [
            "SELECT record_id, 1 - (embedding %s %s) AS score FROM embeddings WHERE table_name = %s"
            % (op, "%s")
        ]
        params: List[Any] = [query_embedding, table]
        if filters:
            for k, v in filters.items():
                sql.append("AND (metadata ->> %s) = %s")
                params.extend([k, str(v)])
        sql.append("ORDER BY score DESC LIMIT %s")
        params.append(top_k)
        query_str = " ".join(sql)
        with self._cursor() as cur:
            cur.execute(query_str, params)
            rows = cur.fetchall() or []

        # retrieve the actual records from the target table
        retlist = []
        constructor = TABLE_MODEL_MAP.get(table, dict)
        cur.execute(f"SELECT * FROM {table} WHERE id IN %s", ([r["record_id"] for r in rows],))
        for record in cur.fetchall() or []:
            rec_obj = constructor(**record) if constructor != dict else record
            score = next((r["score"] for r in rows if r["record_id"] == record["id"]), None)
            retlist.append((rec_obj, score))

        return retlist

__all__ = ["PostgresDB"]
