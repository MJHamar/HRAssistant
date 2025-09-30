"""
Service layer for the API. Implements business logic and persistence.
"""
import os
from typing import List, Optional, Tuple, Dict, Any
from uuid import uuid4
from sqlalchemy.orm import Session
from sqlalchemy import select, delete
from dotenv import load_dotenv

from ..utils import doc_to_md
from ..db import get_session_sync, create_tables, engine
from ..db.model import Document, DocumentChunk, Job, Candidate, QuestionnaireItem, Questionnaire
from ..model.lm import configure_dspy, get_dspy_modules, make_questionnaire as lm_make_questionnaire, score_candidate as lm_score_candidate


# Global singletons initialized at startup
_db_engine = None
_dspy_configured = False


def init_service() -> None:
    """
    Initialize global service components.

    This function should be called once at application startup to:
    1. Load configuration from .env file
    2. Ensure database is accessible and tables are created
    3. Initialize DB engine as singleton
    4. Configure DSPy and instantiate global prompting modules
    """
    global _db_engine, _dspy_configured

    # Load environment variables from .env file
    load_dotenv()

    # Verify database connection and create tables
    try:
        create_tables()
        # Test connection
        session = get_session_sync()
        session.close()
        print("✓ Database connection established and tables created")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize database: {e}")

    # Configure DSPy
    if not _dspy_configured:
        api_key = os.getenv("DSPY_API_KEY")
        if not api_key:
            raise RuntimeError("DSPY_API_KEY environment variable is required")

        provider = os.getenv("DSPY_PROVIDER", "gemini")
        model = os.getenv("DSPY_MODEL", "gemini-2.5-flash")

        configure_dspy(
            provider=provider,
            model=model,
            api_key=api_key
        )

        # Initialize global DSPy modules
        get_dspy_modules()
        _dspy_configured = True
        print("✓ DSPy configured and modules initialized")


class HRService:
    def __init__(self):
        self._db: Session = get_session_sync()

    def set_db(self, db: Session) -> None:
        self._db = db

    @property
    def db(self) -> Session:
        return self._db

    def close(self):
        """Close the database session."""
        self._db.close()

    # ---- Documents ----
    def convert_document(self, document_name: str, document_content: bytes) -> Tuple[str, List[DocumentChunk]]:
        """Convert raw document bytes into unified text content plus structured chunks."""
        # Existing util returns (converted_doc, page_chunked, metadata)
        converted_doc, page_chunked, _ = doc_to_md(document_name, document_content)
        # page_chunked expected list of dicts with 'text' and maybe 'metadata'
        return converted_doc, page_chunked

    def upload_document(self, document_name: str, document_content: bytes) -> str:
        content, page_chunked = self.convert_document(document_name, document_content)
        doc_id = str(uuid4())
        # attach document_id to chunks
        chunks: List[DocumentChunk] = []
        for idx, raw in enumerate(page_chunked):
            chunks.append(
                DocumentChunk(
                    document_id=doc_id,
                    idx=idx,
                    metadata=raw.get("metadata"),
                    toc_items=raw.get("toc_items"),
                    tables=raw.get("tables"),
                    images=raw.get("images"),
                    graphics=raw.get("graphics"),
                    text=raw.get("text"),
                )
            )
        doc = Document(id=doc_id, content=content)
        self.db.add(doc)
        for chunk in chunks:
            self.db.add(chunk)
        self.db.commit()
        return doc_id

    def list_documents(self) -> List[Document]:
        stmt = select(Document)
        return list(self.db.exec(stmt))

    def get_document(self, document_id: str) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]]]:
        stmt = select(Document).where(Document.id == document_id)
        doc = self.db.exec(stmt).first()
        if not doc:
            return None, None

        # Get chunks separately
        chunks_stmt = select(DocumentChunk).where(DocumentChunk.document_id == document_id)
        chunks = list(self.db.exec(chunks_stmt))
        chunk_payload = [c.model_dump() for c in chunks]
        return doc.content, chunk_payload

    def delete_document(self, document_id: str) -> bool:
        # Delete chunks first
        chunks_stmt = delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
        self.db.exec(chunks_stmt)

        # Delete document
        doc_stmt = delete(Document).where(Document.id == document_id)
        result = self.db.exec(doc_stmt)
        self.db.commit()
        return result.rowcount > 0

    # ---- Jobs ----
    def upload_job_description(self, job_title: str, job_description: str, company_name: Optional[str] = None) -> str:
        job_id = str(uuid4())
        job = Job(id=job_id, company_name=company_name, job_title=job_title, job_description=job_description)
        self.db.add(job)
        self.db.commit()
        return job_id

    def generate_questionnaire(self, job_id: str):
        """Generate and persist a questionnaire for a job, return the questionnaire items list for API response."""
        stmt = select(Job).where(Job.id == job_id)
        job = self.db.exec(stmt).first()
        if not job:
            return None
        pred = lm_make_questionnaire(job_description=job.job_description)
        # Expecting pred.criteria to be a list of objects with criterion and importance
        items: List[QuestionnaireItem] = []
        criteria = getattr(pred, "criteria", []) or []
        for c in criteria:
            # c may be a pydantic model or simple dict-like
            criterion = getattr(c, "criterion", None) or (c.get("criterion") if isinstance(c, dict) else None)
            importance = getattr(c, "importance", None) or (c.get("importance") if isinstance(c, dict) else None)
            if criterion:
                items.append(QuestionnaireItem(criterion=criterion, importance=importance))

        questionnaire = Questionnaire(job_id=job_id, questionnaire=items)
        self.db.merge(questionnaire)
        self.db.commit()
        # API expects just the list of items under key "questionnaire"
        return [i.model_dump() for i in items]

    def list_jobs(self) -> List[Job]:
        stmt = select(Job)
        return list(self.db.exec(stmt))

    def get_job(self, job_id: str) -> Optional[Job]:
        stmt = select(Job).where(Job.id == job_id)
        return self.db.exec(stmt).first()

    def delete_job(self, job_id: str) -> bool:
        stmt = delete(Job).where(Job.id == job_id)
        result = self.db.exec(stmt)
        self.db.commit()
        return result.rowcount > 0

    def patch_job(self, job_id: str, **fields) -> Optional[Job]:
        # Get existing job
        stmt = select(Job).where(Job.id == job_id)
        job = self.db.exec(stmt).first()
        if not job:
            return None

        # Update fields
        changed = {k: v for k, v in fields.items() if v is not None}
        if changed:
            for key, value in changed.items():
                setattr(job, key, value)
            self.db.merge(job)
            self.db.commit()

        return job

    # ---- Candidates ----
    def upload_candidate(self, candidate_name: str, candidate_cv_id: str) -> str:
        candidate_id = str(uuid4())
        cand = Candidate(id=candidate_id, candidate_name=candidate_name, candidate_cv_id=candidate_cv_id)
        self.db.add(cand)
        self.db.commit()
        return candidate_id

    def list_candidates(self) -> List[Candidate]:
        stmt = select(Candidate)
        return list(self.db.exec(stmt))

    def get_candidate(self, candidate_id: str) -> Optional[Candidate]:
        stmt = select(Candidate).where(Candidate.id == candidate_id)
        return self.db.exec(stmt).first()

    def delete_candidate(self, candidate_id: str) -> bool:
        stmt = delete(Candidate).where(Candidate.id == candidate_id)
        result = self.db.exec(stmt)
        self.db.commit()
        return result.rowcount > 0

    def patch_candidate(self, candidate_id: str, **fields) -> Optional[Candidate]:
        # Get existing candidate
        stmt = select(Candidate).where(Candidate.id == candidate_id)
        candidate = self.db.exec(stmt).first()
        if not candidate:
            return None

        # Update fields
        changed = {k: v for k, v in fields.items() if v is not None}
        if changed:
            for key, value in changed.items():
                setattr(candidate, key, value)
            self.db.merge(candidate)
            self.db.commit()

        return candidate

    def score_candidate(self, candidate_id: str, job_id: str) -> Optional[Dict[str, Any]]:
        # Get candidate and job using SQLAlchemy
        cand_stmt = select(Candidate).where(Candidate.id == candidate_id)
        cand = self.db.exec(cand_stmt).first()

        job_stmt = select(Job).where(Job.id == job_id)
        job = self.db.exec(job_stmt).first()

        if not cand or not job:
            return None

        # Fetch candidate CV content
        cv_text: Optional[str] = None
        if cand.candidate_cv_id:
            cv_stmt = select(Document).where(Document.id == cand.candidate_cv_id)
            cv_doc = self.db.exec(cv_stmt).first()
            if cv_doc:
                cv_text = cv_doc.content
        if not cv_text:
            # No CV to score
            return None

        # Get a questionnaire: use latest existing or generate if not present
        q_stmt = select(Questionnaire).where(Questionnaire.job_id == job_id).limit(1)
        questionnaire = self.db.exec(q_stmt).first()

        if questionnaire:
            q_items = questionnaire.questionnaire
        else:
            # Generate and persist
            gen_items = self.generate_questionnaire(job_id)
            if gen_items is None:
                return None
            q_items = [QuestionnaireItem(**qi) if isinstance(qi, dict) else qi for qi in gen_items]

        # Prepare input for LM
        lm_input_questionnaire = [
            {"criterion": qi.criterion, "importance": qi.importance} for qi in q_items
        ]
        pred = lm_score_candidate(candidate_cv=cv_text, questionnaire=lm_input_questionnaire)
        scores = getattr(pred, "scores", None)
        if not scores:
            return None

        report: Dict[str, Any] = {
            "candidate_id": candidate_id,
            "job_id": job_id,
            "questionnaire": lm_input_questionnaire,
            "scores": list(scores),
        }
        return report


__all__ = ["HRService"]