"""
Service layer for the API. Implements business logic and persistence.
"""
from typing import List, Optional, Tuple, Dict, Any
from uuid import uuid4

from ..func.lm import score_candidate as lm_score_candidate, make_questionnaire as lm_make_questionnaire
from ..utils import doc_to_md
from ..db import PostgresDB, BaseDb
from ..db.model import Document, Job, Candidate, QuestionnaireItem, Questionnaire


class HRService:
    def __init__(self):
        self._db: BaseDb = PostgresDB()
    
    def set_db(self, db: BaseDb) -> None:
        self._db = db
    
    @property
    def db(self) -> BaseDb:
        return self._db

    # ---- Documents ----
    def convert_document(self, document_name: str, document_content: bytes) -> Tuple[str, List[str], Dict[str, Any]]:
        """Convert raw document bytes into markdown, chunked content, and metadata."""
        contents, chunked_content, metadata = doc_to_md(document_content)
        # include document name in metadata if not present
        if metadata is None:
            metadata = {}
        if document_name and "document_name" not in metadata:
            metadata["document_name"] = document_name
        return contents, chunked_content, metadata

    def upload_document(self, document_name: str, document_content: bytes) -> str:
        """Convert and persist a document, returning its id."""
        contents, chunks, metadata = self.convert_document(document_name, document_content)
        doc_id = str(uuid4())
        doc = Document(id=doc_id, contents=contents, chunks=chunks, metadata=metadata)
        self.db.upsert_document(doc)
        return doc_id

    def list_documents(self) -> List[Document]:
        return self.db.list_documents()

    def get_document(self, document_id: str) -> Tuple[Optional[str], Optional[List[str]], Optional[Dict[str, Any]]]:
        doc = self.db.get_document(document_id)
        if not doc:
            return None, None, None
        return doc.contents, doc.chunks, doc.metadata

    def delete_document(self, document_id: str) -> bool:
        return self.db.delete_document(document_id)

    # ---- Jobs ----
    def upload_job_description(self, job_title: str, job_description: str, company_name: Optional[str] = None) -> str:
        job_id = str(uuid4())
        job = Job(id=job_id, company_name=company_name, job_title=job_title, job_description=job_description)
        self.db.upsert_job(job)
        return job_id

    def generate_questionnaire(self, job_id: str):
        """Generate and persist a questionnaire for a job, return the questionnaire items list for API response."""
        job = self.db.get_job(job_id)
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

        q_id = str(uuid4())
        questionnaire = Questionnaire(id=q_id, job_id=job_id, questionnaire=items)
        self.db.upsert_questionnaire(questionnaire)
        # API expects just the list of items under key "questionnaire"
        return [i.model_dump() for i in items]

    def list_jobs(self) -> List[Job]:
        return self.db.list_jobs()

    def get_job(self, job_id: str) -> Optional[Job]:
        return self.db.get_job(job_id)

    def delete_job(self, job_id: str) -> bool:
        return self.db.delete_job(job_id)

    # ---- Candidates ----
    def upload_candidate(self, candidate_name: str, candidate_cv_id: str) -> str:
        candidate_id = str(uuid4())
        cand = Candidate(id=candidate_id, candidate_name=candidate_name, candidate_cv_id=candidate_cv_id)
        self.db.upsert_candidate(cand)
        return candidate_id

    def list_candidates(self) -> List[Candidate]:
        return self.db.list_candidates()

    def get_candidate(self, candidate_id: str) -> Optional[Candidate]:
        return self.db.get_candidate(candidate_id)

    def delete_candidate(self, candidate_id: str) -> bool:
        return self.db.delete_candidate(candidate_id)

    def score_candidate(self, candidate_id: str, job_id: str) -> Optional[Dict[str, Any]]:
        cand = self.db.get_candidate(candidate_id)
        job = self.db.get_job(job_id)
        if not cand or not job:
            return None

        # Fetch candidate CV contents
        cv_text: Optional[str] = None
        if cand.candidate_cv_id:
            cv_doc = self.db.get_document(cand.candidate_cv_id)
            if cv_doc:
                cv_text = cv_doc.contents
        if not cv_text:
            # No CV to score
            return None

        # Get a questionnaire: use latest existing or generate if not present
        questionnaires = self.db.list_questionnaires(job_id=job_id, limit=1)
        if questionnaires:
            q_items = questionnaires[0].questionnaire
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