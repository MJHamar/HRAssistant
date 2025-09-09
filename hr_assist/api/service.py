"""
Service layer for the API. Implements business logic and persistence.
"""
from typing import List, Optional, Tuple, Dict, Any
from uuid import uuid4

from ..func.lm import score_candidate as lm_score_candidate, make_questionnaire as lm_make_questionnaire
from ..utils import doc_to_md
from ..db import PostgresDB, BaseDb
from ..db.model import Document, DocumentChunk, Job, Candidate, QuestionnaireItem, Questionnaire


class HRService:
    def __init__(self):
        self._db: BaseDb = PostgresDB()
    
    def set_db(self, db: BaseDb) -> None:
        self._db = db

    @property
    def db(self) -> BaseDb:
        return self._db

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
        doc = Document(id=doc_id, content=content, chunks=chunks)
        self.db.upsert_document(doc)
        return doc_id
    
    def list_documents(self) -> List[Document]:
        return self.db.list_documents()

    def get_document(self, document_id: str) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]]]:
        doc = self.db.get_document(document_id)
        if not doc:
            return None, None
        chunk_payload = [c.model_dump() for c in (doc.chunks or [])]
        return doc.content, chunk_payload

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

    def patch_job(self, job_id: str, **fields) -> Optional[Job]:
        # Use DB generic modify and then fetch
        changed = {k: v for k, v in fields.items() if v is not None}
        if not changed:
            return self.db.get_job(job_id)
        ok = self.db.modify("jobs", key={"id": job_id}, changes=changed)
        if not ok:
            return None
        return self.db.get_job(job_id)

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

    def patch_candidate(self, candidate_id: str, **fields) -> Optional[Candidate]:
        changed = {k: v for k, v in fields.items() if v is not None}
        if not changed:
            return self.db.get_candidate(candidate_id)
        ok = self.db.modify("candidates", key={"id": candidate_id}, changes=changed)
        if not ok:
            return None
        return self.db.get_candidate(candidate_id)

    def score_candidate(self, candidate_id: str, job_id: str) -> Optional[Dict[str, Any]]:
        cand = self.db.get_candidate(candidate_id)
        job = self.db.get_job(job_id)
        if not cand or not job:
            return None

        # Fetch candidate CV content
        cv_text: Optional[str] = None
        if cand.candidate_cv_id:
            cv_doc = self.db.get_document(cand.candidate_cv_id)
            if cv_doc:
                cv_text = cv_doc.content
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