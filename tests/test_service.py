import types
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pytest

from hr_assist.api.service import HRService
from hr_assist.db.base import BaseDb
from hr_assist.db.model import Candidate, CandidateFitness, Document, Job, Questionnaire, QuestionnaireItem


class FakeDb(BaseDb):
    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self.jobs: Dict[str, Job] = {}
        self.candidates: Dict[str, Candidate] = {}
        self.questionnaires: Dict[str, Questionnaire] = {}
        self.candidate_fitness: Dict[Tuple[str, str, str], CandidateFitness] = {}

    # generic
    def close(self) -> None:  # pragma: no cover
        pass

    def ping(self) -> bool:  # pragma: no cover
        return True

    def query(
        self,
        table: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        order_by: Optional[Iterable[str]] = None,
        columns: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:  # pragma: no cover
        store = getattr(self, table)
        items = list(store.values())
        if limit is not None:
            items = items[offset : offset + limit]
        return [i.model_dump() for i in items]

    # documents
    def upsert_document(self, doc: Document) -> None:
        self.documents[doc.id] = doc

    def get_document(self, doc_id: str) -> Optional[Document]:
        return self.documents.get(doc_id)

    def list_documents(self, limit: Optional[int] = None, offset: int = 0) -> List[Document]:
        docs = list(self.documents.values())
        if limit is not None:
            docs = docs[offset : offset + limit]
        return docs

    def delete_document(self, doc_id: str) -> bool:
        return self.documents.pop(doc_id, None) is not None

    # jobs
    def upsert_job(self, job: Job) -> None:
        self.jobs[job.id] = job

    def get_job(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def list_jobs(self, limit: Optional[int] = None, offset: int = 0) -> List[Job]:
        jobs = list(self.jobs.values())
        if limit is not None:
            jobs = jobs[offset : offset + limit]
        return jobs

    def delete_job(self, job_id: str) -> bool:
        return self.jobs.pop(job_id, None) is not None

    # candidates
    def upsert_candidate(self, candidate: Candidate) -> None:
        self.candidates[candidate.id] = candidate

    def get_candidate(self, candidate_id: str) -> Optional[Candidate]:
        return self.candidates.get(candidate_id)

    def list_candidates(self, limit: Optional[int] = None, offset: int = 0) -> List[Candidate]:
        cands = list(self.candidates.values())
        if limit is not None:
            cands = cands[offset : offset + limit]
        return cands

    def delete_candidate(self, candidate_id: str) -> bool:
        return self.candidates.pop(candidate_id, None) is not None

    # questionnaire
    def upsert_questionnaire(self, questionnaire: Questionnaire) -> None:
        self.questionnaires[questionnaire.id] = questionnaire

    def get_questionnaire(self, questionnaire_id: str) -> Optional[Questionnaire]:
        return self.questionnaires.get(questionnaire_id)

    def list_questionnaires(
        self, job_id: Optional[str] = None, limit: Optional[int] = None, offset: int = 0
    ) -> List[Questionnaire]:
        items = list(self.questionnaires.values())
        if job_id is not None:
            items = [q for q in items if q.job_id == job_id]
        if limit is not None:
            items = items[offset : offset + limit]
        return items

    # fitness
    def upsert_candidate_fitness(self, fitness: CandidateFitness) -> None:  # pragma: no cover
        key = (fitness.candidate_id, fitness.job_id, fitness.questionnaire_id)
        self.candidate_fitness[key] = fitness

    def get_candidate_fitness(
        self, candidate_id: str, job_id: str, questionnaire_id: str
    ) -> Optional[CandidateFitness]:  # pragma: no cover
        return self.candidate_fitness.get((candidate_id, job_id, questionnaire_id))

    # vectors
    def upsert_embedding(
        self, table: str, record_id: str, embedding: List[float], metadata: Optional[Dict[str, Any]] = None
    ) -> None:  # pragma: no cover
        pass

    def vector_search(
        self,
        table: str,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        metric: str = "cosine",
    ) -> List[Tuple[str, float]]:  # pragma: no cover
        return []


@pytest.fixture()
def service(monkeypatch) -> HRService:
    db = FakeDb()
    svc = HRService(db=db)
    return svc


def test_upload_document_persists_and_returns_id(service: HRService, monkeypatch):
    # monkeypatch doc_to_md used inside service.convert_document
    monkeypatch.setattr("hr_assist.api.service.doc_to_md", lambda b: ("MD", ["C1", "C2"], {"k": "v"}))
    doc_id = service.upload_document("resume.pdf", b"fake-bytes")
    assert isinstance(doc_id, str) and doc_id
    doc = service.db.get_document(doc_id)
    assert doc is not None
    assert doc.contents == "MD"
    assert doc.chunks == ["C1", "C2"]
    # service should add document_name to metadata
    assert doc.metadata.get("document_name") == "resume.pdf"


def test_get_document_not_found_returns_nones(service: HRService):
    contents, chunks, metadata = service.get_document("missing")
    assert contents is None and chunks is None and metadata is None


def test_upload_job_description_persists_job(service: HRService):
    job_id = service.upload_job_description("Engineer", "Do work", company_name="ACME")
    job = service.db.get_job(job_id)
    assert job is not None
    assert job.job_title == "Engineer"
    assert job.company_name == "ACME"


def test_generate_questionnaire_missing_job_returns_none(service: HRService):
    assert service.generate_questionnaire("nope") is None


def test_generate_questionnaire_persists_and_returns_items(service: HRService, monkeypatch):
    # Arrange job
    job_id = service.upload_job_description("Engineer", "Build things")

    class Pred:
        criteria = [
            {"criterion": "Python", "importance": "high"},
            {"criterion": "Databases", "importance": "medium"},
        ]

    monkeypatch.setattr("hr_assist.api.service.lm_make_questionnaire", lambda **kw: Pred())
    items = service.generate_questionnaire(job_id)
    assert isinstance(items, list) and len(items) == 2
    # Persisted
    qs = service.db.list_questionnaires(job_id=job_id)
    assert len(qs) == 1
    assert [qi.criterion for qi in qs[0].questionnaire] == ["Python", "Databases"]


def test_upload_candidate_option_a(service: HRService):
    # Document already uploaded (frontend step 1)
    # Simulate by inserting doc into DB directly
    doc = Document(id="doc-1", contents="cv", chunks=["cv"], metadata={})
    service.db.upsert_document(doc)
    candidate_id = service.upload_candidate("Jane", "doc-1")
    cand = service.db.get_candidate(candidate_id)
    assert cand is not None and cand.candidate_cv_id == "doc-1"


def test_score_candidate_returns_none_when_missing_refs(service: HRService):
    # missing candidate
    assert service.score_candidate("c1", "j1") is None
    # candidate exists but job missing
    cand = Candidate(id="c1", candidate_name="A", candidate_cv_id="d1")
    service.db.upsert_candidate(cand)
    assert service.score_candidate("c1", "j1") is None
    # job exists but missing CV doc or contents
    job = Job(id="j1", company_name=None, job_title="T", job_description="D")
    service.db.upsert_job(job)
    # no document yet
    assert service.score_candidate("c1", "j1") is None


def test_score_candidate_uses_existing_questionnaire_and_lm(service: HRService, monkeypatch):
    # Setup candidate, job, CV doc
    doc = Document(id="d1", contents="CV text", chunks=[], metadata={})
    service.db.upsert_document(doc)
    cand = Candidate(id="c1", candidate_name="A", candidate_cv_id="d1")
    service.db.upsert_candidate(cand)
    job = Job(id="j1", company_name=None, job_title="T", job_description="D")
    service.db.upsert_job(job)
    # Existing questionnaire
    q = Questionnaire(
        id="q1",
        job_id="j1",
        questionnaire=[QuestionnaireItem(criterion="Python", importance="high"), QuestionnaireItem(criterion="DB", importance="medium")],
    )
    service.db.upsert_questionnaire(q)

    class Pred:
        scores = [0.5, 0.7]

    # Make sure we don't accidentally call generate_questionnaire
    def fail_generate(_):  # pragma: no cover
        raise AssertionError("should not generate questionnaire when one exists")

    monkeypatch.setattr("hr_assist.api.service.lm_score_candidate", lambda **kw: Pred())
    monkeypatch.setattr(HRService, "generate_questionnaire", fail_generate)

    report = service.score_candidate("c1", "j1")
    assert report is not None
    assert report["candidate_id"] == "c1"
    assert report["job_id"] == "j1"
    assert report["scores"] == [0.5, 0.7]
    assert [q["criterion"] for q in report["questionnaire"]] == ["Python", "DB"]
