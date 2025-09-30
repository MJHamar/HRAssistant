from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel
from sqlalchemy.orm import Session

from hr_assist.model.embed import init_embedder

from ..db.model import Candidate, Questionnaire, Document, Job, QuestionnaireItem
from ..model.embed import embedder


from .service import HRService
from ..search.pipeline import HRSearchService


# Documents
class DocumentUploadRequest(BaseModel):
    # For multipart/form-data uploads, the file is sent separately as UploadFile.
    # Keep only optional metadata here.
    document_name: Optional[str] = None

class DocumentUploadResponse(BaseModel):
    document_id: str

class DocumentListResponse(BaseModel):
    documents: List[Document]

DocumentResponse = Document

class DocumentDeleteResponse(BaseModel):
    message: Optional[str]

# Jobs
class JobUploadRequest(BaseModel):
    company_name: Optional[str]
    job_title: str
    job_description: str

class JobUploadResponse(BaseModel):
    job_id: str

class JobPatchRequest(BaseModel):
    """Partial update for a Job. All fields optional."""
    company_name: Optional[str] = None
    job_title: Optional[str] = None
    job_description: Optional[str] = None

QuestionnaireResponse = Questionnaire

class JobListResponse(BaseModel):
    jobs: List[Job]

JobResponse = Job

class JobDeleteResponse(BaseModel):
    message: Optional[str]

# Candidates
class CandidateUploadRequest(BaseModel):
    candidate_name: str
    candidate_cv_id: str

class CandidateUploadResponse(BaseModel):
    candidate_id: str

class CandidatePatchRequest(BaseModel):
    """Partial update for a Candidate. All fields optional."""
    candidate_name: Optional[str] = None
    candidate_cv_id: Optional[str] = None

class CandidateListResponse(BaseModel):
    candidates: List[Candidate]

CandidateResponse = Candidate

class CandidateDeleteResponse(BaseModel):
    message: Optional[str]

class CandidateScoreResponse(BaseModel):
    report: Dict[str, Any]


# Search API Models
class SearchSessionRequest(BaseModel):
    similarity_metric: Optional[str] = 'cosine'  # e.g., 'cosine', 'euclidean'
    num_questions: Optional[int] = 15
    rank_k: Optional[int] = 200  # Number of top candidates to consider for reranking
class SearchSessionResponse(BaseModel):
    job_id: str
    status: str
    questionnaire_count: int
    ideal_candidate_available: bool
    candidate_scores_count: int

class QuestionnaireGenerateRequest(BaseModel):
    use_existing_questions: bool = True
    precise_num_questions: bool = False

class QuestionnaireItemRequest(BaseModel):
    criterion: str
    importance: Optional[str] = None

class QuestionnaireRemoveRequest(BaseModel):
    criterion: Optional[str] = None
    index: Optional[int] = None

class QuestionnaireUpdateRequest(BaseModel):
    questionnaire: List[QuestionnaireItem]

class CandidateScoreUpdateRequest(BaseModel):
    candidate_id: str
    score: float

class CandidateScoresUpdateRequest(BaseModel):
    scores: List[Dict[str, Union[str, float]]]  # List of {candidate_id: str, score: float}

class GenerateScoresRequest(BaseModel):
    candidate_ids: Optional[List[str]] = None

class RankedCandidatesResponse(BaseModel):
    ranked_candidates: List[Candidate]


class UserSession(BaseModel):
    """User session containing database connection and services."""
    session_id: str
    db: Session
    base_service: HRService
    search_service: Optional[HRSearchService] = None

    def init_search_session(self, job_id: str, similarity_metric: str = 'cosine', num_questions: int = 15, rank_k = 200) -> HRSearchService:
        """Initialize or get the search service for a specific job."""
        from ..search.pipeline import HRSearchService
        from ..model.lm import make_questionnaire, score_candidate, make_resume
        from ..model.embed import PreTrainedEmbedder
        from sqlalchemy import select

        if self.search_service is not None:
            # Check if it's for the same job
            if str(self.search_service._job.id) == job_id:
                return self.search_service

        # Get the job
        stmt = select(Job).where(Job.id == job_id)
        job = self.db.exec(stmt).first()
        if not job:
            raise ValueError(f"Job with id {job_id} not found")

        # Create search service
        self.search_service = HRSearchService(
            db=self.db,
            embedder=embedder,
            ic_module=make_resume,
            q_module=make_questionnaire,
            s_module=score_candidate,
            job=job,
            similarity_metric=similarity_metric,
            num_questions=num_questions,
            rank_k=rank_k,
            parallelize_reranker=False #FIXME: parallelize once we know it works.
        )

        return self.search_service


__all__ = [
    # Documents
    "DocumentUploadRequest",
    "DocumentUploadResponse",
    "DocumentListResponse",
    "DocumentResponse",
    "DocumentDeleteResponse",
    # Jobs
    "JobUploadRequest",
    "JobUploadResponse",
    "JobPatchRequest",
    "QuestionnaireResponse",
    "JobListResponse",
    "JobResponse",
    "JobDeleteResponse",
    # Candidates
    "CandidateUploadRequest",
    "CandidateUploadResponse",
    "CandidatePatchRequest",
    "CandidateListResponse",
    "CandidateResponse",
    "CandidateDeleteResponse",
    "CandidateScoreResponse",
    # Search
    "SearchSessionResponse",
    "QuestionnaireGenerateRequest",
    "QuestionnaireItemRequest",
    "QuestionnaireRemoveRequest",
    "QuestionnaireUpdateRequest",
    "CandidateScoreUpdateRequest",
    "CandidateScoresUpdateRequest",
    "GenerateScoresRequest",
    "RankedCandidatesResponse",
    # Session
    "UserSession",
]