from fastapi import Cookie, FastAPI, Header, Path, HTTPException, Depends, Request, Response, UploadFile, File, Form
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi_sessions.backends.implementations import InMemoryBackend
from fastapi_sessions.session_verifier import SessionVerifier
from fastapi_sessions.frontends.implementations import SessionCookie, CookieParameters
from uuid import uuid4
from sqlalchemy import select

from hr_assist.api.model import (
    DocumentUploadRequest,
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentResponse,
    DocumentDeleteResponse,
    JobUploadRequest,
    JobUploadResponse,
    JobPatchRequest,
    QuestionnaireResponse,
    JobListResponse,
    JobResponse,
    JobDeleteResponse,
    CandidateUploadRequest,
    CandidateUploadResponse,
    CandidatePatchRequest,
    CandidateListResponse,
    CandidateResponse,
    CandidateDeleteResponse,
    CandidateScoreResponse,
    # Search models
    SearchSessionResponse,
    QuestionnaireGenerateRequest,
    QuestionnaireItemRequest,
    QuestionnaireRemoveRequest,
    QuestionnaireUpdateRequest,
    CandidateScoreUpdateRequest,
    CandidateScoresUpdateRequest,
    GenerateScoresRequest,
    RankedCandidatesResponse,
    UserSession,
)

from .. import __version__
from .service import HRService, init_service
from ..db import get_session_sync
from ..db.model import Job, QuestionnaireItem, Questionnaire, JobCandidateScore
from ..search.ranking import PgRanker
from ..model.lm import (
    make_questionnaire as questionnaire_module,
    make_resume as ideal_candidate_module,
    score_candidate as scoring_module,
    )
from ..search.pipeline import HRSearchService


app = FastAPI(title="HR Assistant API", version=__version__)


# Session management
class BasicVerifier(SessionVerifier[uuid4, UserSession]):
    def __init__(
        self,
        *,
        identifier: str,
        auto_error: bool,
        backend: InMemoryBackend[uuid4, UserSession],
        auth_http_exception: HTTPException,
    ):
        self._identifier = identifier
        self._auto_error = auto_error
        self._backend = backend
        self._auth_http_exception = auth_http_exception

    @property
    def identifier(self):
        return self._identifier

    @property
    def backend(self):
        return self._backend

    @property
    def auto_error(self):
        return self._auto_error

    @property
    def auth_http_exception(self):
        return self._auth_http_exception

    def verify_session(self, model: UserSession) -> bool:
        """If the session exists, it is valid"""
        return True


# Session configuration
cookie_params = CookieParameters()
cookie = SessionCookie(
    cookie_name="hr_session",
    identifier="general_verifier",
    auto_error=True,
    secret_key="DONOTUSE",  # TODO: Use proper secret key from env
    cookie_params=cookie_params,
)
backend = InMemoryBackend[uuid4, UserSession]()
verifier = BasicVerifier(
    identifier="general_verifier",
    auto_error=True,
    backend=backend,
    auth_http_exception=HTTPException(status_code=403, detail="invalid session"),
)


# Initialize service on startup
@app.on_event("startup")
async def startup_event():
    """Initialize global services on application startup."""
    init_service()


def get_user_session(request: Request) -> UserSession:
    """Get or create user session."""
    session_id = cookie(request)
    if session_id is None:
        # Create new session
        session_id = uuid4()
        db_session = get_session_sync()
        base_service = HRService(db_session)

        user_session = UserSession(
            session_id=str(session_id),
            db=db_session,
            base_service=base_service,
            search_service=None  # Will be created when needed
        )

        backend.create(session_id, user_session)
        cookie.attach_to_response(request, session_id)
        return user_session

    # Get existing session
    user_session = backend.read(session_id)
    return user_session

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3140"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the HR Assistant API. Visit /openapi.json for API documentation."}

@app.get("/openapi.json", include_in_schema=False)
def get_openapi_json():
    return app.openapi()

@app.post("/documents", tags=["Documents"], summary="Upload and process a document", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    document_name: Optional[str] = Form(None),
    session: UserSession = Depends(get_user_session),
):
    content_bytes = await file.read()
    name = document_name or file.filename
    document_id = session.base_service.upload_document(name, content_bytes)
    return {"document_id": document_id}

@app.get("/documents", tags=["Documents"], summary="List all documents", response_model=DocumentListResponse)
def list_documents(session: UserSession = Depends(get_user_session)):
    documents = session.base_service.list_documents()
    return {
        "documents": documents
    }

@app.get("/documents/{document_id}", tags=["Documents"], summary="Retrieve a document", response_model=DocumentResponse)
def get_document(document_id: str, session: UserSession = Depends(get_user_session)):
    content, chunks = session.base_service.get_document(document_id)
    if not content:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "id": document_id,
        "content": content,
        "chunks": chunks,
    }

@app.delete("/documents/{document_id}", tags=["Documents"], summary="Delete a document", response_model=DocumentDeleteResponse)
def delete_document(document_id: str, session: UserSession = Depends(get_user_session)):
    success = session.base_service.delete_document(document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Document deleted successfully"}

@app.post("/jobs", tags=["Jobs"], summary="Upload a new job description", response_model=JobUploadResponse)
def upload_job_description(request: JobUploadRequest, session: UserSession = Depends(get_user_session)):
    job_id = session.base_service.upload_job_description(request.job_title, request.job_description, request.company_name)
    return {
        "job_id": job_id
    }

@app.get("/jobs/{job_id}/questionnaire", tags=["Jobs"], summary="Generate a questionnaire for a job", response_model=QuestionnaireResponse)
def generate_questionnaire(job_id: str, session: UserSession = Depends(get_user_session)):
    questionnaire = session.base_service.generate_questionnaire(job_id)
    if not questionnaire:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job_id,
        "questionnaire": questionnaire
    }

@app.get("/jobs", tags=["Jobs"], summary="List all jobs", response_model=JobListResponse)
def list_jobs(session: UserSession = Depends(get_user_session)):
    jobs = session.base_service.list_jobs()
    return {
        "jobs": jobs
    }

@app.get("/jobs/{job_id}", tags=["Jobs"], summary="Retrieve a job description", response_model=JobResponse)
def get_job(job_id: str, session: UserSession = Depends(get_user_session)):
    job = session.base_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.delete("/jobs/{job_id}", tags=["Jobs"], summary="Delete a job description", response_model=JobDeleteResponse)
def delete_job(job_id: str, session: UserSession = Depends(get_user_session)):
    success = session.base_service.delete_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"message": "Job deleted successfully"}

@app.patch("/jobs/{job_id}", tags=["Jobs"], summary="Patch (partial update) a job", response_model=JobResponse)
def patch_job(job_id: str, request: JobPatchRequest, session: UserSession = Depends(get_user_session)):
    try:
        updated = session.base_service.patch_job(
            job_id,
            company_name=request.company_name,
            job_title=request.job_title,
            job_description=request.job_description,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError:
        raise HTTPException(status_code=500, detail="Failed to update job")
    if not updated:
        raise HTTPException(status_code=404, detail="Job not found")
    return updated

@app.post("/candidates", tags=["Candidates"], summary="Upload a new candidate CV", response_model=CandidateUploadResponse)
def upload_candidate(request: CandidateUploadRequest, session: UserSession = Depends(get_user_session)):
    candidate_id = session.base_service.upload_candidate(request.candidate_name, request.candidate_cv_id)
    return {
        "candidate_id": candidate_id
    }

@app.get("/candidates", tags=["Candidates"], summary="List all candidates", response_model=CandidateListResponse)
def list_candidates(session: UserSession = Depends(get_user_session)):
    candidates = session.base_service.list_candidates()
    return {
        "candidates": candidates
    }

@app.get("/candidates/{candidate_id}", tags=["Candidates"], summary="Retrieve a candidate CV", response_model=CandidateResponse)
def get_candidate(candidate_id: str, session: UserSession = Depends(get_user_session)):
    candidate = session.base_service.get_candidate(candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return candidate

@app.delete("/candidates/{candidate_id}", tags=["Candidates"], summary="Delete a candidate CV", response_model=CandidateDeleteResponse)
def delete_candidate(candidate_id: str, session: UserSession = Depends(get_user_session)):
    success = session.base_service.delete_candidate(candidate_id)
    if not success:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return {"message": "Candidate deleted successfully"}

@app.patch("/candidates/{candidate_id}", tags=["Candidates"], summary="Patch (partial update) a candidate", response_model=CandidateResponse)
def patch_candidate(candidate_id: str, request: CandidatePatchRequest, session: UserSession = Depends(get_user_session)):
    try:
        updated = session.base_service.patch_candidate(
            candidate_id,
            candidate_name=request.candidate_name,
            candidate_cv_id=request.candidate_cv_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError:
        raise HTTPException(status_code=500, detail="Failed to update candidate")
    if not updated:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return updated

@app.post("/candidates/{candidate_id}/score/{job_id}", tags=["Candidates"], summary="Score a candidate against a job", response_model=CandidateScoreResponse)
def score_candidate(candidate_id: str, job_id: str, session: UserSession = Depends(get_user_session)):
    report = session.base_service.score_candidate(candidate_id, job_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Candidate or Job not found")
    return report


@app.get("/search/{job_id}", tags=["Search"], summary="Initialize or get search session for a job", response_model=SearchSessionResponse)
def init_search_session(job_id: str, similarity_metric: str = 'cosine', num_questions: int = 15, rank_k: int = 200, session: UserSession = Depends(get_user_session)):
    """Initialize search session for a specific job."""
    # Check if job exists
    stmt = select(Job).where(Job.id == job_id)
    job = session.db.exec(stmt).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Initialize search service
    search_service = session.init_search_session(
        job_id, similarity_metric=similarity_metric, num_questions=num_questions, rank_k=rank_k)

    return {
        "job_id": job_id,
        "status": "initialized",
        "questionnaire_count": len(search_service.questionnaire.questionnaire) if search_service.questionnaire else 0,
        "ideal_candidate_available": search_service.ideal_candidate is not None,
        "candidate_scores_count": len(search_service.candidate_scores)
    }


# Questionnaire endpoints
@app.post("/search/{job_id}/questionnaire/generate", tags=["Search"], summary="Generate questionnaire for job")
def generate_questionnaire(job_id: str, request: QuestionnaireGenerateRequest, session: UserSession = Depends(get_user_session)):
    """Generate a questionnaire for the job."""
    search_service = session.init_search_session(job_id)
    search_service.generate_questionnaire(num_questions=request.num_questions)
    return {"message": "Questionnaire generated successfully"}


@app.get("/search/{job_id}/questionnaire", tags=["Search"], summary="Get questionnaire for job")
def get_questionnaire(job_id: str, session: UserSession = Depends(get_user_session)):
    """Get the questionnaire for the job."""
    search_service = session.init_search_session(job_id)
    questionnaire = search_service.questionnaire()
    if not questionnaire:
        raise HTTPException(status_code=404, detail="Questionnaire not found")
    return questionnaire


@app.put("/search/{job_id}/questionnaire", tags=["Search"], summary="Set questionnaire for job")
def set_questionnaire(job_id: str, request: QuestionnaireUpdateRequest, session: UserSession = Depends(get_user_session)):
    """Set the questionnaire for the job."""
    search_service = session.init_search_session(job_id)
    questionnaire = Questionnaire(job_id=job_id, questionnaire=request.questionnaire)
    search_service.set_questionnaire(questionnaire)
    return {"message": "Questionnaire updated successfully"}


@app.delete("/search/{job_id}/questionnaire", tags=["Search"], summary="Delete questionnaire for job")
def delete_questionnaire(job_id: str, session: UserSession = Depends(get_user_session)):
    """Delete the questionnaire for the job."""
    search_service = session.init_search_session(job_id)
    search_service.delete_questionnaire()
    return {"message": "Questionnaire deleted successfully"}


@app.post("/search/{job_id}/questionnaire/items", tags=["Search"], summary="Add questionnaire item")
def add_questionnaire_item(job_id: str, item: QuestionnaireItemRequest, session: UserSession = Depends(get_user_session)):
    """Add an item to the questionnaire."""
    search_service = session.init_search_session(job_id)
    questionnaire_item = QuestionnaireItem(
        criterion=item.criterion,
        importance=item.importance
    )
    search_service.add_questionnaire_item(questionnaire_item)
    return {"message": "Questionnaire item added successfully"}


@app.delete("/search/{job_id}/questionnaire/items", tags=["Search"], summary="Remove questionnaire item")
def remove_questionnaire_item(job_id: str, request: QuestionnaireRemoveRequest, session: UserSession = Depends(get_user_session)):
    """Remove an item from the questionnaire."""
    search_service = session.init_search_session(job_id)
    search_service.remove_questionnaire_item(criterion=request.criterion, index=request.index)
    return {"message": "Questionnaire item removed successfully"}


# Ideal candidate endpoints
@app.post("/search/{job_id}/ideal-candidate/generate", tags=["Search"], summary="Generate ideal candidate resume")
def generate_ideal_candidate(job_id: str, session: UserSession = Depends(get_user_session)):
    """Generate an ideal candidate resume for the job."""
    search_service = session.init_search_session(job_id)
    ideal_candidate = search_service.generate_ideal_candidate()
    return {"ideal_candidate_resume": ideal_candidate.ideal_candidate_resume}


@app.get("/search/{job_id}/ideal-candidate", tags=["Search"], summary="Get ideal candidate resume")
def get_ideal_candidate(job_id: str, session: UserSession = Depends(get_user_session)):
    """Get the ideal candidate resume for the job."""
    search_service = session.init_search_session(job_id)
    ideal_candidate = search_service.ideal_candidate()
    if ideal_candidate is None:
        raise HTTPException(status_code=404, detail="Ideal candidate not found")
    return {"ideal_candidate_resume": ideal_candidate.ideal_candidate_resume}


@app.put("/search/{job_id}/ideal-candidate", tags=["Search"], summary="Set ideal candidate resume")
def set_ideal_candidate(job_id: str, resume: str, session: UserSession = Depends(get_user_session)):
    """Set the ideal candidate resume for the job."""
    search_service = session.init_search_session(job_id)
    search_service.set_ideal_candidate(resume)
    return {"message": "Ideal candidate resume updated successfully"}


@app.delete("/search/{job_id}/ideal-candidate", tags=["Search"], summary="Delete ideal candidate resume")
def delete_ideal_candidate(job_id: str, session: UserSession = Depends(get_user_session)):
    """Delete the ideal candidate resume for the job."""
    search_service = session.init_search_session(job_id)
    search_service.delete_ideal_candidate()
    return {"message": "Ideal candidate resume deleted successfully"}


# Candidate scoring endpoints
@app.get("/search/{job_id}/candidates/rank", tags=["Search"], summary="Rank candidates for job", response_model=RankedCandidatesResponse)
def rank_candidates(job_id: str, top_k: Optional[int] = None, session: UserSession = Depends(get_user_session)):
    """Rank candidates for the job."""
    search_service = session.init_search_session(job_id)
    ranked_candidates = search_service.rank_candidates(top_k=top_k)
    return {"ranked_candidates": ranked_candidates}


@app.get("/search/{job_id}/candidates/scores", tags=["Search"], summary="Get candidate scores for job")
def get_candidate_scores(job_id: str, session: UserSession = Depends(get_user_session)):
    """Get candidate scores for the job."""
    search_service = session.init_search_session(job_id)
    scores = search_service.candidate_scores
    return {"candidate_scores": scores}


@app.put("/search/{job_id}/candidates/scores", tags=["Search"], summary="Update candidate scores")
def update_candidate_scores(job_id: str, scores: CandidateScoresUpdateRequest, session: UserSession = Depends(get_user_session)):
    """Update candidate scores for the job."""
    search_service = session.init_search_session(job_id)
    job_scores = [
        JobCandidateScore(
            job_id=job_id,
            candidate_id=score_dict["candidate_id"],
            score=score_dict["score"]
        )
        for score_dict in scores.scores
    ]
    search_service.update_candidate_scores(job_scores)
    return {"message": "Candidate scores updated successfully"}


@app.post("/search/{job_id}/candidates/{candidate_id}/score", tags=["Search"], summary="Add or update candidate score")
def add_candidate_score(job_id: str, candidate_id: str, score: CandidateScoreUpdateRequest, session: UserSession = Depends(get_user_session)):
    """Add or update a candidate score."""
    search_service = session.init_search_session(job_id)
    search_service.add_candidate_score(candidate_id, score.score)
    return {"message": "Candidate score updated successfully"}


@app.delete("/search/{job_id}/candidates/{candidate_id}/score", tags=["Search"], summary="Delete candidate score")
def delete_candidate_score(job_id: str, candidate_id: str, session: UserSession = Depends(get_user_session)):
    """Delete a candidate score."""
    search_service = session.init_search_session(job_id)
    search_service.delete_candidate_score(candidate_id)
    return {"message": "Candidate score deleted successfully"}


@app.post("/search/{job_id}/candidates/generate-scores", tags=["Search"], summary="Generate scores for candidates")
def generate_scores(job_id: str, request: GenerateScoresRequest, session: UserSession = Depends(get_user_session)):
    """Generate scores for candidates based on questionnaire."""
    search_service = session.init_search_session(job_id)
    ranked_candidates = search_service.generate_scores(candidate_ids=request.candidate_ids)
    return {"ranked_candidates": ranked_candidates}

