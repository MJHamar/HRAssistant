from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from ..db.model import Candidate, Questionnaire, Document, Job


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

class CandidateListResponse(BaseModel):
    candidates: List[Candidate]

CandidateResponse = Candidate

class CandidateDeleteResponse(BaseModel):
    message: Optional[str]

class CandidateScoreResponse(BaseModel):
    report: Dict[str, Any]


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
    "QuestionnaireResponse",
    "JobListResponse",
    "JobResponse",
    "JobDeleteResponse",
    # Candidates
    "CandidateUploadRequest",
    "CandidateUploadResponse",
    "CandidateListResponse",
    "CandidateResponse",
    "CandidateDeleteResponse",
    "CandidateScoreResponse",
]