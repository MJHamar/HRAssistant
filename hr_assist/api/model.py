from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from ..db.model import Candidate, Questionnaire, Document, Job


# Documents
class DocumentUploadRequest(BaseModel):
    document_text: str
    document_name: str

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
    candidate_cv: str

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

class DocumentUploadResponse(BaseModel):
    document_id: str
    status: str

class GenericRequest(BaseModel):
    payload: dict

class GenericResponse(BaseModel):
    result: dict
    status: str