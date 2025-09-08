from fastapi import Cookie, FastAPI, Header, Path, HTTPException, Depends, Request, Response, UploadFile, File, Form
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

from hr_assist.api.model import (
    DocumentUploadRequest,
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentResponse,
    DocumentDeleteResponse,
    JobUploadRequest,
    JobUploadResponse,
    QuestionnaireResponse,
    JobListResponse,
    JobResponse,
    JobDeleteResponse,
    CandidateUploadRequest,
    CandidateUploadResponse,
    CandidateListResponse,
    CandidateResponse,
    CandidateDeleteResponse,
    CandidateScoreResponse,
)

from .. import __version__
from .service import HRService

app = FastAPI(title="HR Assistant API", version=__version__)

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
    service: HRService = Depends(),
):
    content_bytes = await file.read()
    name = document_name or file.filename
    document_id = service.upload_document(name, content_bytes)
    return {"document_id": document_id}

@app.get("/documents", tags=["Documents"], summary="List all documents", response_model=DocumentListResponse)
def list_documents(service: HRService = Depends()):
    documents = service.list_documents()
    return {
        "documents": documents
    }

@app.get("/documents/{document_id}", tags=["Documents"], summary="Retrieve a document", response_model=DocumentResponse)
def get_document(document_id: str, service: HRService = Depends()):
    contents, chunks, metadata = service.get_document(document_id)
    if not contents:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "document_id": document_id,
        "contents": contents,
        "chunks": chunks,
        "metadata": metadata
    }

@app.delete("/documents/{document_id}", tags=["Documents"], summary="Delete a document", response_model=DocumentDeleteResponse)
def delete_document(document_id: str, service: HRService = Depends()):
    success = service.delete_document(document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Document deleted successfully"}

@app.post("/jobs", tags=["Jobs"], summary="Upload a new job description", response_model=JobUploadResponse)
def upload_job_description(request: JobUploadRequest, service: HRService = Depends()):
    job_id = service.upload_job_description(request.job_title, request.job_description, request.company_name)
    return {
        "job_id": job_id
    }

@app.get("/jobs/{job_id}/questionnaire", tags=["Jobs"], summary="Generate a questionnaire for a job", response_model=QuestionnaireResponse)
def generate_questionnaire(job_id: str, service: HRService = Depends()):
    questionnaire = service.generate_questionnaire(job_id)
    if not questionnaire:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job_id,
        "questionnaire": questionnaire
    }

@app.get("/jobs", tags=["Jobs"], summary="List all jobs", response_model=JobListResponse)
def list_jobs(service: HRService = Depends()):
    jobs = service.list_jobs()
    return {
        "jobs": jobs
    }

@app.get("/jobs/{job_id}", tags=["Jobs"], summary="Retrieve a job description", response_model=JobResponse)
def get_job(job_id: str, service: HRService = Depends()):
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.delete("/jobs/{job_id}", tags=["Jobs"], summary="Delete a job description", response_model=JobDeleteResponse)
def delete_job(job_id: str, service: HRService = Depends()):
    success = service.delete_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"message": "Job deleted successfully"}

@app.post("/candidates", tags=["Candidates"], summary="Upload a new candidate CV", response_model=CandidateUploadResponse)
def upload_candidate(request: CandidateUploadRequest, service: HRService = Depends()):
    candidate_id = service.upload_candidate(request.candidate_name, request.candidate_cv_id)
    return {
        "candidate_id": candidate_id
    }

@app.get("/candidates", tags=["Candidates"], summary="List all candidates", response_model=CandidateListResponse)
def list_candidates(service: HRService = Depends()):
    candidates = service.list_candidates()
    return {
        "candidates": candidates
    }

@app.get("/candidates/{candidate_id}", tags=["Candidates"], summary="Retrieve a candidate CV", response_model=CandidateResponse)
def get_candidate(candidate_id: str, service: HRService = Depends()):
    candidate = service.get_candidate(candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return candidate

@app.delete("/candidates/{candidate_id}", tags=["Candidates"], summary="Delete a candidate CV", response_model=CandidateDeleteResponse)
def delete_candidate(candidate_id: str, service: HRService = Depends()):
    success = service.delete_candidate(candidate_id)
    if not success:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return {"message": "Candidate deleted successfully"}

@app.post("/candidates/{candidate_id}/score/{job_id}", tags=["Candidates"], summary="Score a candidate against a job", response_model=CandidateScoreResponse)
def score_candidate(candidate_id: str, job_id: str, service: HRService = Depends()):
    report = service.score_candidate(candidate_id, job_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Candidate or Job not found")
    return report

