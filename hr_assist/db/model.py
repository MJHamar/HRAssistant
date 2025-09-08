from typing import Optional, List, Dict, Any

from pydantic import BaseModel

class DocumentChunk(BaseModel):
    document_id: str
    idx: int
    metadata: Optional[Dict[str, Any]]
    toc_items: Optional[List[Any]]
    tables: Optional[List[Any]]
    images: Optional[List[Any]]
    graphics: Optional[List[Any]]
    text: Optional[str]

class Document(BaseModel):
    id: str
    content: Optional[str]
    chunks: Optional[List[DocumentChunk]]

class Job(BaseModel):
    id: str
    company_name: Optional[str]
    job_title: str
    job_description: str

class Candidate(BaseModel):
    id: str
    candidate_name: Optional[str]
    candidate_cv_id: Optional[str] # document_id of the candidate's CV

class QuestionnaireItem(BaseModel):
    criterion: str
    importance: Optional[str]

class Questionnaire(BaseModel):
    id: str
    job_id: str
    questionnaire: List[QuestionnaireItem]

class CandidateFitness(BaseModel):
    "candidate_fitness - job_id - questionnaire_id is primary key."
    candidate_id: str
    job_id: str
    questionnaire_id: str
    scores: List[float]
