from typing import Optional, List, Dict, Any
import uuid

from pydantic import BaseModel, Field

class DocumentChunk(BaseModel):
    __tablename__ = "document_chunks"
    __primary_key__ = ("document_id", "idx")
    document_id: str
    idx: int
    metadata: Optional[Dict[str, Any]]
    toc_items: Optional[List[Any]]
    tables: Optional[List[Any]]
    images: Optional[List[Any]]
    graphics: Optional[List[Any]]
    text: Optional[str]

class Document(BaseModel):
    __tablename__ = "documents"
    __primary_key__ = ("id",)
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: Optional[str]
    chunks: Optional[List[DocumentChunk]]

class Job(BaseModel):
    __tablename__ = "jobs"
    __primary_key__ = ("id",)
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    company_name: Optional[str]
    job_title: str
    job_description: str

class Candidate(BaseModel):
    __tablename__ = "candidates"
    __primary_key__ = ("id",)
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    candidate_name: Optional[str]
    candidate_cv_id: Optional[str] # document_id of the candidate's CV

class QuestionnaireItem(BaseModel):
    criterion: str
    importance: Optional[str]

class Questionnaire(BaseModel):
    __tablename__ = "questionnaires"
    __primary_key__ = ("job_id",)
    job_id: str
    questionnaire: List[QuestionnaireItem]

class CandidateFitness(BaseModel):
    __tablename__ = "candidate_fitness"
    __primary_key__ = ("candidate_id", "job_id", "questionnaire_id")
    "candidate_fitness - job_id - questionnaire_id is primary key."
    candidate_id: str
    job_id: str
    questionnaire_id: str
    scores: List[float]
