from typing import Optional, List, Dict, Any
import uuid

from sqlmodel import SQLModel, Field, Relationship, Column, JSON
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy import Float


class DocumentChunk(SQLModel, table=True):
    __tablename__ = "document_chunks"

    document_id: str = Field(primary_key=True)
    idx: int = Field(primary_key=True)
    metadata: Optional[Dict[str, Any]] = Field(sa_column=Column(JSON), default=None)
    toc_items: Optional[List[Any]] = Field(sa_column=Column(JSON), default=None)
    tables: Optional[List[Any]] = Field(sa_column=Column(JSON), default=None)
    images: Optional[List[Any]] = Field(sa_column=Column(JSON), default=None)
    graphics: Optional[List[Any]] = Field(sa_column=Column(JSON), default=None)
    text: Optional[str] = None
    embedding: Optional[List[float]] = Field(sa_column=Column(ARRAY(Float)), default=None)


class Document(SQLModel, table=True):
    __tablename__ = "documents"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    content: Optional[str] = None
    embedding: Optional[List[float]] = Field(sa_column=Column(ARRAY(Float)), default=None)

    # Relationship to chunks
    chunks: List[DocumentChunk] = Relationship(back_populates=None)


class Job(SQLModel, table=True):
    __tablename__ = "jobs"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    company_name: Optional[str] = None
    job_title: str
    job_description: str


class JobIdealCandidate(SQLModel, table=True):
    __tablename__ = "job_ideal_candidates"

    job_id: str = Field(primary_key=True)
    ideal_candidate_resume: Optional[str] = None


class Candidate(SQLModel, table=True):
    __tablename__ = "candidates"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    candidate_name: Optional[str] = None
    candidate_cv_id: Optional[str] = None  # document_id of the candidate's CV


class QuestionnaireItem(SQLModel):
    criterion: str
    importance: Optional[str] = None


class Questionnaire(SQLModel, table=True):
    __tablename__ = "questionnaires"

    job_id: str = Field(primary_key=True)
    questionnaire: List[QuestionnaireItem] = Field(sa_column=Column(JSON))


class JobCandidateScore(SQLModel, table=True):
    __tablename__ = "job_candidate_scores"

    job_id: str = Field(primary_key=True)
    candidate_id: str = Field(primary_key=True)
    score: float


class CandidateFitness(SQLModel, table=True):
    __tablename__ = "candidate_fitness"

    candidate_id: str = Field(primary_key=True)
    job_id: str = Field(primary_key=True)
    questionnaire_id: str = Field(primary_key=True)
    scores: List[float] = Field(sa_column=Column(JSON))
