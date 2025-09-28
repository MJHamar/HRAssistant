"""
Interface module for database operations in the HR Assist application.
Supports SQLAlchemy with SQLModel for database operations.
"""

from .model import (
    Document,
    DocumentChunk,
    Job,
    JobIdealCandidate,
    Candidate,
    QuestionnaireItem,
    Questionnaire,
    JobCandidateScore,
    CandidateFitness,
)
from .database import get_session, get_session_sync, create_tables, engine
from .similarity import SimilarityFunction, sim_

__all__ = [
    "Document",
    "DocumentChunk",
    "Job",
    "JobIdealCandidate",
    "Candidate",
    "QuestionnaireItem",
    "Questionnaire",
    "JobCandidateScore",
    "CandidateFitness",
    "get_session",
    "get_session_sync",
    "create_tables",
    "engine",
    "SimilarityFunction",
    "sim_",
]