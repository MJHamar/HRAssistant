"""
Module for LM-related functionality
"""
from typing import List, Dict, Any

import dspy

from .prompts import ScoreCandidate, MakeQuestionnaire, IdealResumeSignature

make_resume, make_questionnaire, score_candidate = None, None, None

def configure_dspy(provider: str = 'gemini', model: str = 'gemini-2.5-flash', api_key: str = None, **kwargs):
    assert api_key is not None, "API key must be provided"
    lm = dspy.LM(model=f"{provider}/{model}", api_key=api_key, **kwargs)
    dspy.configure(lm=lm)
    return lm

def get_dspy_modules():
    # TODO: incorporate teleprompter, teach dynamically
    global make_resume
    if make_resume is None:
        make_resume = dspy.ChainOfThought(IdealResumeSignature)
    global make_questionnaire
    if make_questionnaire is None:
        make_questionnaire = dspy.ChainOfThought(MakeQuestionnaire)
    global score_candidate
    if score_candidate is None:
        score_candidate = dspy.ChainOfThought(ScoreCandidate)

__all__ = [
    "configure_dspy",
    "make_resume",
    "make_questionnaire",
    "score_candidate"
]