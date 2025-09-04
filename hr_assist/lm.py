"""
Module for LM-related functionality
"""
import dspy

from .prompts import ScoreCandidate, MakeQuestionnaire

make_questionnaire = dspy.ChainOfThought(MakeQuestionnaire)
score_candidate = dspy.ChainOfThought(ScoreCandidate)

def configure_dspy(provider: str = 'gemini', model: str = 'gemini-2.5-flash', api_key: str = None, **kwargs):
    assert api_key is not None, "API key must be provided"
    lm = dspy.LM(model=f"{provider}/{model}", api_key=api_key, **kwargs)
    dspy.configure(lm=lm)
    return lm

__all__ = [
    "make_questionnaire",
    "score_candidate",
    "configure_dspy"
]