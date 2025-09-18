"""
Collection of DSPy.Signatures for prompting LLMs.
"""
from typing import Literal, List
import dspy 
import pydantic

class IdealResumeSignature(dspy.Signature):
    """
    Signature for generating an ideal candidate resume for a given job description
    
    dspy version of the prompt template in ConFit-v2/src/utils/convert_by_llm.py
    """
    target_job: str = dspy.InputField(desc="The target job description to generate an ideal candidate resume for")
    ideal_candidate_resume: str = dspy.OutputField(desc="The generated ideal candidate resume that matches the target job description")

class Question(pydantic.BaseModel):
    criterion: str = dspy.OutputField(desc="A single criterion in a questionnaire")
    importance: Literal["low", "medium", "high"] = dspy.OutputField(desc="How important it is that the candidate satisfies this criterion")

class MakeQuestionnaire(dspy.Signature):
    """
    Signature for generating a questionnaire based on job description
    """
    job_description: str = dspy.InputField(desc="Job description to base the questionnaire on")
    criteria: List[Question] = dspy.OutputField(desc="List of criteria for the questionnaire")

class ScoreCandidate(dspy.Signature):
    """
    Signature for scoring a candidate based on their CV and a questionnaire
    """
    candidate_cv: str = dspy.InputField(desc="The CV of the candidate to be scored")
    questionnaire: List[Question] = dspy.InputField(desc="The questionnaire to score the candidate against")
    scores: List[float] = dspy.OutputField(desc="List of scores for each question in the questionnaire")

