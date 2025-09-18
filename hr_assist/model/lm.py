"""
Module for LM-related functionality
"""
from typing import List, Dict, Any

import dspy

from .prompts import ScoreCandidate, MakeQuestionnaire, IdealResumeSignature


def configure_dspy(provider: str = 'gemini', model: str = 'gemini-2.5-flash', api_key: str = None, **kwargs):
    assert api_key is not None, "API key must be provided"
    lm = dspy.LM(model=f"{provider}/{model}", api_key=api_key, **kwargs)
    dspy.configure(lm=lm)
    return lm

class HrAssistantLM(dspy.Module):
    """
    DSPy Module defining all the prompting classes that the HRManager supports
    """
    def __init__(self, callbacks=None):
        super().__init__(callbacks)
        self._make_resume = dspy.ChainOfThought(IdealResumeSignature)
        self._make_questionnaire = dspy.ChainOfThought(MakeQuestionnaire)
        self._score_candidate = dspy.ChainOfThought(ScoreCandidate)
    
    # _____________________________________________
    # Helpers for building examples with type hints
    # _____________________________________________    
    
    @staticmethod
    def build_resume_example(job_description: str, ideal_candidate_resume: str) -> dspy.Example:
        return dspy.Example(
            inputs={"target_job": job_description},
            outputs={"ideal_candidate_resume": ideal_candidate_resume}
        )
    @staticmethod
    def build_resume_examples(job_descriptions: List[str], ideal_candidate_resumes: List[str]) -> List[dspy.Example]:
        assert len(job_descriptions) == len(ideal_candidate_resumes), "Job descriptions and ideal candidate resumes must have the same length"
        return [HrAssistant.build_resume_example(jd, icr) for jd, icr in zip(job_descriptions, ideal_candidate_resumes)]
    
    @staticmethod
    def build_questionnaire_example(job_description: str, criteria: List[Dict[str, Any]]) -> dspy.Example:
        return dspy.Example(
            inputs={"job_description": job_description},
            outputs={"criteria": criteria}
        )
    @staticmethod
    def build_questionnaire_examples(job_descriptions: List[str], criteria_list: List[List[Dict[str, Any]]]) -> List[dspy.Example]:
        assert len(job_descriptions) == len(criteria_list), "Job descriptions and criteria lists must have the same length"
        return [HrAssistant.build_questionnaire_example(jd, cl) for jd, cl in zip(job_descriptions, criteria_list)]
    
    @staticmethod
    def built_score_example(candidate_cv: str, questionnaire: List[Dict[str, Any]], scores: List[float]) -> dspy.Example:
        return dspy.Example(
            inputs={"candidate_cv": candidate_cv, "questionnaire": questionnaire},
            outputs={"scores": scores}
        )
    @staticmethod
    def build_score_examples(candidate_cvs: List[str], questionnaires: List[List[Dict[str, Any]]], scores_list: List[List[float]]) -> List[dspy.Example]:
        assert len(candidate_cvs) == len(questionnaires) == len(scores_list), "Candidate CVs, questionnaires, and scores lists must have the same length"
        return [HrAssistant.built_score_example(cv, qn, sc) for cv, qn, sc in zip(candidate_cvs, questionnaires, scores_list)]
    
    # _____________________________________________
    # Internal helper methods
    # _____________________________________________
    
    def _build_prompter(self, base_module: dspy.Module, few_shot_examples: List[dspy.Example] = []):
        if few_shot_examples:
            # we construct a few-shot prompt on the fly
            optimizer = dspy.LabeledFewShot(k=len(few_shot_examples))
            prompter = optimizer.compile(base_module, trainset=few_shot_examples, sample=False)
        else:
            prompter = base_module
        return prompter
    
    # _____________________________________________
    # interface methods
    # _____________________________________________
    
    def generate_resume(self, target_job: str, examples: List[dspy.Example] = []) -> str:
        """
        Generate an ideal candidate resume for a given job description
        """
        prompter = self._build_prompter(self._make_resume, few_shot_examples=examples)
        result = prompter(target_job=target_job)
        return result.ideal_candidate_resume
    
    def generate_questionnaire(self, job_description: str, examples: List[dspy.Example] = []) -> List[Dict[str, Any]]:
        """
        Generate a questionnaire based on job description
        """
        prompter = self._build_prompter(self._make_questionnaire, few_shot_examples=examples)
        result = prompter(job_description=job_description)
        return [q.dict() for q in result.criteria]
    
    def generate_candidate_scores(self, candidate_cv: str, questionnaire: List[Dict[str, Any]], examples: List[dspy.Example] = []) -> List[float]:
        """
        Score a candidate based on their CV and a questionnaire
        """
        prompter = self._build_prompter(self._score_candidate, few_shot_examples=examples)
        result = prompter(candidate_cv=candidate_cv, questionnaire=questionnaire)
        return result.scores

__all__ = [
    "configure_dspy",
    "HrAssistantLM",
]