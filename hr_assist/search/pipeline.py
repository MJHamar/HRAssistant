from typing import Optional, List, Dict, Type, Tuple, Union

from dspy import Module, Example
from dspy.teleprompt import BootstrapFewShot, LabeledFewShot
from sqlalchemy.orm import Session
from sqlalchemy import select, delete

from ..db.model import (
    Document, Job, Candidate, Questionnaire, QuestionnaireItem,
    CandidateFitness, JobCandidateScore, JobIdealCandidate)
from ..model.prompts import Question, MakeQuestionnaire, ScoreCandidate, IdealResumeSignature
from .ranking import BaseRanker, BaseReranker, ScoringReranker, PgRanker
from ..model.embed import PreTrainedEmbedder

class HRSearchPipeline:
    def __init__(
            self,
            db: Session,
            embedder: PreTrainedEmbedder,
            ic_module: Module,  # for generating an ideal candidate
            q_module: Module,  # for generating questionnaires
            s_module: Module,  # for scoring candidates
            job: Job,
            ranker_cls: Optional[Type[BaseRanker]] = PgRanker,
            reranker_cls: Optional[Type[BaseReranker]] = ScoringReranker,
            document_table: Optional[str] = "documents",
            candidate_table: Optional[str] = "candidates",
            similarity_metric: Optional[str] = "cosine",
            num_questions: Optional[int] = 10,
            rank_k: Optional[int] = 100,
            ) -> None:

        # Passed attributes
        self._db = db
        self._embedder = embedder
        self._ic_module = ic_module
        self._q_module = q_module
        self._s_module = s_module
        self._job = job
        self._ranker_cls = ranker_cls
        self._reranker_cls = reranker_cls

        self._document_table = document_table
        self._candidate_table = candidate_table
        self._similarity_metric = similarity_metric
        self._num_questions = num_questions
        self._rank_k = rank_k

        # Derived attributes
        self._questionnaire = self._init_questionnaire()
        self._ideal_candidate = self._init_ideal_candidate()
        self._candidates, self._candidate_scores = self._init_candidates_scores()
        self._candidate_fitness = self._init_candidate_fitness()

    def _init_questionnaire(self):
        # Query for existing questionnaire
        stmt = select(Questionnaire).where(Questionnaire.job_id == self._job.id)
        questionnaire = self._db.exec(stmt).first()

        if questionnaire is None:
            # Create new questionnaire
            questionnaire = Questionnaire(
                job_id=self._job.id,
                questionnaire=[]
            )
            self._db.add(questionnaire)
            self._db.commit()

        return questionnaire

    def _init_ideal_candidate(self):
        # Query for existing ideal candidate
        stmt = select(JobIdealCandidate).where(JobIdealCandidate.job_id == self._job.id)
        ideal_candidate = self._db.exec(stmt).first()

        if ideal_candidate is None:
            # Create new ideal candidate
            ideal_candidate = JobIdealCandidate(
                job_id=self._job.id,
                ideal_candidate_resume=""
            )
            self._db.add(ideal_candidate)
            self._db.commit()

        return ideal_candidate

    def _init_candidates_scores(self):
        # Query candidate scores for this job
        stmt = select(JobCandidateScore).where(
            JobCandidateScore.job_id == self._job.id
        ).order_by(JobCandidateScore.score.desc())
        candidate_scores = list(self._db.exec(stmt))

        if candidate_scores:
            # Get the actual candidate objects from the scores
            candidate_ids = [score.candidate_id for score in candidate_scores]
            candidates = []
            for candidate_id in candidate_ids:
                stmt = select(Candidate).where(Candidate.id == candidate_id)
                candidate = self._db.exec(stmt).first()
                if candidate:
                    candidates.append(candidate)
        else:
            candidates = []
            candidate_scores = []

        return candidates, candidate_scores

    def _init_candidate_fitness(self):
        # Query candidate fitness for this job
        stmt = select(CandidateFitness).where(CandidateFitness.job_id == self._job.id)
        candidate_fitness = list(self._db.exec(stmt))
        return candidate_fitness

    @property
    def db(self) -> Session:
        return self._db

    @property
    def ranker(self) -> BaseRanker:
        assert self._ranker_cls is not None, "Ranker class is not specified."
        return self._ranker_cls(
            db=self._db,
            embedding_fn=self._embedder,
            table=self._document_table,
            similarity_metric=self._similarity_metric
        )

    @staticmethod
    def make_questionnaire_example(jd: str, questions: List[QuestionnaireItem]) -> Example[Questionnaire]:
        # first convert questions to prompts.Question
        questions = [Question(criterion=q.criterion, importance=q.importance) for q in questions]
        # then make an example
        return Example(
            job_description=jd,
            questions=questions
        )

    def generate_questionnaire(
            self,
            use_existing_questions: bool = True,
            precise_num_questions: bool = False
        ) -> Questionnaire:
        # if the questionnaire is already specified, do nothing
        if len(self._questionnaire.questionnaire) >= self._num_questions:
            return self._questionnaire
        # if it is partly speficied, use the existing one as a few-shot example
        if use_existing_questions and len(self._questionnaire.questionnaire) > 0:
            prompter = LabeledFewShot().compile(
                self._q_module,
                trainset=HRSearchPipeline.make_questionnaire_example(self._job.job_description, self._questionnaire.questionnaire)
            )
        else:
            prompter = self._q_module
        # generate the questionnaire
        #TODO: we might want to specify how many questions to generate
        result = prompter(job_description=self._job.job_description)
        # make questions unique
        seen = {
            q.criterion for q in self._questionnaire.questionnaire
            } or set()
        unique_questions = [
            Question(criterion=q.criterion, importance=q.importance)
            for q in self._questionnaire.questionnaire or []]
        for q in result.criteria:
            if q.criterion not in seen:
                unique_questions.append(q)
                seen.add(q.criterion)
        assert not precise_num_questions or self._num_questions is not None, "If precise_num_questions is True, num_questions must be specified"
        assert not precise_num_questions or len(unique_questions) >= self._num_questions, "If precise_num_questions is True, there must be at least num_questions unique questions."
        # trim to num_questions if there are more than num_questions
        # we order by importance and remove the least important ones first

        if precise_num_questions and len(unique_questions) > self._num_questions:
            importance_order = {"high": 2, "medium": 1, "low": 0}
            unique_questions.sort(key=lambda q: importance_order[q.importance], reverse=True)
            unique_questions = unique_questions[:self._num_questions]
        # convert back to db model
        self._questionnaire = Questionnaire(
            job_id=self._job.id,
            questionnaire=[
                QuestionnaireItem(criterion=q.criterion, importance=q.importance)
                for q in unique_questions
            ]
        )
        # update the database and the internal state
        self._db.merge(self._questionnaire)
        self._db.commit()
        return self._questionnaire

    @property
    def questionnaire(self) -> Optional[Questionnaire]:
        return self._questionnaire

    def set_questionnaire(self, questionnaire: Questionnaire) -> None:
        assert questionnaire.job_id == self._job.id, "Questionnaire job_id must match the job id"
        self._db.merge(questionnaire)
        self._db.commit()
        self._questionnaire = questionnaire

    def delete_questionnaire(self) -> None:
        self._questionnaire.questionnaire = []
        self._db.merge(self._questionnaire)
        self._db.commit()

    def add_questionnaire_item(self, item: QuestionnaireItem) -> None:
        if self._questionnaire is None:
            self._questionnaire = Questionnaire(
                job_id=self._job.id,
                questionnaire=[item]
            )
        else:
            self._questionnaire.questionnaire.append(item)
        self._db.merge(self._questionnaire)
        self._db.commit()

    def remove_questionnaire_item(self, criterion: Optional[str] = None, index: Optional[int] = None) -> None:
        assert (criterion is not None) != (index is not None), "Either criterion or index must be specified, but not both."
        if criterion is not None:
            self._questionnaire.questionnaire = [
                q for q in self._questionnaire.questionnaire
                if q.criterion != criterion
            ]
        else:
            assert 0 <= index < len(self._questionnaire.questionnaire), "Index out of range."
            self._questionnaire.questionnaire.pop(index)
        self._db.merge(self._questionnaire)
        self._db.commit()

    def generate_ideal_candidate(self) -> str:
        """
        Generate an ideal candidate resume for the job description.

        Override the existing ideal candidate if any.
        """
        # TODO: we COULD prompt this better with few-shot examples
        # however, it is arguably out of scope for this per-JD pipeline.
        # BootstrapFewShot examples should be given at another level.
        result = self._ic_module(target_job=self._job.job_description)
        ideal_candidate = result.ideal_candidate_resume
        self._ideal_candidate = JobIdealCandidate(
            job_id=self._job.id,
            ideal_candidate_resume=ideal_candidate
        )
        self._db.add(self._ideal_candidate)
        self._db.commit()
        return self._ideal_candidate

    @property
    def ideal_candidate(self) -> Optional[str]:
        return self._ideal_candidate

    def set_ideal_candidate(self, ideal_candidate: str) -> None:
        assert ideal_candidate is not None and len(ideal_candidate) > 0, "Ideal candidate resume cannot be empty."
        self._ideal_candidate.ideal_candidate_resume = ideal_candidate
        self._db.merge(self._ideal_candidate)
        self._db.commit()

    def delete_ideal_candidate(self) -> None:
        if self._ideal_candidate:
            stmt = delete(JobIdealCandidate).where(JobIdealCandidate.job_id == self._job.id)
            self._db.exec(stmt)
            self._db.commit()
        self._ideal_candidate = self._init_ideal_candidate()

    def rank_candidates(self, top_k: Optional[int] = None) -> List[Candidate]:
        ranked_docs = self.ranker.rank(
            query=self._job.job_description,
            top_k=top_k or self._rank_k
        )
        ranked_candidates = []
        for doc, score in ranked_docs:
            # retrieve the corresponding candidate
            stmt = select(Candidate).where(Candidate.candidate_cv_id == doc.id)
            candidate = self._db.exec(stmt).first()
            if candidate:
                ranked_candidates.append((candidate, score))

        # remove all existing candidate scores for this job
        stmt = delete(JobCandidateScore).where(JobCandidateScore.job_id == self._job.id)
        self._db.exec(stmt)
        self._db.commit()

        # add the new scores
        new_scores = [
            JobCandidateScore(
                job_id=self._job.id,
                candidate_id=candidate.id,
                score=score
            )
            for candidate, score in ranked_candidates
        ]
        self.update_candidate_scores(new_scores)

        return [rc[0] for rc in ranked_candidates]

    @property
    def candidate_scores(self) -> List[JobCandidateScore]:
        """Get the current candidate scores for this job."""
        return self._candidate_scores

    def update_candidate_scores(self, scores: List[JobCandidateScore]) -> None:
        """Update candidate scores in the database and internal state."""
        for score in scores:
            self._db.merge(score)
        self._db.commit()
        self._candidate_scores = scores

    def add_candidate_score(self, candidate_id: str, score: float) -> None:
        """Add or update a single candidate score."""
        job_score = JobCandidateScore(
            job_id=self._job.id,
            candidate_id=candidate_id,
            score=score
        )
        self._db.merge(job_score)
        self._db.commit()

        # Update internal state
        existing_idx = None
        for i, existing_score in enumerate(self._candidate_scores):
            if existing_score.candidate_id == candidate_id:
                existing_idx = i
                break

        if existing_idx is not None:
            self._candidate_scores[existing_idx] = job_score
        else:
            self._candidate_scores.append(job_score)

        # Re-sort by score descending
        self._candidate_scores.sort(key=lambda x: x.score, reverse=True)

    def delete_candidate_score(self, candidate_id: str) -> None:
        """Delete a candidate score."""
        stmt = delete(JobCandidateScore).where(
            JobCandidateScore.job_id == self._job.id,
            JobCandidateScore.candidate_id == candidate_id
        )
        self._db.exec(stmt)
        self._db.commit()

        # Update internal state
        self._candidate_scores = [
            score for score in self._candidate_scores
            if score.candidate_id != candidate_id
        ]

    def generate_scores(self, candidate_ids: Optional[Union[str, List[str]]] = None) -> None:
        """
        Generate scores for candidates based on the questionnaire.

        If candidate_ids is None, score all candidates with existing scores.
        """
        if self._questionnaire is None or len(self._questionnaire.questionnaire) == 0:
            raise ValueError("Questionnaire is not set or empty. Cannot generate scores.")

        # retrieve the candidates to score
        if candidate_ids is None:
            candidates = self._candidates
        elif isinstance(candidate_ids, str):
            candidates = [self._db.get_candidate(candidate_ids)]
        else:
            candidates = [self._db.get_candidate(cid) for cid in candidate_ids]

        # TODO instantiate and use the QuestionnaireScorer class here

