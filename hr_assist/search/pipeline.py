from typing import Optional, List, Dict, Type

from dspy import Module, Example
from dspy.teleprompt import BootstrapFewShot, LabeledFewShot

from ..db.base import BaseDb
from ..db.model import (
    Document, Job, Candidate, Questionnaire, QuestionnaireItem,
    CandidateFitness)
from ..model.prompts import Question, MakeQuestionnaire, ScoreCandidate, IdealResumeSignature
from .ranking import BaseRanker, BaseReranker, ScoringReranker, PgRanker
from ..model.embed import PreTrainedEmbedder

class HRSearchPipeline:
    def __init__(
            self,
            db: BaseDb,
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
        questionnaire = self._db.query(
            Questionnaire.__tablename__, {"job_id": job.id}
        )
        self._questionnaire = questionnaire[0] if len(questionnaire) else Questionnaire(
            job_id=job.id,
            questionnaire=[]
        )
        self._ideal_candidate = None
        self._candidates = None # top-k documents from the ranker
        self._candidate_fitness = None # fitness scores from the reranker

    @property
    def db(self) -> BaseDb:
        return self._db

    @property
    def ranker(self) -> Optional[BaseRanker]:
        if self._ranker_cls is None:
            return None
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
        self.db.modify(
            table=Questionnaire.__tablename__,
            key=self._questionnaire.job_id,
            changes=dict(**self._questionnaire.model_dump()))
        return self._questionnaire

    @property
    def questionnaire(self) -> Optional[Questionnaire]:
        return self._questionnaire

    def set_questionnaire(self, questionnaire: Questionnaire) -> None:
        assert questionnaire.job_id == self._job.id, "Questionnaire job_id must match the job id"
        self.db.modify(
            table=Questionnaire.__tablename__,
            key=questionnaire.job_id,
            changes=dict(**questionnaire.model_dump()))
        self._questionnaire = questionnaire

    def delete_questionnaire(self) -> None:
        self._questionnaire.questionnaire = []
        self.db.modify(
            table=Questionnaire.__tablename__,
            key=self._questionnaire.job_id,
            changes=dict(**self._questionnaire.model_dump()))

    def add_questionnaire_item(self, item: QuestionnaireItem) -> None:
        if self._questionnaire is None:
            self._questionnaire = Questionnaire(
                job_id=self._job.id,
                questionnaire=[item]
            )
        else:
            self._questionnaire.questionnaire.append(item)
        self.db.modify(
            table=Questionnaire.__tablename__,
            key=self._questionnaire.job_id,
            changes=dict(**self._questionnaire.model_dump())
        )

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
        self.db.modify(
            table=Questionnaire.__tablename__,
            key=self._questionnaire.job_id,
            changes=dict(**self._questionnaire.model_dump())
        )