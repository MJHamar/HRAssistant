

from typing import Optional, List, Dict
from hr_assist.search.ranking import BaseRanker, BaseReranker
from ..db.model import Document, Questionnaire
from ..db import BaseDb


class HRSearchPipeline:
    def __init__(
        self,
        db: BaseDb,
        ranker: BaseRanker,
        reranker: Optional[BaseReranker] = None,
        # hyper parameters
        rank_k: int = 500,
        questionnaire: Questionnaire = None
        ):
        self.ranker = ranker
        self.reranker = reranker
        self.db = db
        self.rank_k = rank_k
        self.questionnaire = questionnaire
    
    def search(self, job_description: str) -> list:
        # step 1: retrieve
        results: Document = self.ranker.rank(query=job_description, top_k=self.rank_k)
        
        
        
        results = self.rerank(job_description, results)
        
        return results

    def rank(self, job_description: str) -> List[Document]:
        return self.ranker.rank(query=job_description, top_k=self.rank_k)
    
    def rerank(self, job_description: str, documents: List[Document]) -> List[Document]:
        if self.reranker:
            return self.reranker.rerank(query=job_description, documents=documents)
        return documents