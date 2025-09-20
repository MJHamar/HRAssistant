

from typing import Optional
from hr_assist.search.ranking import BaseRanker, BaseReranker
from ..db import BaseDb


class HRSearchPipeline:
    def __init__(
        self,
        db: BaseDb,
        ranker: BaseRanker,
        reranker: Optional[BaseReranker] = None,
        # hyper parameters
        rank_k: int = 500,
        ):
        self.ranker = ranker
        self.reranker = reranker
        self.db = db
    
    def search(self, job_description: str) -> list:
        # step 1: retrieve
        results = self.ranker.rank(query=job_description, )