"""
Run evaluation on a given dataset and HR Assistant model.
"""
from torch import Module
import dspy

from .search.ranking import ScoringReranker, PgRanker
from .data.dataset import HRDataset

