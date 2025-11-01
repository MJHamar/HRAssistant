"""Dataset-specific definition of the RawDataHandler for the Alkalmazas Tesztelo dataset."""

from typing import Generator, Tuple
import csv

from .raw_data import RawDataHandler

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class AlkalmazasTeszteloHandler(RawDataHandler):
    pass