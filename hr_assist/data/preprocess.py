"""
Preprocessing pipeline processing complete databases for training and inference.

This module is intended to process entire databases, which are not standardized into a standardized set of inputs to the HR model.

The final output format is the following:
- A dictionary of job descriptions mapped to unique ids (text)
- A dictionary of candidate resumes mapped to unique ids (text)
- for each job - candidate pair, an overall rating (accepted/ fit/ unfit/ unknown)
    mapped to natural number scores (2/1/0/-1)

# Input Data
- Job descriptions can come as plain text, docx or pdf files
- Candidate resumes can come as plain text, docx or pdf files
- Ratings are xls or csv files in one of the following formats:
    - job_id, candidate_id, rating
    - job_id, candidate_id, fit (yes/no)
    - instead of job_id, there might be separate files for each job
    - instead of candidate_id, candidate name might be used.
- Processing the candidate CVs might not be straightforward either, typically they documents that carry the candidate name, but we cannot expect any specific format (i.e. it could be <first_name> <last_name>, or <last_name>, <first_name>, <first_name>_<last_name>, etc. capitalization might differ, etc.)

## Directory Structure
- We cannot assume much about the directory structure of the different datasets. We will implement this with dedicated RawDataHandler classes.

# Preprocessing Steps
- Instantiate a RawDataHandler for the specific dataset
- Obtain paths to all job description files
- Extract the job descriptions
- Obtain paths to all candidate resume files
- Extract the candidate resumes
- Get the ratings for each job-candidate pair
- Map the ratings to the standardized format
- Save the processed data to a standardized format (jsonl)
"""

import glob
import os
import json
import logging
from typing import Generator, List, Dict, Optional, Tuple, Union
from pathlib import Path
import json
from abc import ABC, abstractmethod

from ..utils.preprocess import convert_to_md

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class RawDataHandler(ABC):
    """Base class for handling raw data from different datasets."""

    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)

    @abstractmethod
    def get_job_descriptions(self) -> Generator[Dict[str, str], None, None]:
        """Extract job descriptions from the dataset.

        Returns:
            A dictionary mapping job IDs to job description texts.
        """
        pass

    @abstractmethod
    def get_candidate_resumes(self) -> Generator[Dict[str, str], None, None]:
        """Extract candidate resumes from the dataset.

        Returns:
            A dictionary mapping candidate IDs to resume texts.
        """
        pass

    @abstractmethod
    def get_ratings(self) -> Generator[Tuple[str, str, str], None, None]:
        """Get ratings for each job-candidate pair.

        Returns:
            A generator yielding tuples of (job_id, candidate_id, rating).
        """
        pass

    ### Common helper methods ###
    def _read_file(self, file_path: Path) -> str:
        """Extract text from a PDF file."""
        try:
            return convert_to_md(file_path)
        except Exception as e:
            logger.error(f"Error reading PDF file {file_path}: {e}")
            return ""

    def _find_file_by_name(self, root_dir: Union[str, Path], name: str, recursive: bool = False, extensions: Optional[List[str]]=None) -> Optional[Path]:
        """Find a file by name in the data directory, optionally filtering by extensions."""
        root_dir = Path(root_dir)
        if not root_dir.exists() or not root_dir.is_dir():
            logger.error(f"Directory {root_dir} does not exist or is not a directory.")
            raise ValueError(f"Directory {root_dir} does not exist or is not a directory.")
        if extensions is None:
            extensions = ['.pdf', '.docx', '.txt']

        matches = []
        for path in glob(root_dir / ('**' if recursive else '') / '*'):
            p = Path(path)
            if p.is_file() and p.suffix.lower() in extensions:
                stm = p.stem.lower()
                words = stm.split()
                flower = name.lower()
                if all(flower.find(w) != -1 for w in words):
                    matches.append(p)
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            raise ValueError(f"Multiple files found matching {name} in {root_dir}: {matches}")
        logger.warning(f"File named {name} not found in {root_dir}.")
        return None




