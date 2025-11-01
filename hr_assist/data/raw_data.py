
import glob
import os
import json
import logging
from typing import Generator, List, Dict, Optional, Tuple, Union
from pathlib import Path
import json
from abc import ABC, abstractmethod
from unicode import unicode

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
    def get_ratings(self) -> Generator[Tuple[str, str, int], None, None]:
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
                stm = unicode(p.stem.lower()) # normalize to unicode for matching
                words = stm.split()
                flower = unicode(name.lower()) # normalize to unicode for matching
                if all(flower.find(w) != -1 for w in words):
                    matches.append(p)
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            raise ValueError(f"Multiple files found matching {name} in {root_dir}: {matches}")
        logger.warning(f"File named {name} not found in {root_dir}.")
        return None




