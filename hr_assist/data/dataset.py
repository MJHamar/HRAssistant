"""
Dataset definition for HR Assist project.

Defines the HRDataset class, which encapsulates job descriptions,
candidate resumes, and their associated ratings.

Can be loaded from a JSONL file or constructed manually.
"""
from typing import List, Dict, Optional, Tuple, Union

from pathlib import Path
import json
from torch.utils.data import Dataset

class HRDataset(Dataset):
    def __init__(
        self,
        jobs: Dict[str, str],
        candidates: Dict[str, str],
        ratings: List[Tuple[str, str, int]],
    ):
        self._validate_input(jobs, candidates, ratings)
        self.jobs = jobs
        self.candidates = candidates
        self.ratings = ratings

    def _validate_input(self, jobs, candidates, ratings):
        assert len(jobs) > 0, "Jobs list cannot be empty."
        assert len(candidates) > 0, "Candidates list cannot be empty."
        assert len(ratings) > 0, "Ratings list cannot be empty."
        assert len(jobs)*len(candidates) == len(ratings), "Number of ratings must equal number of jobs times number of candidates."

        assert isinstance(jobs, dict), "Jobs must be a dictionary mapping job IDs to descriptions."
        assert isinstance(candidates, dict), "Candidates must be a dictionary mapping candidate IDs to resumes."
        assert isinstance(ratings, list), "Ratings must be a list of (job_id, candidate_id, rating) tuples."
        for r in ratings:
            assert isinstance(r, tuple) and len(r) == 3, "Each rating must be a tuple of (job_id, candidate_id, rating)."
            job_id, candidate_id, rating = r
            assert job_id in jobs, f"Job ID {job_id} in ratings not found in jobs."
            assert candidate_id in candidates, f"Candidate ID {candidate_id} in ratings not found in candidates."
            assert isinstance(rating, int) and 1 <= rating <= 5, "Rating must be an integer between 1 and 5."
            assert rating in [-1, 0, 1, 2], "Rating must be one of the following values: -1 (unkown), 0 (unfit), 1 (fit), 2 (accepted)."

    @classmethod
    def from_jsonl(cls, file_path: Union[str, Path]) -> "HRDataset":
        """Load dataset from a JSONL file.

        Each line in the file should be a JSON object with keys:
        - "jobs": list of job dicts with "id" and "description"
        - "candidates": list of candidate dicts with "id" and "resume"
        - "ratings": list of tuples (job_id, candidate_id, rating)

        Args:
            file_path: Path to the JSONL file.

        Returns:
            An instance of HRDataset.
        """
        file_path = Path(file_path)
        jobs = []
        candidates = []
        ratings = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                jobs.extend(data.get("jobs", []))
                candidates.extend(data.get("candidates", []))
                ratings.extend(data.get("ratings", []))

        return cls(jobs=jobs, candidates=candidates, ratings=ratings)

    def to_jsonl(self, file_path: Union[str, Path]) -> None:
        """Save dataset to a JSONL file.

        Args:
            file_path: Path to save the JSONL file.
        """
        file_path = Path(file_path)
        with open(file_path, "w", encoding="utf-8") as f:
            data = {
                "jobs": [{"id": k, "description": v} for k, v in self.jobs.items()],
                "candidates": [{"id": k, "resume": v} for k, v in self.candidates.items()],
                "ratings": self.ratings,
            }
            json.dump(data, fp=f)
            f.write("\n")

    def add_candidate(self, id_: str, resume: str) -> None:
        if id_ in self.candidates:
            raise ValueError(f"Candidate with id {id_} already exists.")
        self.candidates[id_] = resume

    def add_job(self, id_: str, description: str) -> None:
        if id_ in self.jobs:
            raise ValueError(f"Job with id {id_} already exists.")
        self.jobs[id_] = description

    def add_rating(self, job_id: str, candidate_id: str, rating: int) -> None:
        if job_id not in self.jobs:
            raise ValueError(f"Job with id {job_id} does not exist.")
        if candidate_id not in self.candidates:
            raise ValueError(f"Candidate with id {candidate_id} does not exist.")
        if rating not in [1, 2, 0, -1]:
            raise ValueError("Rating must be one of the following values: -1 (unknown), 0 (unfit), 1 (fit), 2 (accepted).")
        self.ratings.append((job_id, candidate_id, rating))

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx: int) -> Tuple[str, str, int]:
        job_id, candidate_id, rating = self.ratings[idx]
        job = self.jobs[job_id]
        candidate = self.candidates[candidate_id]
        return job, candidate, rating
