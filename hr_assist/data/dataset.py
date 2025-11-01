"""
Dataset definition for HR Assist project.

Defines the HRDataset class, which encapsulates job descriptions,
candidate resumes, and their associated ratings.

Can be loaded from a JSONL file or constructed manually.
"""
from typing import List, Dict, Optional, Tuple, Union

from uuid import uuid4 as uuid
from pathlib import Path
import json
from torch.utils.data import Dataset
from collections import defaultdict

class HRDataset(Dataset):
    def __init__(
        self,
        jobs: Dict[str, str],
        candidates: Dict[str, str],
        ratings: Union[List[Tuple[str, str, int]], Dict[Tuple[str, str], int]] = None,
    ):
        """Create an HRDataset.

        Args:
            jobs: mapping job_id -> description
            candidates: mapping candidate_id -> resume
            ratings: either a list of (job_id, candidate_id, rating) tuples or a
                dict keyed by (job_id, candidate_id) -> rating. If omitted, an
                empty (sparse) ratings map is created where missing pairs default
                to -1.
        """
        # validate basic inputs
        assert isinstance(jobs, dict), "Jobs must be a dict"
        assert isinstance(candidates, dict), "Candidates must be a dict"

        # store jobs/candidates
        self.jobs = dict(jobs)
        self.candidates = dict(candidates)

        # pre-sort for deterministic indexing.
        self.jobs_list = sorted(self.jobs.keys())
        self.candidates_list = sorted(self.candidates.keys())

        # ratings stored sparsely as dict keyed by (job_id, candidate_id)
        # default value for missing entries is -1 (unknown)
        self.ratings = defaultdict(lambda: -1)

        if ratings is None:
            return

        # accept either list-of-tuples or mapping
        if isinstance(ratings, dict):
            for (job_id, cand_id), rating in ratings.items():
                self._safe_set_rating(job_id, cand_id, rating)
        else:
            # assume iterable of triples
            for triple in ratings:
                assert isinstance(triple, (list, tuple)) and len(triple) == 3, "Each rating must be a (job_id, candidate_id, rating) triple"
                assert isinstance(triple[0], str) and isinstance(triple[1], str), "job_id and candidate_id must be strings"
                assert isinstance(triple[2], int), "rating must be an integer"
                job_id, cand_id, rating = triple
                self._safe_set_rating(job_id, cand_id, rating)

    def _safe_set_rating(self, job_id: str, cand_id: str, rating: int) -> None:
        if job_id not in self.jobs:
            raise ValueError(f"Job with id {job_id} does not exist.")
        if cand_id not in self.candidates:
            raise ValueError(f"Candidate with id {cand_id} does not exist.")
        if rating not in [-1, 0, 1, 2]:
            raise ValueError("Rating must be one of the following values: -1 (unknown), 0 (unfit), 1 (fit), 2 (accepted).")
        self.ratings[(job_id, cand_id)] = rating

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
        jobs: Dict[str, str] = {}
        candidates: Dict[str, str] = {}
        ratings: List[Tuple[str, str, int]] = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                for j in data.get("jobs", []):
                    # expect {"id": ..., "description": ...}
                    jobs[j["id"]] = j.get("description", "")
                for c in data.get("candidates", []):
                    candidates[c["id"]] = c.get("resume", "")
                for r in data.get("ratings", []):
                    # ratings expected as triples [job_id, candidate_id, rating] or tuples
                    if isinstance(r, (list, tuple)) and len(r) == 3:
                        ratings.append((r[0], r[1], r[2]))

        return cls(jobs=jobs, candidates=candidates, ratings=ratings)

    def to_jsonl(self, file_path: Union[str, Path]) -> None:
        """Save dataset to a JSONL file.

        Args:
            file_path: Path to save the JSONL file.
        """
        file_path = Path(file_path)
        with open(file_path, "w", encoding="utf-8") as f:
            # write sparse ratings as list of triples for portability
            ratings_list = [
                [job_id, cand_id, rating]
                for (job_id, cand_id), rating in self.ratings.items()
                if rating != -1
            ]
            data = {
                "jobs": [{"id": k, "description": v} for k, v in self.jobs.items()],
                "candidates": [{"id": k, "resume": v} for k, v in self.candidates.items()],
                "ratings": ratings_list,
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
        self._safe_set_rating(job_id, candidate_id, rating)

    def __len__(self) -> int:
        # virtual full cartesian size (deterministic indexing over jobs x candidates)
        return len(self.jobs) * len(self.candidates)

    def __getitem__(self, idx: int) -> Tuple[str, str, int]:
        # Deterministic mapping from linear index to (job, candidate):
        # use sorted key lists for reproducible ordering across runs.
        J = len(self.jobs)
        C = len(self.candidates)
        total = J * C
        if total == 0:
            raise IndexError("Empty dataset")
        if not (0 <= idx < total):
            raise IndexError("Index out of range")


        i = idx // C
        j = idx % C

        job_id = self.jobs_list[i]
        cand_id = self.candidates_list[j]

        job = self.jobs[job_id]
        candidate = self.candidates[cand_id]
        rating = self.ratings.get((job_id, cand_id), -1)
        return job, candidate, rating

    def merge(self, other: "HRDataset") -> "HRDataset":
        """Merge another HRDataset into this one.

        Args:
            other: Another HRDataset instance.

        Returns:
            A new HRDataset instance containing data from both datasets.
        """
        # Merge jobs and candidates deterministically.
        merged_jobs = dict(self.jobs)
        merged_cands = dict(self.candidates)

        # mappings for other's ids -> merged ids
        job_map: Dict[str, str] = {}
        cand_map: Dict[str, str] = {}

        # Merge jobs from other
        for oj_id, oj_desc in other.jobs.items():
            if oj_id not in merged_jobs:
                merged_jobs[oj_id] = oj_desc
                job_map[oj_id] = oj_id
            else:
                if merged_jobs[oj_id] == oj_desc:
                    job_map[oj_id] = oj_id
                else:
                    # deterministic suffixing
                    n = 1
                    new_id = f"{oj_id}__dup__{n}"
                    while new_id in merged_jobs:
                        n += 1
                        new_id = f"{oj_id}__dup__{n}"
                    merged_jobs[new_id] = oj_desc
                    job_map[oj_id] = new_id

        # Merge candidates from other
        for oc_id, oc_resume in other.candidates.items():
            if oc_id not in merged_cands:
                merged_cands[oc_id] = oc_resume
                cand_map[oc_id] = oc_id
            else:
                if merged_cands[oc_id] == oc_resume:
                    cand_map[oc_id] = oc_id
                else:
                    n = 1
                    new_id = f"{oc_id}__dup__{n}"
                    while new_id in merged_cands:
                        n += 1
                        new_id = f"{oc_id}__dup__{n}"
                    merged_cands[new_id] = oc_resume
                    cand_map[oc_id] = new_id

        # Build merged ratings (sparse)
        merged_ratings: Dict[Tuple[str, str], int] = {}
        # start with self ratings
        for (j_id, c_id), r in self.ratings.items():
            merged_ratings[(j_id, c_id)] = r

        # then incorporate other's ratings, remapped
        for (oj_id, oc_id), r in other.ratings.items():
            mj = job_map.get(oj_id)
            mc = cand_map.get(oc_id)
            if mj is None or mc is None:
                # skip ratings that reference unknown ids
                continue
            existing = merged_ratings.get((mj, mc), -1)
            # deterministic policy: keep existing if it is known (not -1), otherwise accept other's
            if existing == -1 and r != -1:
                merged_ratings[(mj, mc)] = r

        # return a new HRDataset using the sparse merged_ratings dict
        return HRDataset(jobs=merged_jobs, candidates=merged_cands, ratings=merged_ratings)
