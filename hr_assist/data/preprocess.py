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
from typing import Union
from pathlib import Path

from .dataset import HRDataset
from .raw_data import RawDataHandler

def preprocess_dataset(handler: RawDataHandler, output_path: Union[str, Path]) -> HRDataset:
    """Preprocess the dataset using the provided RawDataHandler and save to output_path.

    Args:
        handler (RawDataHandler): An instance of a RawDataHandler for the specific dataset.
        output_path (str): Path to save the processed dataset.
    """
    dataset = HRDataset(
        jobs={},
        candidates={},
        ratings=[],
    )

    # Process job descriptions
    for job in handler.get_job_descriptions():
        dataset.add_job(job_id=job['id'], job_description=job['description'])

    # Process candidate resumes
    for candidate in handler.get_candidate_resumes():
        dataset.add_candidate(candidate_id=candidate['id'], resume=candidate['resume'])

    # Process ratings
    for job_id, candidate_id, rating in handler.get_ratings():
        dataset.add_rating(job_id=job_id, candidate_id=candidate_id, rating=rating)

    # Save the processed dataset
    dataset.save_to_jsonl(output_path)

    return dataset

def preprocess_ai_data_entry_mgr(root_dir: Union[str, Path], output_path: Union[str, Path]) -> None:
    """Preprocess the AI Data Entry Manager dataset and save to output_path.

    Args:
        root_dir (Union[str, Path]): Root directory of the AI Data Entry Manager dataset.
        output_path (Union[str, Path]): Path to save the processed dataset.
    """
    from .ai_data_entry_mgr import AIDataEntryMgrHandler

    handler = AIDataEntryMgrHandler(data_dir=root_dir)
    dataset = preprocess_dataset(handler, output_path)
    return dataset

def preprocess_alkalmazas_tesztelo(root_dir: Union[str, Path], output_path: Union[str, Path]) -> None:
    """Preprocess the Alkalmazas Tesztelo dataset and save to output_path.

    Args:
        root_dir (Union[str, Path]): Root directory of the Alkalmazas Tesztelo dataset.
        output_path (Union[str, Path]): Path to save the processed dataset.
    """
    from .alkalmazas_tesztelo import AlkalmazasTeszteloHandler

    handler = AlkalmazasTeszteloHandler(data_dir=root_dir)
    dataset = preprocess_dataset(handler, output_path)
    return dataset

def preprocess_penzugyi_auditor(root_dir: Union[str, Path], output_path: Union[str, Path]) -> None:
    """Preprocess the Penzugyi Auditor dataset and save to output_path.

    Args:
        root_dir (Union[str, Path]): Root directory of the Penzugyi Auditor dataset.
        output_path (Union[str, Path]): Path to save the processed dataset.
    """
    from .penzugyi_auditor import PenzugyiAuditorHandler

    handler = PenzugyiAuditorHandler(data_dir=root_dir)
    dataset = preprocess_dataset(handler, output_path)
    return dataset

