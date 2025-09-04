"""
Module representing the service layer for the API.
"""
from ..func.lm import score_candidate, make_questionnaire
from ..utils import doc_to_md
from tempfile import TemporaryDirectory, TemporaryFile


class HRService:
    def __init__(self):
        self._db = None  # Placeholder for database connection
        
    @property
    def db(self):
        if self._db is None:
            pass
        return self._db
    
    def convert_document(document_name: str, document_content: bytes):
        with TemporaryFile('wb') as temp_file:
            temp_file.write(document_content)
            temp_file.seek(0)
            content, chunked_content, metadata = doc_to_md(temp_file)

            return content, chunked_content, metadata

    def upload_document(self, document_name: str, document_content: bytes):
        content, chunked_content, metadata = self.convert_document(document_name, document_content)
        return content, chunked_content, metadata

    def generate_questionnaire(self, job_description: str):
        return make_questionnaire(job_description=job_description)
    

__all__ = ["HRService"]