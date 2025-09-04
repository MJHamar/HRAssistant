"""
Interface module for database operations in the HR Assist application.
Supports 
- SQL databases
- Vector databases

Current implementation only supports PostgreSQL and pg-vector databases.
"""

from .base import BaseDb
from .pg import PostgresDB

__all__ = ["BaseDb", "PostgresDB"]