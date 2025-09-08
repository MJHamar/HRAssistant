-- PostgreSQL schema for HR Assistant
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    content TEXT
);

CREATE TABLE IF NOT EXISTS document_chunks (
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    idx INT NOT NULL,
    metadata JSONB,
    toc_items JSONB,
    tables JSONB,
    images JSONB,
    graphics JSONB,
    text TEXT,
    PRIMARY KEY (document_id, idx)
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    company_name TEXT,
    job_title TEXT NOT NULL,
    job_description TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS candidates (
    id TEXT PRIMARY KEY,
    candidate_name TEXT,
    candidate_cv_id TEXT
);

CREATE TABLE IF NOT EXISTS questionnaires (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    questionnaire JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS candidate_fitness (
    candidate_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    questionnaire_id TEXT NOT NULL,
    scores REAL[] NOT NULL,
    PRIMARY KEY (candidate_id, job_id, questionnaire_id)
);

CREATE TABLE IF NOT EXISTS embeddings (
    table_name TEXT NOT NULL,
    record_id TEXT NOT NULL,
    embedding vector(1536),
    metadata JSONB,
    PRIMARY KEY (table_name, record_id)
);
