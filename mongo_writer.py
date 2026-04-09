"""
mongo_writer.py — Write arrhythmia analysis results to MongoDB.
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

import config

log = logging.getLogger(__name__)

_client: MongoClient | None = None


def get_collection():
    global _client
    if _client is None:
        _client = MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=5000)
    return _client[config.MONGO_DB][config.MONGO_COLLECTION]


def write_result(result: dict, retries: int = 3) -> str:
    """
    Write arrhythmia analysis result to MongoDB.
    Returns the document UUID on success.
    Raises RuntimeError after retries are exhausted.
    """
    doc_uuid = str(uuid.uuid4())
    doc = {**result, "uuid": doc_uuid}

    for attempt in range(1, retries + 1):
        try:
            col = get_collection()
            col.insert_one(doc)
            log.info(f"{result.get('admissionId')} | Written to MongoDB (uuid={doc_uuid})")
            return doc_uuid
        except (ConnectionFailure, ServerSelectionTimeoutError) as exc:
            log.warning(f"MongoDB write attempt {attempt}/{retries} failed: {exc}")
            if attempt < retries:
                time.sleep(2 ** attempt)   # 2s, 4s backoff
            else:
                raise RuntimeError(f"MongoDB unavailable after {retries} attempts") from exc
        except Exception as exc:
            log.error(f"MongoDB unexpected error: {exc}")
            raise
