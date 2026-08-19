"""Shared pytest fixtures: an isolated in-memory SQLite DB per test, and a
FastAPI TestClient wired to it instead of the real database.

Using :memory: SQLite (rather than a shared test.db file) means tests never
leak state into each other or into local dev's real database.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.database import Base, get_db
from src.main import app


@pytest.fixture()
def db_session():
    """Fresh in-memory SQLite session, tables created, per test.

    StaticPool forces every checkout to reuse the same single connection.
    Without it, a plain :memory: engine hands each thread its own separate
    (and separately empty) database — and TestClient runs endpoint code in a
    worker thread, so the tables created here would be invisible to it.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(autocommit=False, autoflush=False, bind=engine)()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def client(db_session):
    """TestClient with get_db overridden to use db_session instead of the app's real DB."""

    def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
