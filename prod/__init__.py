"""
ANZSIC Occupation Code Classifier — Production Package
=======================================================
Hexagonal (Ports & Adapters) architecture.

Layer map
─────────────────────────────────────────────────────
  config/       All tuneable settings & prompt strings
  domain/       Pure business objects (models, exceptions) — no I/O
  ports/        Abstract interfaces (Python Protocols)
  adapters/     Concrete implementations of each Port (GCP, Postgres…)
  services/     Orchestration logic; depends only on Ports, never Adapters
  interfaces/   Delivery layer: CLI, Streamlit UI, (future) FastAPI
  tests/        Full test suite: unit / integration / e2e

Swapping any external dependency (LLM, embedding model, database):
  1. Write a new adapter in adapters/ implementing the relevant Port
  2. Change the single wiring line in services/container.py
  3. Done — zero other files touched
"""
__version__ = "1.0.0"
