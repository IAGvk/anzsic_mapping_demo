# Ports (Interfaces)

Ports are the **abstract contracts** that define what the system needs from the
outside world — without specifying how those needs are met.

They use Python's `typing.Protocol` with `@runtime_checkable`, which means:

- Any class that implements the required methods is a valid adapter
- **No inheritance required** — duck typing at its best
- Mock adapters in tests satisfy the protocol without importing any real adapter

---

## Swapping a provider

To replace any provider (e.g. swap Gemini for GPT-4o):

1. Write a new adapter class with the methods defined by the Port
2. Change **one import line** in `services/container.py`
3. Done — no other file changes

---

## EmbeddingPort

Defines the contract for turning text into dense float vectors.

::: prod.ports.embedding_port
    options:
      members:
        - EmbeddingPort

---

## LLMPort

Defines the contract for generating a JSON string from a prompt pair.

::: prod.ports.llm_port
    options:
      members:
        - LLMPort

---

## DatabasePort

Defines the contract for the hybrid search database backend.

Note that RRF fusion is deliberately **not** part of this port — it lives in
`services/retriever.py` as pure Python so it can be unit-tested without a
database connection.

::: prod.ports.database_port
    options:
      members:
        - DatabasePort
