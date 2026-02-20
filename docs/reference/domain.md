# Domain Models

The domain layer contains all the pure Python types used throughout the system.
Nothing here imports from infrastructure — these models are the shared language
between every layer.

---

## Models

The key object flow is:

```
SearchRequest → [pipeline] → ClassifyResponse
                                  └── results: list[ClassifyResult]
                                        (internally via Candidate objects)
```

::: prod.domain.models
    options:
      members:
        - SearchMode
        - SearchRequest
        - Candidate
        - ClassifyResult
        - ClassifyResponse

---

## Exceptions

All errors raised by the system are subclasses of `ANZSICError`.
This means callers can catch the base class for broad handling, or individual
subclasses for fine-grained recovery.

```python
from prod.domain.exceptions import ANZSICError, AuthenticationError

try:
    response = pipeline.classify(request)
except AuthenticationError:
    # Token expired — re-authenticate and retry
    ...
except ANZSICError as e:
    # Any other classifier error
    logger.error("Classification failed: %s", e)
```

When a FastAPI layer is added, the exception hierarchy maps directly to HTTP
status codes (see docstring in `exceptions.py`).

::: prod.domain.exceptions
    options:
      members:
        - ANZSICError
        - ConfigurationError
        - AuthenticationError
        - EmbeddingError
        - LLMError
        - DatabaseError
        - RetrievalError
        - RerankError
