# Config & Settings

Configuration is split into two files: `settings.py` for all tunable
parameters, and `prompts.py` for all LLM prompt strings.

This separation means you can tune retrieval behaviour (changing `RRF_K`,
`RETRIEVAL_N`) or swap LLM providers (changing `GCP_GEMINI_MODEL`) purely
through environment variables — no code changes needed.

---

## Settings

All parameters are read from environment variables at startup (via `python-dotenv`).
Defaults are shown below. Override any value in your `.env` file.

```bash title="Key environment variables"
GCP_PROJECT_ID=top-arc-65ca
GCP_LOCATION_ID=australia-southeast1
GCP_EMBED_MODEL=text-embedding-005
GCP_GEMINI_MODEL=gemini-2.5-flash
GCLOUD_PATH=/path/to/gcloud
HTTPS_PROXY=cloudproxy.auiag.corp:8080  # omit if not behind a proxy
DB_DSN=dbname=anzsic_db
RRF_K=60
RETRIEVAL_N=20
TOP_K=5
EMBED_DIM=768
```

::: prod.config.settings
    options:
      members:
        - Settings
        - get_settings

---

## Prompts

All LLM prompt strings live in one place. To tune how Gemini re-ranks
candidates, edit `RERANK_SYSTEM_BASE`. To change the output schema, edit
`RERANK_USER_TEMPLATE` and update the corresponding Pydantic models.

::: prod.config.prompts
    options:
      members:
        - build_system_prompt
        - build_candidate_block
        - build_user_message
