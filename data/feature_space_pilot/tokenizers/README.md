# Offline tokenizer data

The o200k_base rank table is the exact public OpenAI tiktoken resource:
https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken

SHA-256: `446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d`. The regex, special tokens and model mapping come from OpenAI tiktoken 0.12.0 (MIT license included). No network fetch or credentials are needed after checkout/dependency installation. Unknown model mappings fail instead of silently falling back.

Counts tokenize the complete canonical transport JSON and add an explicit 512-token framing reserve. They are conservative local input budgets, not exact provider counts. The controller and client validate before reservations/transport; actual provider usage settles the shared ledger and any envelope breach halts execution.
