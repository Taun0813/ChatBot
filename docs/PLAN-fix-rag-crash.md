# PLAN-fix-rag-crash

## Goal
Fix `AttributeError: 'NoneType' object has no attribute 'pc'` in `core/rag_model.py` and enable proper RAG functionality handling.

## Analysis
- **Crash Cause**: `app.py` sets `rag_config["enabled"] = False`. This causes `AgnoRouter` to skip initializing `pinecone_client`.
- `RAGModel` receives `pinecone_client=None`.
- `_generate_embedding` tries to access `self.pinecone_client.pc`, resulting in the error.
- **User Confusion**: User added Gemini API Key, but the system is failing on Pinecone Client. The system depends on Pinecone for embeddings (`pc.inference.embed`).

## Proposed Changes

### [MODIFY] [core/rag_model.py](file:///d:/Code/NCKH/ChatBot/core/rag_model.py)
- Update `_generate_embedding` to check if `self.pinecone_client` is initialized.
- If not initialized, raise a clear `ValueError("Pinecone client not initialized")`.
- Update `search_products` to handle this case gracefully (e.g., return empty list with specific log warning).

## Verification Plan

### Automated Test
- Create `tests/test_rag_crash.py` that initializes `RAGModel` with `None` client and calls `search_products`.
- Verify it returns empty list or raises handled exception instead of crashing with AttributeError.

### Manual Verification
- Run `python app.py` (assuming user wants to verify runtime) - but we can't change `app.py` config without user permission. We will focus on preventing the *unexpected* crash.

## Note to User
- We will notify the user that to use Search, they must:
    1. Enable RAG in `app.py`.
    2. Provide Pinecone API Key in `.env` (not just Gemini).
