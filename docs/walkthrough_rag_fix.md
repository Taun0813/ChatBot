# Walkthrough - RAG Crash Fix

## The Issue
Users encountered an error when trying to search for products (e.g., "suggest phones...").
**Error Log**: `Failed to generate embedding: 'NoneType' object has no attribute 'pc'`
**Root Cause**:
1. `app.py` has `rag_config["enabled"] = False`.
2. This prevents `AgnoRouter` from initializing `pinecone_client`.
3. `RAGModel` is initialized with `pinecone_client=None`.
4. When `search_products` is called, it tries to access `self.pinecone_client.pc`, leading to the crash.

## The Fix
Modified `core/rag_model.py` to:
1. Add checks in `_generate_embedding` to ensure `pinecone_client` and `pinecone_client.pc` are initialized.
2. Throw a clear `ValueError("Pinecone client not initialized")` if check fails.
3. Update `search_products` and `upsert_product` to handle this error gracefully (return empty list/False instead of crashing).

## Verification
Created `tests/test_rag_crash.py` to simulate the environment with disabled RAG.
**Output**:
```
--- Testing RAG Model Crash Fix ---
Attempting to search products...
ERROR:core.rag_model:Failed to generate embedding: Pinecone client not initialized
ERROR:core.rag_model:Failed to search products: Pinecone client not initialized
SUCCESS: Returned empty list gracefully
```

## Required User Action
To fully fix the *functionality* (not just the crash), the user must:
1. **Enable RAG** in `app.py` or `.env`.
2. **Provide Pinecone API Key** (in addition to Gemini Key), as the system uses Pinecone Inference for embeddings.
