import sys
import os
import asyncio
import logging

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.rag_model import RAGModel

# Configure logging
logging.basicConfig(level=logging.ERROR)

async def main():
    print("--- Testing RAG Model Crash Fix ---")
    
    # Initialize with None client (simulating disabled RAG)
    rag_model = RAGModel(pinecone_client=None, model_loader=None)
    
    print("Attempting to search products...")
    try:
        results = await rag_model.search_products("test query")
        print(f"Search Results: {results}")
        if results == []:
            print("SUCCESS: Returned empty list gracefully")
        else:
            print("FAILURE: Returned unexpected results")
            
    except Exception as e:
        print(f"FAILURE: Crashed with error: {e}")
        # Print full traceback if needed
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
