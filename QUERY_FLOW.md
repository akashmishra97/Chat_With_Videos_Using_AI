# Video Analysis App: User Query Processing & Retrieval Flow

This document describes the flow for handling user queries and retrieving relevant video segments using the enhanced vector database in the Video Analysis App.

---

## 1. Check for Uploaded Video
- The system verifies if a video has been uploaded (`current_video`).
- If not, it returns an error: `{'error': 'No video has been uploaded yet'}`.

## 2. Extract User Message
- Expects a JSON payload containing a `message` field (the user’s query).
- If missing, returns: `{'error': 'No message provided'}`.

## 3. Query the Vector Database
- Logs the user’s message for debugging.
- Sends the message to ChromaDB vector database using `video_collection.query()` (top 5 results).
- Leverages advanced semantic search with sentence-transformers embeddings and hierarchical indexing (video, scene, timestamp).

## 4. Process Query Results
- Checks if any documents are returned.
- For each result, extracts:
    - The relevant context (summary/segment from the video)
    - The associated timestamp(s)
- Collects these into `contexts` and `timestamps` lists.

## 5. Response Construction
- Uses the extracted contexts and timestamps to generate a comprehensive response, summarizing relevant video segments and their timing.

---

### Key Features
- **Hierarchical Indexing:** Efficient search at video, scene, and timestamp levels.
- **Semantic Chunking:** Intelligent segmentation and boundary detection for precise context retrieval.
- **Metadata Enrichment:** Each chunk includes semantic density and temporal metadata.
- **Multi-Stage Retrieval:** Initial semantic search, then reranking for relevance.
- **Context Assembly:** Gathers and organizes relevant snippets and their time positions for user-friendly output.

---

For further details, refer to the implementation in `app.py` and the vector database setup.
