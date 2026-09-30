# RAG Projects

Retrieval-augmented generation projects, each built end to end and measured against a baseline.

| Project | What it does | Result |
|---|---|---|
| [rag-semantic-cache](rag-semantic-cache/) | RAG over the GitLab Handbook with an LLM-verified semantic cache, cost-aware model routing and a small-to-large model cascade | **-34% LLM cost** at equal answer quality (54-question eval) |
| rag-multi-tenant | Permission-aware multi-tenant RAG with role-based retrieval filtering and audit logging | *in progress* |
| rag-graphrag | GraphRAG: knowledge-graph traversal combined with vector search for multi-hop questions | *planned* |

Each project folder has its own README with architecture, results, what didn't work, and how to run it.
