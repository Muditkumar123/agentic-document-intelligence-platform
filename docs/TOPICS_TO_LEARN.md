# Topics To Learn For This Project

Use this as a checklist while studying the Agentic Document Intelligence Platform. The topics are grouped in the same order as the project pipeline.

## 1. Python And Project Engineering

- Python project structure
- `pyproject.toml`
- Virtual environments and Conda
- CLI development with `argparse`
- JSON and JSONL files
- Dataclasses
- Type hints
- Unit testing with `pytest`
- Logging and error handling
- Git basics

## 2. Document Ingestion

- PDF parsing
- Text and Markdown parsing
- Document metadata extraction
- Checksums
- Page numbers
- Chunk IDs
- Text cleaning
- Chunk size
- Chunk overlap
- Why chunking matters for RAG

## 3. NLP Basics

- Tokenization
- Stop words
- N-grams
- Text normalization
- Similarity search
- Entity extraction
- Topic extraction
- Claim, evidence, risk, method, result, and limitation extraction

## 4. Retrieval And RAG

- What RAG is
- Sparse retrieval
- TF-IDF
- Dense retrieval
- Embeddings
- Sentence Transformers
- Cosine similarity
- Inner product similarity
- Top-k retrieval
- FAISS
- Vector indexes
- Candidate generation
- Reranking
- Cross-encoder rerankers
- Bi-encoder vs cross-encoder retrieval
- Query-document pair scoring
- Retrieval latency
- Index size
- Hybrid retrieval
- Reranking

## 5. Retrieval Evaluation

- Golden QA datasets
- Expected chunk IDs
- Expected answer substrings
- Hit rate@k
- Recall@k
- Mean reciprocal rank
- Query latency
- Retrieval benchmark reports
- TF-IDF vs dense retrieval comparison
- Plain retrieval vs reranked retrieval comparison

## 6. LLM Fundamentals

- Large language models
- Local open-source LLMs
- Hugging Face Transformers
- Chat templates
- Context window
- Input tokens
- Output tokens
- Max new tokens
- Temperature
- Greedy decoding
- Hallucination
- Grounded generation
- Cited answer generation

## 7. Model Choices In This Project

- Extractive baseline
- Qwen3-8B
- DeepSeek-R1-Distill-Qwen-14B
- When to use a general instruct model
- When to use a reasoning model
- GPU memory requirements
- bf16 inference
- Quantization basics

## 8. LLMOps

- Prompt templates
- Prompt versioning
- Prompt hashes
- Model profiles
- Provider adapters
- Token tracking
- Latency tracking
- GPU memory tracking
- Citation coverage
- Unsupported sentence detection
- Structured verifier output
- Raw output vs normalized output

## 9. Agentic AI

- Agent workflows
- State machines
- Intent routing
- Planning nodes
- Retrieval nodes
- Evidence verifier nodes
- Writer nodes
- Citation checker nodes
- Tool use
- Multi-step reasoning
- Failure handling

## 10. AgentOps

- Agent run IDs
- Trace events
- Node-level observability
- Input summaries
- Output summaries
- Workflow duration
- Verifier metrics
- Citation checker metrics
- Debugging agent failures
- Failure replay

## 11. MLOps

- Experiment tracking
- Parameters
- Metrics
- Artifacts
- Reproducibility
- MLflow basics
- DVC basics
- Pipeline stages
- Dataset versioning
- Model/version tracking
- Docker basics
- CI smoke tests

## 12. Serving And Deployment

- Local model serving
- OpenAI-compatible APIs
- FastAPI basics
- vLLM
- SGLang
- CUDA devices
- `CUDA_VISIBLE_DEVICES`
- GPU memory allocation
- GPU memory reservation
- Batch size and throughput
- Latency monitoring

## 13. Monitoring And Evaluation

- Retrieval quality monitoring
- Answer quality monitoring
- Citation accuracy
- Hallucination risk
- Text drift
- Query drift
- Latency drift
- Failure rate
- Production feedback loops

## 14. Future Fine-Tuning Topics

- Supervised fine-tuning
- LoRA
- QLoRA
- PEFT
- Training datasets
- Labeling document chunks
- Base model vs fine-tuned model comparison
- Evaluation before and after fine-tuning

## 15. Interview Topics To Practice

- Why RAG is needed
- Why chunking is needed
- Why metadata is important
- Why TF-IDF is a good baseline
- Why dense retrieval is useful
- Why FAISS is used
- How retrieval is evaluated
- How hallucination is reduced
- Difference between MLOps, LLMOps, and AgentOps
- Why Qwen and DeepSeek are used differently
- How a 40 GB GPU affects model choice
- How to debug a bad answer
- What you would improve next

## Suggested Learning Order

1. Python project structure and testing
2. Document ingestion and chunking
3. TF-IDF retrieval
4. Dense embeddings and FAISS
5. Retrieval evaluation metrics
6. RAG and cited answer generation
7. LLMOps and prompt tracking
8. Agent workflow design
9. AgentOps tracing
10. MLOps tracking and DVC
11. Local LLM serving
12. Monitoring and fine-tuning
