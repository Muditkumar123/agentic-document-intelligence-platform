# Academic RAG Evaluation Note

The document intelligence platform evaluates retrieval before answer generation because a grounded answer depends on retrieving the right evidence first.

The evaluation dataset contains golden questions with expected substrings and expected chunk identifiers. Retrieval quality is measured with hit rate at k and mean reciprocal rank.

Mean reciprocal rank rewards systems that place the first correct evidence chunk near the top of the result list.

The academic preset focuses on problem statements, methods, datasets, results, and limitations.
