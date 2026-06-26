# Pallas Kernel Explanation Workflow

Use this workflow when the user asks a conceptual question about JAX, Pallas, TPU kernels, memory hierarchy, profiling, or a kernel file.

## 1. Ground the answer

If the user references a file, read that file first with chunked reads. Then consult:

- `references/knowledge-base.md`;
- local project docs or examples;
- current JAX/Pallas docs when the API detail may have drifted.

Do not answer Pallas API questions from memory alone when the answer affects code.

## 2. Retrieval strategy

Use parallel retrieval for obvious facets, then sequential retrieval for gaps:

1. Query the core API or concept.
2. Query examples.
3. Query known pitfalls or TPU-specific notes.
4. Read the results and only then answer.

For example, a BlockSpec question should check:

- `pl.BlockSpec`;
- `pl.pallas_call`;
- `pl.program_id`;
- rank/index-map rules;
- TPU lowering constraints.

## 3. Response shape

Return:

- a direct answer first;
- code example when helpful;
- caveats and version/API uncertainty;
- file and line references when explaining local code;
- next action if the user wants implementation.
