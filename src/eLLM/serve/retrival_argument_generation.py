import warnings
from abc import abstractmethod, ABC
from typing import Literal, List, Optional
from warnings import warn

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:
    warn("`sentence_transformers` is not installed")
    SentenceTransformer = object

try:
    import faiss
except ModuleNotFoundError:
    warn("`faiss-cpu` or `faiss-gpu` is not installed")

    faiss = object()
    faiss.IndexFlatL2 = None

from .llm_serve import LLMServe


def search(
        query: str,
        index: faiss.IndexFlatL2,
        embedding: SentenceTransformer,
        text_snippets: list[str],
        k: int,
):
    score, index = index.search(
        embedding.encode([query]),
        k=k
    )
    index = index[0].tolist()
    score = score[0].tolist()
    return [(text_snippets[idx], score[i]) for i, idx in enumerate(index)]


class RAGLLMServe(LLMServe, ABC):
    index: Optional[faiss.IndexFlatL2] = None
    embedding: Optional[SentenceTransformer] = None
    text_snippets: Optional[list[str]] = None
    rag_top_k: Optional[int] = 3
    rag_qa_base_question: Optional[str] = None
    threshold: Optional[float] = None

    def add_rag(
            self,
            index: faiss.IndexFlatL2,
            embedding: SentenceTransformer,
            text_snippets: list[str],
            rag_top_k: int,
            rag_qa_base_question: Optional[str] = None,
            threshold: Optional[float] = None
    ):
        self.index = index
        self.embedding = embedding
        self.text_snippets = text_snippets
        self.rag_top_k = rag_top_k
        self.rag_qa_base_question = rag_qa_base_question
        self.threshold = threshold

    def rag_search(
            self,
            query: str,
            k: Optional[int] = None,
            base_question: Optional[str] = None,
            threshold: Optional[float] = None,
            verbose: bool = False
    ):
        index: faiss.IndexFlatL2 | None = self.index
        embedding: SentenceTransformer | None = self.embedding
        text_snippets: list[str] | None = self.text_snippets

        if index is not None and embedding is not None and embedding is not None:
            contexts_and_scores = search(
                query=query,
                embedding=embedding,
                text_snippets=text_snippets,
                k=k or self.rag_top_k,
                index=index
            )
            threshold = threshold or self.threshold

            contexts = [
                context for context, score in contexts_and_scores
            ] if threshold is None else [
                context for context, score in contexts_and_scores if score > threshold
            ]
            if len(contexts) > 0:
                query = self.interactor.retrival_qa_template(
                    question=query,
                    contexts=contexts,
                    base_question=base_question or self.rag_qa_base_question
                )
                if verbose:
                    print(
                        query
                    )

        else:
            warnings.warn("You are not Using data_reader correctly you have to add data_reader via `add_rag` function")
        return query
