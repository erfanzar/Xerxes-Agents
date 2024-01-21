import os

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
    CSVLoader,
    JSONLoader
)
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from ..interactors import BaseInteract

from ..inference import InferenceSession


class PyRAG:

    def __init__(
            self,
            inference_session: InferenceSession,
            embedding: HuggingFaceBgeEmbeddings,
            faiss_stand: FAISS,
            top_k: int,
            seperator: RecursiveCharacterTextSplitter
    ):
        self.inference_session = inference_session
        self.embedding = embedding
        self.faiss_stand = faiss_stand
        self.top_k = top_k
        self.seperator = seperator
        self.docs_searched = 0

    @classmethod
    def create(
            cls,
            inference_session: InferenceSession,
            embedding: HuggingFaceBgeEmbeddings,
            path: str | os.PathLike,
            glob: str = "./*.txt",
            loader_cls: [
                TextLoader,
                PyPDFLoader,
                CSVLoader,
                JSONLoader
            ] = TextLoader,
            chunk_size=500,
            chunk_overlap=200,
            top_k: int = 2
    ):
        loader = DirectoryLoader(
            path,
            glob=glob,
            loader_cls=loader_cls
        )

        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_documents(documents)
        vector_database = FAISS.from_documents(
            documents=texts,
            embedding=embedding,
        )
        return cls(
            inference_session=inference_session,
            embedding=embedding,
            top_k=top_k,
            faiss_stand=vector_database,
            seperator=text_splitter
        )

    def add_data_to_faiss(self, texts: list[str] | str, meta_data=list[dict] | dict):
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(meta_data, dict):
            meta_data = [meta_data]
        if meta_data is None:
            meta_data = [{"source": "UnKnown"} for _ in range(len(texts))]
        documents = self.seperator.create_documents(texts, meta_data)
        self.faiss_stand.add_documents(documents)
        return True

    def search(
            self,
            query: str,
            search_type="similarity",
            top_k: int | None = None
    ):
        self.docs_searched += 1
        return self.faiss_stand.search(
            query=query,
            search_type=search_type,
            k=top_k or self.top_k
        )

    @staticmethod
    def get_contexts_for_interactor(
            documents,
            *,
            query: str | None = None,
            interactor: BaseInteract | None = None
    ):
        contexts = [res.page_content for res in documents]
        if interactor is None:
            return contexts
        assert query is not None, "query can not be None in case you passing Interactor"
        return interactor.retrival_qa_template(
            contexts=contexts,
            question=query,
        )

    def __repr__(self):

        """
        The __repr__ function is used to generate a string representation of an object.
        This function should return a string that can be parsed by the Python interpreter
        to recreate the object. The __repr__ function is called when you use print() on an
        object, or when you type its name in the REPL.

        :param self: Refer to the instance of the class
        :return: A string representation of the object
        """
        string = f"{self.__class__.__name__}(\n"
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                repr_src = f"\t{k} : " + v.__str__().replace("\n", "\n\t") + "\n"
                string += repr_src if len(repr_src) < 500 else f"\t{k} : " + f"{v.__class__.__name__}(...)" + "\n"
        return string + ")"

    def __str__(self):

        """
        The __str__ function is called when you use the print function or when str() is used.
        It should return a string representation of the object.

        :param self: Refer to the instance of the class
        :return: The object's string representation
        """
        return self.__repr__()
