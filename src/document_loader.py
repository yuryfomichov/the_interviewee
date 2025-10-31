"""Document loading and vector database management for AI Interviewee."""

import logging

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_config

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Manages loading career documents and creating vector database."""

    def __init__(self, config=None):
        """Initialize document loader.

        Args:
            config: Configuration instance (creates new if None)
        """
        self.config = config or get_config()
        self.embeddings = None
        self.vector_store = None
        self._initialize_embeddings()

    def _initialize_embeddings(self) -> None:
        """Initialize embedding model."""
        logger.info(f"Loading embedding model: {self.config.embedding_model}")

        # Set up HuggingFace embeddings
        model_kwargs = {"device": self.config.local_model_device}
        encode_kwargs = {"normalize_embeddings": True}

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

        logger.info("Embedding model loaded successfully")

    def load_documents(self) -> list:
        """Load all markdown documents from career data directory.

        Returns:
            List of loaded documents
        """
        career_path = self.config.career_data_path

        if not career_path.exists():
            raise FileNotFoundError(
                f"Career data directory not found: {career_path}\n"
                f"Please create the directory and add your career documents."
            )

        # Check if directory has any markdown files
        md_files = list(career_path.glob("*.md"))
        if not md_files:
            raise ValueError(
                f"No markdown files found in {career_path}\n"
                f"Please add your career documents (e.g., cv.md, interview_answers.md)"
            )

        logger.info(f"Loading documents from {career_path}")
        logger.info(f"Found {len(md_files)} markdown files")

        # Load all markdown files
        loader = DirectoryLoader(
            str(career_path),
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
        )

        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents")

        return documents

    def split_documents(self, documents: list) -> list:
        """Split documents into chunks.

        Args:
            documents: List of documents to split

        Returns:
            List of document chunks
        """
        logger.info(
            f"Splitting documents into chunks "
            f"(size={self.config.chunk_size}, overlap={self.config.chunk_overlap})"
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} document chunks")

        return chunks

    def create_vector_store(self, documents: list, force_rebuild: bool = False) -> Chroma:
        """Create or load vector store from documents.

        Args:
            documents: List of document chunks
            force_rebuild: Force rebuild even if vector store exists

        Returns:
            Chroma vector store instance
        """
        vector_db_path = self.config.vector_db_path

        # Check if vector store already exists
        if vector_db_path.exists() and not force_rebuild and not self.config.rebuild_index:
            logger.info(f"Loading existing vector store from {vector_db_path}")
            self.vector_store = Chroma(
                persist_directory=str(vector_db_path), embedding_function=self.embeddings
            )
            logger.info(f"Loaded vector store with {self.vector_store._collection.count()} chunks")
            return self.vector_store

        # Create new vector store
        logger.info(f"Creating new vector store at {vector_db_path}")
        vector_db_path.mkdir(parents=True, exist_ok=True)

        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=str(vector_db_path),
        )

        logger.info(f"Vector store created with {len(documents)} chunks")
        return self.vector_store

    def initialize(self, force_rebuild: bool = False) -> Chroma:
        """Initialize the complete document loading pipeline.

        Args:
            force_rebuild: Force rebuild of vector database

        Returns:
            Chroma vector store instance
        """
        logger.info("Initializing document loader pipeline")

        # If vector store exists and we're not rebuilding, just load it
        if (
            self.config.vector_db_path.exists()
            and not force_rebuild
            and not self.config.rebuild_index
        ):
            return self.create_vector_store([], force_rebuild=False)

        # Otherwise, load and process documents
        documents = self.load_documents()
        chunks = self.split_documents(documents)
        vector_store = self.create_vector_store(chunks, force_rebuild=force_rebuild)

        logger.info("Document loader initialization complete")
        return vector_store

    def search(self, query: str, k: int | None = None) -> list[tuple]:
        """Search for relevant documents.

        Args:
            query: Search query
            k: Number of results to return (uses config default if None)

        Returns:
            List of (document, score) tuples
        """
        if self.vector_store is None:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")

        k = k if k is not None else self.config.top_k

        results = self.vector_store.similarity_search_with_score(query, k=k)

        # Filter by relevance threshold
        filtered_results = [
            (doc, score) for doc, score in results if score >= self.config.relevance_threshold
        ]

        logger.debug(
            f"Search returned {len(filtered_results)}/{len(results)} results "
            f"above threshold {self.config.relevance_threshold}"
        )

        return filtered_results
