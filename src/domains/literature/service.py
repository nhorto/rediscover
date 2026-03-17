"""Literature service: paper ingestion, embedding, and retrieval."""

import chromadb
from sentence_transformers import SentenceTransformer

from src.domains.literature.config import (
    CATEGORIES,
    CHROMA_PATH,
    COLLECTION_NAME,
    CUTOFF_DATE,
    DEFAULT_QUERY_RESULTS,
    EMBEDDING_MODEL,
)
from src.domains.literature.types import Paper, SearchResult
from src.providers.arxiv import search_papers


class LiteratureService:
    """Manages paper knowledge base: ingest from arXiv, embed, and retrieve."""

    def __init__(
        self,
        chroma_path: str = CHROMA_PATH,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self._chroma_client = chromadb.PersistentClient(path=chroma_path)
        self._collection = self._chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder: SentenceTransformer | None = None
        self._embedding_model_name = embedding_model

    @property
    def embedder(self) -> SentenceTransformer:
        """Lazy-load the embedding model (heavy, ~400MB)."""
        if self._embedder is None:
            self._embedder = SentenceTransformer(self._embedding_model_name)
        return self._embedder

    @property
    def paper_count(self) -> int:
        """Number of papers in the knowledge base."""
        return self._collection.count()

    def ingest_papers(
        self,
        query: str,
        max_results: int = 1000,
        categories: list[str] | None = None,
        cutoff_date: str = CUTOFF_DATE,
    ) -> int:
        """Fetch papers from arXiv, embed abstracts, store in Chroma.

        Returns the number of NEW papers added (skips duplicates).
        """
        cats = categories or CATEGORIES
        papers = search_papers(query, categories=cats, before_date=cutoff_date, max_results=max_results)

        if not papers:
            return 0

        # Filter out papers already in the collection
        existing_ids = set()
        if self._collection.count() > 0:
            all_ids = self._collection.get()["ids"]
            existing_ids = set(all_ids)

        new_papers = [p for p in papers if p.arxiv_id not in existing_ids]
        if not new_papers:
            return 0

        # Embed abstracts
        texts = [f"{p.title}. {p.abstract}" for p in new_papers]
        embeddings = self.embedder.encode(texts, show_progress_bar=True, batch_size=64)

        # Extract year from published date for metadata filtering
        def get_year(published: str) -> int:
            try:
                return int(published[:4])
            except (ValueError, IndexError):
                return 0

        # Store in Chroma (abstract stored separately in metadata so it's not lost)
        self._collection.add(
            ids=[p.arxiv_id for p in new_papers],
            embeddings=[emb.tolist() for emb in embeddings],
            documents=texts,
            metadatas=[
                {
                    "title": p.title,
                    "abstract": p.abstract[:4000],  # Chroma metadata has size limits
                    "year": get_year(p.published),
                    "primary_category": p.primary_category,
                    "published": p.published,
                    "authors": ", ".join(p.authors[:5]),
                }
                for p in new_papers
            ],
        )

        return len(new_papers)

    def search(
        self,
        topic: str,
        n_results: int = DEFAULT_QUERY_RESULTS,
        max_year: int | None = 2023,
    ) -> list[SearchResult]:
        """Search the knowledge base for papers relevant to a topic.

        Args:
            topic: Natural language description of what you're looking for
            n_results: Number of results to return
            max_year: Only return papers published in or before this year
        """
        if self._collection.count() == 0:
            return []

        query_embedding = self.embedder.encode([topic]).tolist()

        where_filter = None
        if max_year is not None:
            where_filter = {"year": {"$lte": max_year}}

        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=min(n_results, self._collection.count()),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, arxiv_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0.0
                # Chroma cosine distance: 0 = identical, 2 = opposite. Convert to similarity.
                similarity = 1.0 - (distance / 2.0)

                paper = Paper(
                    arxiv_id=arxiv_id,
                    title=metadata.get("title", ""),
                    abstract=metadata.get("abstract", ""),
                    authors=metadata.get("authors", "").split(", "),
                    published=metadata.get("published", ""),
                    categories=[metadata.get("primary_category", "")],
                    primary_category=metadata.get("primary_category", ""),
                    pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
                )
                search_results.append(SearchResult(paper=paper, score=similarity))

        return search_results
