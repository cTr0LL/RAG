
# Embedding generation and management module.
# Handles sentence-transformers model loading and embedding creation with batching.


import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import asdict

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

from document_processor import DocumentChunk
from utils import PerformanceTracker, validate_input
from exceptions import EmbeddingError, ModelLoadError


#   Handles embedding generation using sentence-transformers.
class EmbeddingGenerator:


    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get('models', {}).get('embedding_model', 'all-MiniLM-L6-v2')
        self.batch_size = config.get('processing', {}).get('batch_size', 32)
        self.use_gpu = config.get('processing', {}).get('use_gpu', False)

        self.model = None
        self.model_loaded = False
        self.performance_tracker = PerformanceTracker()
        self.logger = logging.getLogger(__name__)

        # Initialize model
        self._load_model()

    # Load the sentence-transformers model
    def _load_model(self):
        try:
            self.logger.info(f"Loading embedding model: {self.model_name}")

            # Set device
            if self.use_gpu and torch.cuda.is_available():
                device = 'cuda'
                self.logger.info("Using GPU for embeddings")
            else:
                device = 'cpu'
                self.logger.info("Using CPU for embeddings")

            # Load model with error handling
            self.model = SentenceTransformer(self.model_name, device=device)

            # Log model information
            max_seq_length = getattr(self.model, 'max_seq_length', 'Unknown')
            embedding_dim = self.model.get_sentence_embedding_dimension()

            self.logger.info(
                f"Model loaded successfully:\n"
                f"  - Model: {self.model_name}\n"
                f"  - Device: {device}\n"
                f"  - Max sequence length: {max_seq_length}\n"
                f"  - Embedding dimension: {embedding_dim}"
            )

            self.model_loaded = True

        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            raise ModelLoadError(f"Could not load model {self.model_name}: {e}") from e

    # Returns numpy array of embeddings with shape (n_texts, embedding_dim)
    def generate_embeddings(
            self,
            texts: List[str], #List of text strings to embed
            show_progress_bar: bool = True
    ) -> np.ndarray:

        try:
            validate_input(texts, list, "texts")

            if not self.model_loaded:
                raise EmbeddingError("Model not loaded")

            if not texts:
                self.logger.warning("No texts provided for embedding")
                return np.array([])

            self.logger.info(f"Generating embeddings for {len(texts)} texts")
            self.performance_tracker.start_timer('embedding_generation')

            # Filter out empty texts and track indices
            valid_texts = []
            valid_indices = []

            for i, text in enumerate(texts):
                if text and text.strip():
                    valid_texts.append(text.strip())
                    valid_indices.append(i)
                else:
                    self.logger.warning(f"Empty text at index {i}, skipping")

            if not valid_texts:
                raise EmbeddingError("No valid texts to embed")

            # Generate embeddings in batches
            all_embeddings = []

            with tqdm(
                    desc="Generating embeddings",
                    total=len(valid_texts),
                    unit="text",
                    disable=not show_progress_bar
            ) as pbar:

                for i in range(0, len(valid_texts), self.batch_size):
                    batch_texts = valid_texts[i:i + self.batch_size]

                    try:
                        # Generate embeddings for batch
                        batch_embeddings = self.model.encode(
                            batch_texts,
                            convert_to_numpy=True,
                            show_progress_bar=False,
                            normalize_embeddings=True  # Normalize for better similarity search
                        )

                        all_embeddings.append(batch_embeddings)
                        pbar.update(len(batch_texts))

                    except Exception as e:
                        self.logger.error(f"Failed to process batch {i // self.batch_size}: {e}")
                        raise EmbeddingError(f"Batch embedding failed: {e}") from e

            # Combine all embeddings
            embeddings = np.vstack(all_embeddings)

            # Create full array with zeros for empty texts
            full_embeddings = np.zeros((len(texts), embeddings.shape[1]))
            full_embeddings[valid_indices] = embeddings

            generation_time = self.performance_tracker.end_timer('embedding_generation')

            self.logger.info(
                f"Successfully generated {embeddings.shape[0]} embeddings "
                f"({embeddings.shape[1]}D) in {generation_time:.2f}s"
            )

            return full_embeddings

        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            raise EmbeddingError(f"Failed to generate embeddings: {e}") from e


    # Generate embeddings for document chunks with metadata.
    def generate_chunk_embeddings(
            self,
            chunks: List[DocumentChunk],
            show_progress_bar: bool = True
    ) -> List[Dict[str, Any]]:

        try:
            validate_input(chunks, list, "chunks")

            if not chunks:
                self.logger.warning("No chunks provided for embedding")
                return []

            self.logger.info(f"Processing {len(chunks)} chunks for embedding")

            # Extract texts from chunks
            texts = [chunk.text for chunk in chunks]

            # Generate embeddings
            embeddings = self.generate_embeddings(texts, show_progress_bar)

            # Combine embeddings with chunk metadata
            embedded_chunks = []

            for chunk, embedding in zip(chunks, embeddings):
                # Skip if embedding is all zeros (empty text)
                if np.any(embedding):
                    embedded_chunk = {
                        'id': chunk.chunk_id,
                        'text': chunk.text,
                        'embedding': embedding,
                        'metadata': {
                            **asdict(chunk),
                            'embedding_model': self.model_name,
                            'embedding_dimension': embedding.shape[0]
                        }
                    }
                    embedded_chunks.append(embedded_chunk)
                else:
                    self.logger.warning(f"Skipping chunk {chunk.chunk_id} due to empty embedding")

            self.logger.info(f"Successfully embedded {len(embedded_chunks)} chunks")

            return embedded_chunks

        except Exception as e:
            self.logger.error(f"Chunk embedding failed: {e}")
            raise EmbeddingError(f"Failed to embed chunks: {e}") from e


    #  Generate embedding for a single query
    def generate_query_embedding(self, query: str) -> np.ndarray:

        try:
            validate_input(query, str, "query")

            if not query.strip():
                raise EmbeddingError("Cannot embed empty query")

            if not self.model_loaded:
                raise EmbeddingError("Model not loaded")

            self.logger.debug(f"Generating embedding for query: {query[:100]}...")

            # Generate embedding
            embedding = self.model.encode(
                [query.strip()],
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )[0]  # Get first (and only) embedding

            return embedding

        except Exception as e:
            self.logger.error(f"Query embedding failed: {e}")
            raise EmbeddingError(f"Failed to embed query: {e}") from e

    # # Compute cosine similarity between query and document embeddings.
    # def compute_similarity(
    #         self,
    #         query_embedding: np.ndarray,
    #         document_embeddings: np.ndarray
    # ) -> np.ndarray:
    #
    #     try:
    #         # Ensure embeddings are normalized
    #         query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    #         doc_norms = document_embeddings / (np.linalg.norm(document_embeddings, axis=1, keepdims=True) + 1e-8)
    #
    #         # Compute cosine similarity
    #         similarities = np.dot(doc_norms, query_norm)
    #
    #         return similarities
    #
    #     except Exception as e:
    #         self.logger.error(f"Similarity computation failed: {e}")
    #         raise EmbeddingError(f"Failed to compute similarity: {e}") from e


    # Get information about the loaded model
    def get_model_info(self) -> Dict[str, Any]:

        if not self.model_loaded:
            return {'status': 'not_loaded'}

        return {
            'status': 'loaded',
            'model_name': self.model_name,
            'max_seq_length': getattr(self.model, 'max_seq_length', 'Unknown'),
            'embedding_dimension': self.model.get_sentence_embedding_dimension(),
            'device': str(self.model.device),
            'batch_size': self.batch_size
        }


    # Get embedding performance statistics
    def get_performance_stats(self) -> Dict[str, Any]:
        return {
            'performance_metrics': self.performance_tracker.metrics,
            'model_info': self.get_model_info()
        }