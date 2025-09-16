# Vector storage and retrieval module using ChromaDB.
# Handles document storage, similarity search, and metadata management.

import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import uuid

import chromadb
from chromadb.config import Settings
import numpy as np
from tqdm import tqdm

from utils import PerformanceTracker, validate_input, FileManager
from exceptions import VectorStoreError, CollectionError, RetrievalError


    #   Local ChromaDB-based vector storage for document embeddings
class ChromaVectorStore:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vector_config = config.get('vector_store', {})

        # Configuration
        self.collection_name = self.vector_config.get('collection_name', 'academic_documents')
        self.persist_directory = self.vector_config.get('persist_directory', 'data/chroma_db')
        self.distance_function = self.vector_config.get('distance_function', 'cosine')

        # Initialize components
        self.client = None
        self.collection = None
        self.performance_tracker = PerformanceTracker()
        self.logger = logging.getLogger(__name__)

        # Setup vector store
        self._initialize_client()
        self._initialize_collection()
    
    #   Initialize ChromaDB client with persistence
    def _initialize_client(self):

        try:
            # Ensure persistence directory exists
            persist_path = FileManager.ensure_directory(self.persist_directory)

            self.logger.info(f"Initializing ChromaDB client at: {persist_path}")

            # Configure ChromaDB settings
            settings = Settings(
                persist_directory=str(persist_path),
                anonymized_telemetry=False,
                is_persistent=True
            )

            # Create client
            self.client = chromadb.PersistentClient(
                path=str(persist_path),
                settings=settings
            )

            self.logger.info("ChromaDB client initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise VectorStoreError(f"Client initialization failed: {e}") from e

    def _initialize_collection(self):
        #Initialize or load the document collection
        try:
            # First, try to create the collection (this is safer than trying to get first)
            try:
                self.collection = self.client.create_collection(
                    name=self.collection_name
                )
                self.logger.info(f"Created new collection: {self.collection_name}")

            except Exception as create_error:
                # If creation failed, the collection might already exist - try to get it
                try:
                    self.collection = self.client.get_collection(
                        name=self.collection_name
                    )
                    collection_count = self.collection.count()
                    self.logger.info(
                        f"Loaded existing collection '{self.collection_name}' "
                        f"with {collection_count} documents"
                    )

                except Exception as get_error:
                    # Both create and get failed
                    self.logger.error(f"Failed to create collection: {create_error}")
                    self.logger.error(f"Failed to get collection: {get_error}")
                    raise CollectionError(
                        f"Could not create or access collection '{self.collection_name}'. "
                        f"Create error: {create_error}. Get error: {get_error}"
                    )

        except Exception as e:
            self.logger.error(f"Failed to initialize collection: {e}")
            raise CollectionError(f"Collection initialization failed: {e}") from e

    #   Add embedded documents to the vector store
    def add_documents(
            self,
            embedded_chunks: List[Dict[str, Any]],
            show_progress: bool = True
    ) -> bool:
        #   embedded_chunks: List of dictionaries with embeddings and metadata
        #    show_progress: Whether to show progress bar

        #   Returns True if success

        try:
            validate_input(embedded_chunks, list, "embedded_chunks")

            if not embedded_chunks:
                self.logger.warning("No documents to add")
                return True

            self.logger.info(f"Adding {len(embedded_chunks)} documents to vector store")
            self.performance_tracker.start_timer('document_insertion')

            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            metadatas = []
            documents = []

            with tqdm(
                    desc="Preparing documents",
                    total=len(embedded_chunks),
                    unit="doc",
                    disable=not show_progress
            ) as pbar:

                for chunk_data in embedded_chunks:
                    try:
                        # Generate unique ID if not present
                        chunk_id = chunk_data.get('id', str(uuid.uuid4()))

                        # Prepare embedding
                        embedding = chunk_data['embedding']
                        if isinstance(embedding, np.ndarray):
                            embedding = embedding.tolist()

                        # Prepare metadata (ChromaDB doesn't support nested dicts)
                        metadata = self._flatten_metadata(chunk_data.get('metadata', {}))

                        # Add to lists
                        ids.append(chunk_id)
                        embeddings.append(embedding)
                        metadatas.append(metadata)
                        documents.append(chunk_data['text'])

                        pbar.update(1)

                    except Exception as e:
                        self.logger.error(f"Failed to prepare chunk {chunk_data.get('id', 'unknown')}: {e}")
                        continue

            if not ids:
                raise VectorStoreError("No valid documents to add")

            # Add to ChromaDB in batches
            batch_size = 100  # ChromaDB recommended batch size

            with tqdm(
                    desc="Inserting to database",
                    total=len(ids),
                    unit="doc",
                    disable=not show_progress
            ) as pbar:

                for i in range(0, len(ids), batch_size):
                    batch_slice = slice(i, i + batch_size)

                    try:
                        self.collection.add(
                            ids=ids[batch_slice],
                            embeddings=embeddings[batch_slice],
                            metadatas=metadatas[batch_slice],
                            documents=documents[batch_slice]
                        )

                        pbar.update(len(ids[batch_slice]))

                    except Exception as e:
                        self.logger.error(f"Failed to insert batch {i // batch_size}: {e}")
                        raise VectorStoreError(f"Batch insertion failed: {e}") from e

            insertion_time = self.performance_tracker.end_timer('document_insertion')

            # Log success
            total_docs = self.collection.count()
            self.logger.info(
                f"Successfully added {len(ids)} documents in {insertion_time:.2f}s. "
                f"Total documents in collection: {total_docs}"
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to add documents: {e}")
            raise VectorStoreError(f"Document addition failed: {e}") from e

    #   Search for similar documents using query embedding
    def search(
            self,
            query_embedding: np.ndarray,
            top_k: int = 5,
            filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:

        #    query_embedding: Query embedding vector
        #    top_k: return top_k results
        #    filters: Optional metadata filters

        #   Returns list of search results with documents and metadata

        try:
            validate_input(query_embedding, np.ndarray, "query_embedding")
            validate_input(top_k, int, "top_k")

            if top_k <= 0:
                raise RetrievalError("top_k must be positive")

            self.logger.debug(f"Searching for top {top_k} similar documents")
            self.performance_tracker.start_timer('document_search')

            # Convert embedding to list for ChromaDB
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()

            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filters,  # Metadata filters
                include=['documents', 'metadatas', 'distances']
            )

            search_time = self.performance_tracker.end_timer('document_search')

            # Process results
            processed_results = []

            if results['documents'] and results['documents'][0]:  # Check if results exist
                for i in range(len(results['documents'][0])):
                    result = {
                        'id': results['ids'][0][i],
                        'document': results['documents'][0][i],
                        'metadata': self._unflatten_metadata(results['metadatas'][0][i]),
                        'distance': results['distances'][0][i],
                        'similarity': 1 - results['distances'][0][i]  # Convert distance to similarity
                    }
                    processed_results.append(result)

            self.logger.debug(
                f"Found {len(processed_results)} results in {search_time:.3f}s"
            )

            return processed_results

        except Exception as e:
            self.logger.error(f"Document search failed: {e}")
            raise RetrievalError(f"Search failed: {e}") from e


    def delete_collection(self) -> bool:
        #Delete the entire collection
        try:
            self.client.delete_collection(name=self.collection_name)
            self.logger.info(f"Deleted collection: {self.collection_name}")

            # Reinitialize collection
            self._initialize_collection()
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete collection: {e}")
            raise VectorStoreError(f"Collection deletion failed: {e}") from e

    def get_collection_info(self) -> Dict[str, Any]:
        #Get information about the current collection
        try:
            count = self.collection.count()

            # Get a sample document to understand structure
            sample_results = self.collection.peek(limit=1)

            sample_metadata = {}
            if sample_results['metadatas'] and sample_results['metadatas'][0]:
                sample_metadata = sample_results['metadatas'][0]

            return {
                'collection_name': self.collection_name,
                'document_count': count,
                'distance_function': self.distance_function,
                'persist_directory': self.persist_directory,
                'sample_metadata_keys': list(sample_metadata.keys()) if sample_metadata else []
            }

        except Exception as e:
            self.logger.error(f"Failed to get collection info: {e}")
            return {'error': str(e)}

    def _flatten_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        #Flatten nested metadata for ChromaDB compatibility
        flattened = {}

        for key, value in metadata.items():
            if isinstance(value, (dict, list)):
                # Convert complex types to JSON strings
                flattened[key] = json.dumps(value)
            elif isinstance(value, (str, int, float, bool)):
                flattened[key] = value
            else:
                # Convert other types to string
                flattened[key] = str(value)

        return flattened

    def _unflatten_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        #Restore flattened metadata structure
        unflattened = {}

        for key, value in metadata.items():
            if isinstance(value, str):
                try:
                    # Try to parse JSON strings back to objects
                    unflattened[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    # Keep as string if not valid JSON
                    unflattened[key] = value
            else:
                unflattened[key] = value

        return unflattened

    def get_performance_stats(self) -> Dict[str, Any]:
        #Get vector store performance statistics
        return {
            'performance_metrics': self.performance_tracker.metrics,
            'collection_info': self.get_collection_info()
        }