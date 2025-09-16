# Main RAG pipeline
# Coordinates JSON document processing, embedding generation, vector storage, and response generation.


import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from document_processor import JSONDocumentProcessor  # Updated import
from embeddings import EmbeddingGenerator
from vector_store import ChromaVectorStore
from llm_interface import LLMInterface
from utils import (
    ConfigManager, Logger, PerformanceTracker,
    FileManager, validate_input
)
from exceptions import RAGBaseException


class RAGPipeline:
    #Main RAG system pipeline orchestrator - Updated for JSON processing

    #   Initialize the RAG pipeline with all components
    def __init__(self, config_path: str = "config/config.json"):

        # Load configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config

        # Setup logging
        self.logger_setup = Logger(self.config)
        self.logger = logging.getLogger(__name__)

        # Initialize performance tracking
        self.performance_tracker = PerformanceTracker()

        # Initialize pipeline components
        self.logger.info("Initializing RAG pipeline components...")

        try:
            self.document_processor = JSONDocumentProcessor(self.config)  # Updated to JSON processor
            self.embedding_generator = EmbeddingGenerator(self.config)
            self.vector_store = ChromaVectorStore(self.config)
            self.llm_interface = LLMInterface(self.config)

            self.logger.info("RAG pipeline initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize RAG pipeline: {e}")
            raise RAGBaseException(f"Pipeline initialization failed: {e}") from e

        # Configuration for retrieval
        self.retrieval_config = self.config.get('retrieval', {})
        self.top_k = self.retrieval_config.get('top_k', 5)
        self.similarity_threshold = self.retrieval_config.get('similarity_threshold', 0.0)

        # Output configuration
        self.output_config = self.config.get('output', {})
        self.save_to_file = self.output_config.get('save_to_file', True)
        self.print_to_console = self.output_config.get('print_to_console', True)
        self.output_directory = self.output_config.get('output_directory', 'data/outputs')

    #   Add JSON documents to the RAG system
    def add_documents(self, json_paths: List[str]) -> bool:

        try:
            validate_input(json_paths, list, "json_paths")

            if not json_paths:
                self.logger.warning("No JSON paths provided")
                return True

            self.logger.info(f"Adding {len(json_paths)} JSON documents to RAG system")
            self.performance_tracker.start_timer('total_document_addition')

            # Validate JSON files exist
            valid_paths = []
            for json_path in json_paths:
                path = Path(json_path)
                if path.exists() and path.suffix.lower() == '.json':
                    valid_paths.append(str(path.absolute()))
                else:
                    self.logger.warning(f"Skipping invalid JSON path: {json_path}")

            if not valid_paths:
                raise RAGBaseException("No valid JSON files found")

            # Step 1: Process JSON files and create chunks
            self.logger.info("Step 1: Processing JSON documents...")
            chunks = self.document_processor.process_multiple_jsons(valid_paths)

            if not chunks:
                raise RAGBaseException("No chunks created from JSON processing")

            # Step 2: Generate embeddings for chunks
            self.logger.info("Step 2: Generating embeddings...")
            embedded_chunks = self.embedding_generator.generate_chunk_embeddings(chunks)

            if not embedded_chunks:
                raise RAGBaseException("No embeddings generated")

            # Step 3: Add to vector store
            self.logger.info("Step 3: Adding to vector store...")
            success = self.vector_store.add_documents(embedded_chunks)

            if not success:
                raise RAGBaseException("Failed to add documents to vector store")

            total_time = self.performance_tracker.end_timer('total_document_addition')

            # Log success summary
            self.logger.info(
                f"Successfully added documents to RAG system:\n"
                f"  - JSON files processed: {len(valid_paths)}\n"
                f"  - Chunks created: {len(chunks)}\n"
                f"  - Embeddings generated: {len(embedded_chunks)}\n"
                f"  - Total processing time: {total_time:.2f}s"
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to add documents: {e}")
            raise RAGBaseException(f"Document addition failed: {e}") from e

    #   Query the RAG system with a question
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        #   Returns dictionary containing answer and metadata
        
        try:
            validate_input(question, str, "question")

            if not question.strip():
                raise RAGBaseException("Cannot process empty question")

            # Use provided top_k or default from config
            k = top_k if top_k is not None else self.top_k

            self.logger.info(f"Processing query: {question[:100]}...")
            self.performance_tracker.start_timer('total_query_processing')

            # Step 1: Generate query embedding
            self.logger.debug("Step 1: Generating query embedding...")
            self.performance_tracker.start_timer('query_embedding')

            query_embedding = self.embedding_generator.generate_query_embedding(question)

            embedding_time = self.performance_tracker.end_timer('query_embedding')

            # Step 2: Retrieve relevant documents
            self.logger.debug("Step 2: Retrieving relevant documents...")
            self.performance_tracker.start_timer('document_retrieval')

            retrieved_docs = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=k
            )

            # Filter by similarity threshold
            if self.similarity_threshold > 0:
                retrieved_docs = [
                    doc for doc in retrieved_docs
                    if doc.get('similarity', 0) >= self.similarity_threshold
                ]

            retrieval_time = self.performance_tracker.end_timer('document_retrieval')

            # Step 3: Generate response
            self.logger.debug("Step 3: Generating response...")
            self.performance_tracker.start_timer('response_generation')

            if not retrieved_docs:
                # No relevant documents found
                response = {
                    'query': question,
                    'answer': "I couldn't find any relevant information in the documents to answer your question. Please try rephrasing your question or check if the relevant documents have been added to the system.",
                    'sources_count': 0,
                    'sources': [],
                    'retrieval_successful': False
                }
            else:
                response = self.llm_interface.generate_response(
                    query=question,
                    retrieved_docs=retrieved_docs,
                    include_sources=include_sources
                )
                response['retrieval_successful'] = True

            generation_time = self.performance_tracker.end_timer('response_generation')
            total_time = self.performance_tracker.end_timer('total_query_processing')

            # Add timing information
            response['performance'] = {
                'query_embedding_time': embedding_time,
                'document_retrieval_time': retrieval_time,
                'response_generation_time': generation_time,
                'total_time': total_time,
                'documents_retrieved': len(retrieved_docs)
            }

            # Output handling
            self._handle_output(response)

            self.logger.info(
                f"Query processed successfully in {total_time:.2f}s "
                f"(retrieved {len(retrieved_docs)} documents)"
            )

            return response

        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            raise RAGBaseException(f"Query failed: {e}") from e

    def interactive_session(self):
        #Start an interactive Q&A session
        self.logger.info("Starting interactive RAG session...")
        print("\n" + "="*60)
        print("RAG System Interactive Session (JSON Processing)")
        print("="*60)
        print("Type 'quit', 'exit', or 'q' to end the session")
        print("Type 'help' for available commands")
        print("Type 'stats' to see system statistics")
        print("-"*60)

        session_count = 0

        while True:
            try:
                # Get user input
                question = input("\nEnter your question: ").strip()

                # Handle special commands
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\nEnding interactive session...")
                    break

                elif question.lower() == 'help':
                    self._show_help()
                    continue

                elif question.lower() == 'stats':
                    self._show_stats()
                    continue

                elif not question:
                    print("Please enter a question.")
                    continue

                # Process query
                session_count += 1
                print(f"\nProcessing question {session_count}...")

                try:
                    response = self.query(question, include_sources=True)
                    self._display_interactive_response(response)

                except Exception as e:
                    print(f"Error processing query: {e}")
                    self.logger.error(f"Interactive query failed: {e}")

            except KeyboardInterrupt:
                print("\n\nSession interrupted by user.")
                break

            except EOFError:
                print("\nSession ended.")
                break

        print(f"\nSession completed. Processed {session_count} questions.")
        self._show_final_stats()

    #   Handle response output (console and/or file)
    def _handle_output(self, response: Dict[str, Any]):

        # Format response for output
        output_text = self._format_output_text(response)

        # Print to console if enabled
        if self.print_to_console:
            print("\n" + "="*60)
            print("RAG SYSTEM RESPONSE")
            print("="*60)
            print(output_text)
            print("="*60)

        # Save to file if enabled
        if self.save_to_file:
            try:
                query_text = response.get('query', 'unknown_query')
                safe_filename = f"rag_response_{query_text[:50]}"

                FileManager.save_text_output(
                    content=output_text,
                    filename=safe_filename,
                    output_dir=self.output_directory
                )

            except Exception as e:
                self.logger.error(f"Failed to save output to file: {e}")

    #   Format response for text output
    def _format_output_text(self, response: Dict[str, Any]) -> str:

        lines = []

        # Query
        lines.append(f"QUESTION: {response.get('query', 'N/A')}")
        lines.append("")

        # Answer
        lines.append("ANSWER:")
        lines.append(response.get('answer', 'No answer generated'))
        lines.append("")

        # Performance info
        performance = response.get('performance', {})
        lines.append("PERFORMANCE:")
        lines.append(f"  - Total time: {performance.get('total_time', 0):.2f}s")
        lines.append(f"  - Documents retrieved: {performance.get('documents_retrieved', 0)}")
        lines.append(f"  - Retrieval successful: {response.get('retrieval_successful', False)}")

        # Usage info if available
        usage = response.get('usage', {})
        if usage:
            lines.append(f"  - Prompt tokens: {usage.get('prompt_tokens', 0)}")
            lines.append(f"  - Completion tokens: {usage.get('completion_tokens', 0)}")

        lines.append("")

        # Sources
        sources = response.get('sources', [])
        if sources:
            lines.append("SOURCES:")
            for source in sources:
                lines.append(f"  {source['source_number']}. {source['file']}")
                if source['pages']:
                    pages = source['pages']
                    if len(pages) == 1:
                        lines.append(f"     Page: {pages[0]}")
                    else:
                        lines.append(f"     Pages: {pages[0]}-{pages[-1]}")
                lines.append(f"     Similarity: {source['similarity']:.3f}")
                lines.append(f"     Preview: {source['preview']}")
                lines.append("")
        else:
            lines.append("SOURCES: No sources found")

        # Timestamp
        lines.append(f"Generated at: {time.ctime(response.get('timestamp', time.time()))}")

        return "\n".join(lines)

    #   Display response in interactive mode
    def _display_interactive_response(self, response: Dict[str, Any]):

        print("\n" + "-"*50)
        print("ANSWER:")
        print("-"*50)
        print(response.get('answer', 'No answer generated'))

        # Show performance
        performance = response.get('performance', {})
        print(f"\n⏱️  Processed in {performance.get('total_time', 0):.2f}s")
        print(f"📄 Retrieved {performance.get('documents_retrieved', 0)} documents")

        # Show sources summary
        sources = response.get('sources', [])
        if sources:
            print(f"📚 Sources: {len(sources)} documents")
            for i, source in enumerate(sources[:3], 1):  # Show first 3 sources
                print(f"   {i}. {source['file']} (similarity: {source['similarity']:.2f})")
            if len(sources) > 3:
                print(f"   ... and {len(sources) - 3} more")
                
    #   Show help information
    def _show_help(self):
        print("\nAvailable commands:")
        print("  quit, exit, q  - End the session")
        print("  help          - Show this help message")
        print("  stats         - Show system statistics")
        print("\nJust type your question to get an answer based on the loaded documents.")
        
    #   Show system statistics
    def _show_stats(self):
        print("\n" + "="*40)
        print("SYSTEM STATISTICS")
        print("="*40)

        # Collection info
        collection_info = self.vector_store.get_collection_info()
        print(f"Documents in database: {collection_info.get('document_count', 0)}")

        # Model info
        model_info = self.embedding_generator.get_model_info()
        print(f"Embedding model: {model_info.get('model_name', 'Unknown')}")
        print(f"Embedding dimension: {model_info.get('embedding_dimension', 'Unknown')}")

        # Usage stats
        usage_stats = self.llm_interface.get_usage_stats()
        print(f"Total API requests: {usage_stats.get('total_requests', 0)}")
        print(f"Total tokens used: {usage_stats.get('total_prompt_tokens', 0) + usage_stats.get('total_completion_tokens', 0)}")
        print(f"Estimated cost: ${usage_stats.get('total_cost_estimate', 0):.4f}")

        print("="*40)

    #   Show final statistics at end of session
    def _show_final_stats(self):
        print("\n" + "="*50)
        print("SESSION SUMMARY")
        print("="*50)

        # Performance metrics
        self.performance_tracker.log_metrics()

        # Usage statistics
        usage_stats = self.llm_interface.get_usage_stats()
        if usage_stats['total_requests'] > 0:
            print(f"\nAPI Usage Summary:")
            print(f"  - Total requests: {usage_stats['total_requests']}")
            print(f"  - Average response time: {usage_stats.get('average_response_time', 'N/A')}")
            print(f"  - Total cost estimate: ${usage_stats['total_cost_estimate']:.4f}")

    #   Get comprehensive system status information
    def get_system_status(self) -> Dict[str, Any]:
        
        try:
            return {
                'config': {
                    'embedding_model': self.config.get('models', {}).get('embedding_model'),
                    'llm_model': self.config.get('models', {}).get('llm_model'),
                    'chunk_size': self.config.get('chunking', {}).get('chunk_size_tokens'),
                    'top_k': self.top_k
                },
                'collection_info': self.vector_store.get_collection_info(),
                'model_info': self.embedding_generator.get_model_info(),
                'usage_stats': self.llm_interface.get_usage_stats(),
                'performance_stats': {
                    'document_processor': self.document_processor.performance_tracker.metrics,
                    'embedding_generator': self.embedding_generator.get_performance_stats(),
                    'vector_store': self.vector_store.get_performance_stats(),
                    'llm_interface': self.llm_interface.get_performance_stats()
                }
            }
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {'error': str(e)}

    #   Reset the entire system (clear all documents and statistics)
    def reset_system(self, confirm: bool = False) -> bool:

        if not confirm:
            self.logger.warning("Reset not confirmed - no action taken")
            return False

        try:
            self.logger.info("Resetting RAG system...")

            # Clear vector store
            self.vector_store.delete_collection()

            # Reset usage statistics
            self.llm_interface.reset_usage_stats()

            # Reset performance trackers
            self.performance_tracker = PerformanceTracker()

            self.logger.info("System reset completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"System reset failed: {e}")
            raise RAGBaseException(f"Failed to reset system: {e}") from e