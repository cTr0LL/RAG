"""
Main entry point for the RAG system - Updated for JSON processing.
Provides a simple interface to add JSON documents and query the system.
"""

import os
import sys
from pathlib import Path
from typing import List

# Add src directory to path so we can import our modules
sys.path.append(str(Path(__file__).parent / "src"))

from src.rag_pipeline import RAGPipeline
from src.exceptions import RAGBaseException


def setup_environment():
    """Setup environment and check requirements."""
    # Create necessary directories
    directories = [
        "data/json_documents",  # Changed from pdfs to json_documents
        "data/processed",
        "data/chroma_db",
        "data/outputs",
        "logs",
        "config"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    print("   Environment setup complete")
    return True


def main():
    """Main function with example usage."""
    print("="*60)
    print("RAG SYSTEM - Academic Document Q&A (JSON Processing)")
    print("="*60)

    # Setup environment
    if not setup_environment():
        print("   Environment setup failed. Please check your configuration.")
        return

    try:
        # Initialize RAG pipeline
        print("\n   Initializing RAG pipeline...")
        rag = RAGPipeline()

        # Show system status
        print("\n   System Status:")
        status = rag.get_system_status()
        collection_info = status.get('collection_info', {})
        print(f"   Documents in database: {collection_info.get('document_count', 0)}")
        print(f"   Embedding model: {status.get('config', {}).get('embedding_model', 'N/A')}")
        print(f"   LLM model: {status.get('config', {}).get('llm_model', 'N/A')}")

        # Main interaction loop
        while True:
            print("\n" + "="*50)
            print("What would you like to do?")
            print("1. Add JSON documents to the system")
            print("2. Query the system (single question)")
            print("3. Start interactive Q&A session")
            print("4. Show system statistics")
            print("5. Reset system (clear all documents)")
            print("6. Exit")
            print("-"*50)

            try:
                choice = input("Enter your choice (1-6): ").strip()

                if choice == '1':
                    add_documents_interface(rag)

                elif choice == '2':
                    single_query_interface(rag)

                elif choice == '3':
                    rag.interactive_session()

                elif choice == '4':
                    show_statistics_interface(rag)

                elif choice == '5':
                    reset_system_interface(rag)

                elif choice == '6':
                    print("\n   Thank you for using the RAG system!")
                    break

                else:
                    print("   Invalid choice. Please enter a number between 1-6.")

            except KeyboardInterrupt:
                print("\n\n   Goodbye!")
                break

            except EOFError:
                print("\n\n   Goodbye!")
                break

    except RAGBaseException as e:
        print(f"   RAG System Error: {e}")
        print("Please check your configuration and try again.")

    except Exception as e:
        print(f"   Unexpected Error: {e}")
        print("Please check the logs for more details.")


def add_documents_interface(rag: RAGPipeline):
    """Interface for adding JSON documents."""
    print("\n   ADD JSON DOCUMENTS")
    print("-"*30)

    # Option 1: Add files from data/json_documents directory
    json_dir = Path("data/json_documents")
    existing_jsons = list(json_dir.glob("*.json"))

    if existing_jsons:
        print(f"\nFound {len(existing_jsons)} JSON files in data/json_documents/:")
        for i, json_file in enumerate(existing_jsons, 1):
            print(f"  {i}. {json_file.name}")

        add_existing = input("\nAdd these JSON files to the system? (y/n): ").strip().lower()
        if add_existing in ['y', 'yes']:
            try:
                json_paths = [str(json_file) for json_file in existing_jsons]
                print(f"\n   Processing {len(json_paths)} JSON files...")
                rag.add_documents(json_paths)
                print("   Documents added successfully!")
                return
            except Exception as e:
                print(f"   Failed to add documents: {e}")
                return

    # Option 2: Specify custom paths
    print("\nTo add JSON files:")
    print("1. Copy your JSON files (from MinerU web app) to the 'data/json_documents/' directory, OR")
    print("2. Enter the full path to your JSON file(s)")

    custom_path = input("\nEnter JSON file path (or press Enter to skip): ").strip()

    if custom_path:
        json_path = Path(custom_path)
        if json_path.exists() and json_path.suffix.lower() == '.json':
            try:
                print(f"\n   Processing {json_path.name}...")
                rag.add_documents([str(json_path)])
                print("   Document added successfully!")
            except Exception as e:
                print(f"   Failed to add document: {e}")
        else:
            print("   Invalid JSON file path or file does not exist.")
    else:
        print("   Copy your JSON files to 'data/json_documents/' and try again.")


def single_query_interface(rag: RAGPipeline):
    """Interface for single query."""
    print("\n   SINGLE QUERY")
    print("-"*20)

    # Check if there are documents in the system
    status = rag.get_system_status()
    doc_count = status.get('collection_info', {}).get('document_count', 0)

    if doc_count == 0:
        print("   No documents found in the system.")
        print("   Please add some JSON documents first (option 1).")
        return

    print(f"   System contains {doc_count} document chunks")

    question = input("\nEnter your question: ").strip()

    if not question:
        print("   No question entered.")
        return

    # The response is automatically handled by the pipeline
    # (printed to console and saved to file based on config)
    try:
        print(f"\n   Processing your question...")
        response = rag.query(question)

    except Exception as e:
        print(f"   Query failed: {e}")


def show_statistics_interface(rag: RAGPipeline):
    """Interface for showing system statistics."""
    print("\n   SYSTEM STATISTICS")
    print("-"*25)

    try:
        status = rag.get_system_status()

        # Configuration
        config = status.get('config', {})
        print("Configuration:")
        print(f"  - Embedding model: {config.get('embedding_model', 'N/A')}")
        print(f"  - LLM model: {config.get('llm_model', 'N/A')}")
        print(f"  - Chunk size: {config.get('chunk_size', 'N/A')} tokens")
        print(f"  - Retrieval top-k: {config.get('top_k', 'N/A')}")

        # Collection info
        collection_info = status.get('collection_info', {})
        print(f"\nDocument Database:")
        print(f"  - Total chunks: {collection_info.get('document_count', 0)}")
        print(f"  - Collection: {collection_info.get('collection_name', 'N/A')}")
        print(f"  - Distance function: {collection_info.get('distance_function', 'N/A')}")

        # Usage statistics
        usage_stats = status.get('usage_stats', {})
        if usage_stats.get('total_requests', 0) > 0:
            print(f"\nAPI Usage:")
            print(f"  - Total requests: {usage_stats.get('total_requests', 0)}")
            print(f"  - Total tokens: {usage_stats.get('total_prompt_tokens', 0) + usage_stats.get('total_completion_tokens', 0)}")
            print(f"  - Estimated cost: ${usage_stats.get('total_cost_estimate', 0):.4f}")
        else:
            print(f"\nAPI Usage: No queries processed yet")

    except Exception as e:
        print(f"   Failed to get statistics: {e}")


def reset_system_interface(rag: RAGPipeline):
    """Interface for resetting the system."""
    print("\n   RESET SYSTEM")
    print("-"*20)
    print("   This will permanently delete all documents and statistics!")

    confirm1 = input("Are you sure you want to reset? (type 'yes' to confirm): ").strip()

    if confirm1.lower() != 'yes':
        print("   Reset cancelled.")
        return

    confirm2 = input("This action cannot be undone. Type 'RESET' to proceed: ").strip()

    if confirm2 != 'RESET':
        print("   Reset cancelled.")
        return

    try:
        print("   Resetting system...")
        success = rag.reset_system(confirm=True)

        if success:
            print("   System reset successfully!")
        else:
            print("   Reset failed.")

    except Exception as e:
        print(f"   Reset failed: {e}")


if __name__ == "__main__":
    main()