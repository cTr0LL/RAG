class RAGBaseException(Exception):
    #Base exception class for all RAG-related errors
    pass


class DocumentProcessingError(RAGBaseException):
    #Raised when document processing fails
    pass


class PDFExtractionError(DocumentProcessingError):
    #Raised when PDF text extraction fails
    pass


class ChunkingError(DocumentProcessingError):
    #Raised when text chunking fails
    pass


class EmbeddingError(RAGBaseException):
    #Raised when embedding generation or processing fails
    pass


class ModelLoadError(EmbeddingError):
    #Raised when embedding model fails to load
    pass


class VectorStoreError(RAGBaseException):
    #Raised when vector store operations fail
    pass


class CollectionError(VectorStoreError):
    #Raised when collection operations fail
    pass


class RetrievalError(VectorStoreError):
    #Raised when document retrieval fails
    pass


class LLMError(RAGBaseException):
    #Raised when LLM operations fail
    pass


class APIError(LLMError):
    #Raised when API calls fail
    pass


class TokenLimitError(LLMError):
    #Raised when token limits are exceeded
    pass


class ConfigurationError(RAGBaseException):
    #Raised when configuration is invalid
    pass


class ValidationError(RAGBaseException):
    #Raised when input validation fails
    pass