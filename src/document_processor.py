# JSON document processing module for MinerU JSON output.

import logging
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd

from tqdm import tqdm

from utils import TokenCounter, clean_text, validate_input, PerformanceTracker
from exceptions import DocumentProcessingError, ChunkingError

#   Represents a chunk of document text with metadata.
@dataclass
class DocumentChunk:
    text: str
    chunk_id: str
    source_file: str
    page_numbers: List[int]
    token_count: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]

#   Processes MinerU JSON output into structured text
class MinerUJSONProcessor:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

    #   Process MinerU JSON file into clean text and metadata
    def process_json_file(self, json_path: str) -> Tuple[str, Dict[str, Any]]:

        #   Returns tuple of (extracted_text, metadata)

        try:
            validate_input(json_path, str, "json_path")

            json_file = Path(json_path)
            if not json_file.exists():
                raise DocumentProcessingError(f"JSON file not found: {json_path}")

            self.logger.info(f"Processing MinerU JSON: {json_file.name}")

            # Load JSON data
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract text and metadata based on MinerU structure
            extracted_text = self._extract_text_from_json(data)
            metadata = self._extract_metadata_from_json(data, json_file)

            if not extracted_text.strip():
                raise DocumentProcessingError("No text could be extracted from JSON")

            self.logger.info(f"Successfully extracted {len(extracted_text)} characters")
            return extracted_text, metadata

        except Exception as e:
            self.logger.error(f"JSON processing failed for {json_path}: {e}")
            raise DocumentProcessingError(f"Failed to process JSON: {e}") from e

    #   Extract clean text from MinerU JSON structure
    def _extract_text_from_json(self, data: Dict[str, Any]) -> str:
        text_parts = []

        # Handle MinerU JSON structure: pdf_info -> pages -> para_blocks
        pdf_info = data['pdf_info']
        if isinstance(pdf_info, list):
            # Sort pages by page_idx to maintain order
            sorted_pages = sorted(pdf_info, key=lambda x: x.get('page_idx', 0))

            for page in sorted_pages:
                page_text = self._extract_page_text(page)
                if page_text:
                    text_parts.append(page_text)

        # Join all text parts with proper spacing
        full_text = '\n\n'.join(text_parts)

        # Clean and normalize the text
        return self._clean_extracted_text(full_text)


    #   Extract text from a MinerU page structure
    def _extract_page_text(self, page: Dict[str, Any]) -> str:

        page_parts = []

        # Handle MinerU page structure with para_blocks
        para_blocks = page['para_blocks']
        # Sort blocks by their index if available to maintain order
        if para_blocks and isinstance(para_blocks, list):
            sorted_blocks = sorted(para_blocks, key=lambda x: x.get('index', 0))

            for block in sorted_blocks:
                block_text = self._extract_block_text(block)
                if block_text.strip():  # Only add non-empty text
                    page_parts.append(block_text)

        return '\n'.join(page_parts)

    #   Extract text from a MinerU block structure
    def _extract_block_text(self, block: Dict[str, Any]) -> str:
        if not isinstance(block, dict):
            return ""

        block_type = block.get('type', 'text')

        # Handle MinerU-specific block types
        if block_type == 'text':
            return self._extract_text_from_lines(block)

        elif block_type == 'title':
            # Extract title with proper formatting
            title_text = self._extract_text_from_lines(block)
            level = block.get('level', 1)
            # Add title markers for better structure recognition
            return f"\n{'#' * level} {title_text}\n" if title_text else ""

        elif block_type == 'table':
            return self._extract_table_from_blocks(block)

        elif block_type == 'image':
            return self._extract_image_info(block)

        elif block_type == 'interline_equation':
            return self._extract_equation_text(block)

        # Fallback: extract text from lines or content
        else:
            text = self._extract_text_from_lines(block)
            if not text:
                # Try direct content extraction
                text = block.get('content', block.get('text', ''))
            return text

    #   Extract text from MinerU lines structure
    def _extract_text_from_lines(self, block: Dict[str, Any]) -> str:
        text_parts = []

        # Handle lines structure (common in MinerU)
        if 'lines' in block:
            lines = block['lines']
            if isinstance(lines, list):
                for line in lines:
                    if isinstance(line, dict) and 'spans' in line:
                        # Extract text from spans
                        for span in line['spans']:
                            if isinstance(span, dict):
                                # Handle different span types
                                span_type = span.get('type', 'text')
                                content = span.get('content', '')

                                if span_type == 'text':
                                    text_parts.append(content)
                                elif span_type == 'inline_equation':
                                    # Include inline equations in a readable format
                                    text_parts.append(f" [{content}] ")
                                elif span_type == 'interline_equation':
                                    # Block equations on separate lines
                                    text_parts.append(f"\n{content}\n")
                                else:
                                    # Include other content types
                                    text_parts.append(content)

        # Fallback: direct text extraction
        elif 'text' in block:
            text_parts.append(block['text'])

        return ' '.join(text_parts)

    #   Extract table text from MinerU table blocks
    def _extract_table_from_blocks(self, table_block: Dict[str, Any]) -> str:
        # Check for HTML table content
        if 'blocks' in table_block:
            for block in table_block['blocks']:
                if isinstance(block, dict) and block.get('type') == 'table_body':
                    # Extract from lines -> spans -> html
                    lines = block.get('lines', [])
                    for line in lines:
                        if isinstance(line, dict) and 'spans' in line:
                            for span in line['spans']:
                                if isinstance(span, dict) and span.get('type') == 'table':
                                    html = span.get('html', '')
                                    if html:
                                        return self._convert_html_table_to_text(html)

        # Fallback: look for direct html content
        html = table_block.get('html', '')
        if html:
            return self._convert_html_table_to_text(html)

        # Final fallback: extract any text content
        return self._extract_text_from_lines(table_block)

    #   Convert HTML table to readable text format
    def _convert_html_table_to_text(self, html: str) -> str:

        if not html:
            return ""

        # Simple HTML table parsing
        # Remove HTML tags but preserve structure
        import re

        # Replace table row endings with newlines
        html = re.sub(r'</tr>', '\n', html, flags=re.IGNORECASE)
        # Replace cell separators with tabs
        html = re.sub(r'</td>', '\t', html, flags=re.IGNORECASE)
        html = re.sub(r'</th>', '\t', html, flags=re.IGNORECASE)
        # Remove remaining HTML tags
        html = re.sub(r'<[^>]+>', '', html)
        # Clean up whitespace
        html = re.sub(r'\s+', ' ', html)
        html = re.sub(r'\n\s*\n', '\n', html)

        return f"\n[TABLE]\n{html.strip()}\n[/TABLE]\n"

    #   Extract image caption and description
    def _extract_image_info(self, image_block: Dict[str, Any]) -> str:

        image_parts = []

        # Look for image caption in blocks
        if 'blocks' in image_block:
            for block in image_block['blocks']:
                if isinstance(block, dict):
                    block_type = block.get('type', '')
                    if block_type == 'image_caption':
                        caption = self._extract_text_from_lines(block)
                        if caption:
                            image_parts.append(f"[FIGURE CAPTION: {caption}]")
                    elif block_type == 'image_body':
                        # Could extract image path if needed
                        image_parts.append("[FIGURE]")

        # Fallback: look for direct caption
        caption = image_block.get('caption', image_block.get('alt_text', ''))
        if caption and not image_parts:
            image_parts.append(f"[FIGURE: {caption}]")
        elif not image_parts:
            image_parts.append("[FIGURE]")

        return '\n'.join(image_parts) + '\n'

    #   Extract equation text from MinerU equation blocks
    def _extract_equation_text(self, equation_block: Dict[str, Any]) -> str:

        # Extract from lines structure
        content = self._extract_text_from_lines(equation_block)
        if content:
            return f"\n[EQUATION: {content}]\n"

        # Fallback: direct content
        content = equation_block.get('content', '')
        if content:
            return f"\n[EQUATION: {content}]\n"

        return ""

    #   Extract text representation of tables
    def _extract_table_text(self, table_block: Dict[str, Any]) -> str:

        # Handle table structures
        if 'html' in table_block:
            # Parse HTML table (simplified)
            html = table_block['html']
            # Remove HTML tags and format as readable text
            clean_table = re.sub(r'<[^>]+>', ' ', html)
            return f"\nTable: {clean_table}\n"

        elif 'csv' in table_block:
            return f"\nTable: {table_block['csv']}\n"

        elif 'text' in table_block:
            return f"\nTable: {table_block['text']}\n"

        return ""

    #   Extract formula text (LateX and plain text)
    def _extract_formula_text(self, formula_block: Dict[str, Any]) -> str:

        if 'latex' in formula_block:
            return f" {formula_block['latex']} "
        elif 'text' in formula_block:
            return f" {formula_block['text']} "
        return ""

    #   Extract text from document structure
    def _extract_document_text(self, document: Dict[str, Any]) -> str:
        if isinstance(document, str):
            return document
        elif isinstance(document, dict):
            if 'content' in document:
                return self._extract_text_from_json(document['content'])
            elif 'text' in document:
                return document['text']
        return ""

    #   Recursively find text content in any JSON structure
    def _find_text_recursively(self, data: Any) -> List[str]:

        text_parts = []

        if isinstance(data, dict):
            # Look for common text keys
            for key in ['text', 'content', 'body', 'paragraph', 'line']:
                if key in data and isinstance(data[key], str):
                    text_parts.append(data[key])

            # Recurse into nested structures
            for value in data.values():
                if isinstance(value, (dict, list)):
                    text_parts.extend(self._find_text_recursively(value))

        elif isinstance(data, list):
            for item in data:
                text_parts.extend(self._find_text_recursively(item))

        elif isinstance(data, str) and len(data.strip()) > 10:
            # Only include substantial text content
            text_parts.append(data)

        return text_parts

    #   Clean and normalize extracted text
    def _clean_extracted_text(self, text: str) -> str:

        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)

        # Remove page headers/footers patterns
        text = re.sub(r'\n\d+\n', '\n', text)  # Standalone page numbers
        text = re.sub(r'\nPage \d+\n', '\n', text)

        # Fix common OCR issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Missing spaces

        # Clean up line breaks around punctuation
        text = re.sub(r'\n([.,:;!?])', r'\1', text)

        return text.strip()

    #   Extract metadata from MinerU JSON
    def _extract_metadata_from_json(
        self,
        data: Dict[str, Any],
        json_file: Path
    ) -> Dict[str, Any]:

        # Count different content types
        formula_count = self._count_content_type(data, 'formula')
        table_count = self._count_content_type(data, 'table')
        image_count = self._count_content_type(data, 'image')

        # Estimate page count
        page_count = self._estimate_page_count(data)

        # Extract document properties
        language = data.get('language', 'unknown')
        if isinstance(language, list) and language:
            language = language[0]

        metadata = {
            # Basic file info
            'source_file': json_file.name,
            'original_pdf': json_file.stem + '.pdf',  # Assume PDF name
            'file_path': str(json_file.absolute()),
            'file_size_mb': round(json_file.stat().st_size / (1024 * 1024), 2),
            'total_pages': page_count,

            # Processing info
            'extraction_method': 'mineru_json_processing',
            'extraction_timestamp': pd.Timestamp.now().isoformat(),

            # Content analysis
            'language_detected': language,
            'has_equations': formula_count > 0,
            'has_tables': table_count > 0,
            'has_images': image_count > 0,

            # Content counts
            'formula_count': formula_count,
            'table_count': table_count,
            'image_count': image_count,

            # Processing capabilities
            'supports_equations': True,
            'supports_tables': True,
            'supports_images': True,
            'academic_optimized': True,

            # MinerU specific
            'mineru_version': data.get('version', 'unknown'),
            'processing_stats': data.get('stats', {})
        }

        return metadata

    def _count_content_type(self, data: Dict[str, Any], content_type: str) -> int:
        #Count occurrences of specific content type
        count = 0

        def count_recursive(obj):
            nonlocal count
            if isinstance(obj, dict):
                if obj.get('type') == content_type:
                    count += 1
                elif content_type == 'formula' and ('latex' in obj or 'equation' in str(obj).lower()):
                    count += 1
                elif content_type == 'table' and ('table' in str(obj.get('type', '')).lower() or 'html' in obj):
                    count += 1
                elif content_type == 'image' and ('image' in str(obj.get('type', '')).lower()):
                    count += 1
                for value in obj.values():
                    count_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    count_recursive(item)

        count_recursive(data)
        return count

    def _estimate_page_count(self, data: Dict[str, Any]) -> int:
        #Estimate page count from JSON structure
        # Primary: look for page indicators
        if 'pdf_info' in data and isinstance(data['pdf_info'], list):
            return len(data['pdf_info'])
        elif 'pages' in data:
            return len(data['pages']) if isinstance(data['pages'], list) else 1
        elif 'page_count' in data:
            return data['page_count']
        else:
            # Fallback: estimate from content size
            content_size = len(str(data))
            estimated_pages = max(1, content_size // 10000)  # Rough estimate
            return estimated_pages


class TextChunker:
    #Text chunking optimized for academic content

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('chunking', {})
        self.chunk_size = self.config.get('chunk_size_tokens', 512)
        self.overlap = self.config.get('overlap_tokens', 50)
        self.respect_boundaries = self.config.get('respect_sentence_boundaries', True)
        self.min_chunk_size = self.config.get('min_chunk_size', 50)

        # Initialize token counter
        llm_model = config.get('models', {}).get('llm_model', 'gpt-3.5-turbo')
        self.token_counter = TokenCounter(llm_model)

        self.logger = logging.getLogger(__name__)

    def chunk_text(
        self,
        text: str,
        metadata: Dict[str, Any],
        source_file: str
    ) -> List[DocumentChunk]:
        #Create intelligent chunks optimized for academic content
        try:
            validate_input(text, str, "text")
            validate_input(metadata, dict, "metadata")
            validate_input(source_file, str, "source_file")

            if not text.strip():
                raise ChunkingError("Cannot chunk empty text")

            self.logger.info(f"Creating chunks for {source_file}")

            # Use sentence-aware chunking for better context preservation
            if self.respect_boundaries:
                chunks = self._chunk_by_academic_sections(text, source_file, metadata)
            else:
                chunks = self._chunk_by_tokens(text, source_file, metadata)

            # Filter chunks by minimum size
            valid_chunks = [
                chunk for chunk in chunks
                if chunk.token_count >= self.min_chunk_size
            ]

            self.logger.info(
                f"Created {len(valid_chunks)} chunks "
                f"(filtered {len(chunks) - len(valid_chunks)} small chunks)"
            )

            return valid_chunks

        except Exception as e:
            self.logger.error(f"Text chunking failed: {e}")
            raise ChunkingError(f"Failed to chunk text: {e}") from e

    def _chunk_by_academic_sections(
        self,
        text: str,
        source_file: str,
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        #Create chunks optimized for academic content structure

        # Split into logical sections first
        sections = self._split_into_academic_sections(text)

        chunks = []
        char_position = 0

        with tqdm(desc="Creating academic chunks", total=len(sections), unit="section") as pbar:
            for section in sections:
                section_chunks = self._process_section(
                    section, source_file, metadata, len(chunks), char_position
                )
                chunks.extend(section_chunks)
                char_position += len(section) + 1
                pbar.update(1)

        return chunks

    def _split_into_academic_sections(self, text: str) -> List[str]:
        #Split text into academic sections
        # Split by double newlines (paragraph boundaries)
        paragraphs = re.split(r'\n\s*\n', text)

        sections = []
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue

            # If paragraph is very long, split by sentences
            if len(paragraph) > 1500:
                sentences = self._split_into_sentences(paragraph)
                current_section = ""

                for sentence in sentences:
                    if len(current_section + sentence) > 1200:
                        if current_section:
                            sections.append(current_section.strip())
                        current_section = sentence
                    else:
                        current_section += " " + sentence if current_section else sentence

                if current_section:
                    sections.append(current_section.strip())
            else:
                sections.append(paragraph.strip())

        return [s for s in sections if s]

    def _split_into_sentences(self, text: str) -> List[str]:
        #Split text into sentences, handling academic content
        # Handle common abbreviations
        text = re.sub(r'\b(Dr|Prof|Mr|Mrs|Ms|vs|etc|Fig|Table|Eq)\.\s', r'\1<DOT> ', text)

        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+\s+', text)

        # Restore abbreviations
        sentences = [s.replace('<DOT>', '.') for s in sentences]

        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    def _process_section(
        self,
        section: str,
        source_file: str,
        metadata: Dict[str, Any],
        chunk_start_id: int,
        char_position: int
    ) -> List[DocumentChunk]:
        #Process a single section, potentially splitting if too large

        section_tokens = self.token_counter.count_tokens(section)

        if section_tokens <= self.chunk_size:
            chunk = self._create_chunk(
                section, source_file, metadata, chunk_start_id,
                char_position, char_position + len(section)
            )
            return [chunk]
        else:
            return self._split_large_section(
                section, source_file, metadata, chunk_start_id, char_position
            )

    def _split_large_section(
        self,
        section: str,
        source_file: str,
        metadata: Dict[str, Any],
        chunk_start_id: int,
        char_position: int
    ) -> List[DocumentChunk]:
        #Split a large section into multiple chunks with overlap

        sentences = self._split_into_sentences(section)
        chunks = []
        current_chunk_sentences = []
        current_tokens = 0
        local_char_pos = char_position

        for sentence in sentences:
            sentence_tokens = self.token_counter.count_tokens(sentence)

            if current_tokens + sentence_tokens > self.chunk_size and current_chunk_sentences:
                chunk_text = ' '.join(current_chunk_sentences)
                chunk = self._create_chunk(
                    chunk_text, source_file, metadata,
                    chunk_start_id + len(chunks),
                    local_char_pos - len(chunk_text),
                    local_char_pos
                )
                chunks.append(chunk)

                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk_sentences)
                current_chunk_sentences = overlap_sentences + [sentence]
                current_tokens = sum(
                    self.token_counter.count_tokens(s) for s in current_chunk_sentences
                )
            else:
                current_chunk_sentences.append(sentence)
                current_tokens += sentence_tokens

            local_char_pos += len(sentence) + 1

        # Handle remaining sentences
        if current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            chunk = self._create_chunk(
                chunk_text, source_file, metadata,
                chunk_start_id + len(chunks),
                local_char_pos - len(chunk_text),
                local_char_pos
            )
            chunks.append(chunk)

        return chunks

    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        #Get sentences for overlap based on token count
        if not sentences:
            return []

        overlap_sentences = []
        current_tokens = 0

        for sentence in reversed(sentences):
            sentence_tokens = self.token_counter.count_tokens(sentence)
            if current_tokens + sentence_tokens <= self.overlap:
                overlap_sentences.insert(0, sentence)
                current_tokens += sentence_tokens
            else:
                break

        return overlap_sentences

    def _chunk_by_tokens(
        self,
        text: str,
        source_file: str,
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        #Fallback token-based chunking
        chunks = []
        tokens = self.token_counter.encoder.encode(text)

        with tqdm(desc="Token-based chunking", total=len(tokens), unit="token") as pbar:
            i = 0
            chunk_id = 0
            char_position = 0

            while i < len(tokens):
                chunk_tokens = tokens[i:i + self.chunk_size]
                chunk_text = self.token_counter.encoder.decode(chunk_tokens)

                chunk = self._create_chunk(
                    chunk_text, source_file, metadata, chunk_id,
                    char_position, char_position + len(chunk_text)
                )
                chunks.append(chunk)

                i += self.chunk_size - self.overlap
                char_position += len(chunk_text)
                chunk_id += 1

                pbar.update(len(chunk_tokens))

        return chunks

    def _create_chunk(
        self,
        text: str,
        source_file: str,
        metadata: Dict[str, Any],
        chunk_id: int,
        start_char: int,
        end_char: int
    ) -> DocumentChunk:
        #Create a DocumentChunk with comprehensive metadata

        # Estimate page numbers
        chars_per_page = 2000
        start_page = max(1, start_char // chars_per_page + 1)
        end_page = max(start_page, end_char // chars_per_page + 1)

        return DocumentChunk(
            text=text.strip(),
            chunk_id=f"{Path(source_file).stem}_chunk_{chunk_id:04d}",
            source_file=source_file,
            page_numbers=list(range(start_page, end_page + 1)),
            token_count=self.token_counter.count_tokens(text),
            start_char=start_char,
            end_char=end_char,
            metadata={
                **metadata,
                'chunk_method': 'academic_sections',
                'chunk_size_target': self.chunk_size,
                'overlap_tokens': self.overlap,
                'has_equations_chunk': bool(re.search(r'[=><≤≥±∫∑∆∇]|\\[a-zA-Z]+', text)),
                'has_tables_chunk': 'Table:' in text or '|' in text
            }
        )


class JSONDocumentProcessor:
    #Document processor for MinerU JSON files

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.json_processor = MinerUJSONProcessor(config)
        self.chunker = TextChunker(config)
        self.performance_tracker = PerformanceTracker()
        self.logger = logging.getLogger(__name__)

    def process_json(self, json_path: str) -> List[DocumentChunk]:
        #Process a single JSON file into chunks
        try:
            self.performance_tracker.start_timer('total_processing')

            # Step 1: Extract text from JSON
            self.performance_tracker.start_timer('json_extraction')
            text, metadata = self.json_processor.process_json_file(json_path)
            extraction_time = self.performance_tracker.end_timer('json_extraction')

            # Step 2: Create smart chunks
            self.performance_tracker.start_timer('text_chunking')
            chunks = self.chunker.chunk_text(text, metadata, json_path)
            chunking_time = self.performance_tracker.end_timer('text_chunking')

            total_time = self.performance_tracker.end_timer('total_processing')

            self.logger.info(
                f"JSON processing completed:\n"
                f"   Source: {Path(json_path).name}\n"
                f"   Stats: {len(text):,} chars → {len(chunks)} chunks\n"
                f"   Academic: {metadata.get('formula_count', 0)} formulas, "
                f"{metadata.get('table_count', 0)} tables\n"
                f"   Time: {total_time:.2f}s total"
            )

            return chunks

        except Exception as e:
            self.logger.error(f"JSON processing failed for {json_path}: {e}")
            raise DocumentProcessingError(f"Failed to process JSON: {e}") from e

    def process_multiple_jsons(self, json_paths: List[str]) -> List[DocumentChunk]:
        #Process multiple JSON files
        all_chunks = []

        with tqdm(desc="Processing JSON files", total=len(json_paths), unit="file") as pbar:
            for json_path in json_paths:
                try:
                    chunks = self.process_json(json_path)
                    all_chunks.extend(chunks)
                    pbar.set_description(f"Processed {Path(json_path).name}")

                except DocumentProcessingError as e:
                    self.logger.error(f"Failed to process {json_path}: {e}")
                    pbar.set_description(f"Failed {Path(json_path).name}")

                pbar.update(1)

        self.logger.info(
            f"Batch processing completed: {len(all_chunks)} total chunks "
            f"from {len(json_paths)} JSON files"
        )

        return all_chunks