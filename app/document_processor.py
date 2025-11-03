import json
from pathlib import Path
from typing import Any

from docling.document_converter import DocumentConverter

from app.config import CLEAR_DOCS_DIR, RAW_DOCS_DIR  # Import global data directory


class RecursiveTextSplitter:
    def __init__(self, raw_docs_dir: str = None, clear_docs_dir: str = None):
        # Save processed chunks directly to the directory configured in DATA_DIR, so they are accessible by the retriever and indexer
        self.input_docs_path = Path(raw_docs_dir or RAW_DOCS_DIR)
        self.output_chunks_path = Path(clear_docs_dir or CLEAR_DOCS_DIR)
        self.converter = DocumentConverter()

        # Create directories if they don't exist
        self.input_docs_path.mkdir(exist_ok=True)
        self.output_chunks_path.mkdir(exist_ok=True)

    def _recursively_split_text(self, text: str, max_length_limit: int, overlap_size: int) -> list[str]:
        """
        Recursively splits text into smaller chunks if it exceeds the maximum length limit,
        applying an overlap for contextual continuity.
        """
        chunks = []

        # If text is already within the length limit, return as a single chunk
        if len(text) <= max_length_limit:
            chunks.append(text)
            return chunks

        # Find optimal split point - try to split by sentence or word boundary
        split_position = self._find_contextual_split_position(text, max_length_limit)

        if split_position == -1:
            # If no good contextual split found, split simply at the maximum length limit
            split_position = max_length_limit

        # First chunk
        first_chunk = text[:split_position].strip()
        if first_chunk:
            chunks.append(first_chunk)

        # Second chunk starts with overlap
        remaining_text = text[split_position - overlap_size:] if split_position > overlap_size else text[split_position:]
        if remaining_text:
            # Recursively process remaining text
            chunks.extend(self._recursively_split_text(remaining_text, max_length_limit, overlap_size))

        return chunks

    def _find_contextual_split_position(self, text: str, max_length_limit: int) -> int:
        """
        Finds the optimal position for text splitting to preserve context.
        Prefers sentence endings, then commas, then word spaces, working backward from the limit.
        """
        # Look for sentence endings within max_length_limit
        sentence_endings = ['.', '!', '?', '。', '！', '？']
        for i in range(min(max_length_limit, len(text)) - 1, max(0, max_length_limit - 100), -1):
            if text[i] in sentence_endings and (i + 1 >= len(text) or text[i + 1] in [' ', '\n', '"', "'"]):
                return i + 1

        # Look for commas
        for i in range(min(max_length_limit, len(text)) - 1, max(0, max_length_limit - 50), -1):
            if text[i] == ',' and text[i + 1] == ' ':
                return i + 1

        # Look for spaces between words
        for i in range(min(max_length_limit, len(text)) - 1, max(0, max_length_limit - 30), -1):
            if text[i] == ' ':
                return i + 1

        return -1  # No good split point found

    def _parse_text_elements(self, doc_items) -> list[dict[str, Any]]:
        """
        Extracts structured text data from Docling document items (e.g., paragraphs, headings).
        """
        paragraphs = []

        for item in doc_items:
            # Get text representation of the element
            text_content = item.text if hasattr(item, 'text') else str(item)

            if text_content and text_content.strip():
                # Determine element type for title
                element_type = type(item).__name__

                paragraphs.append({
                    'text': text_content.strip(),
                    'title': f"{element_type}_{len(paragraphs) + 1}",
                    'element_type': element_type,
                    'order': len(paragraphs)
                })

        return paragraphs

    def process_and_save_chunks(self, filename: str, max_length_limit: int) -> list[dict[str, Any]]:
        """
        Main document processing method: converts document, splits text, and saves chunk data.
        """
        file_path = self.input_docs_path / filename

        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} not found")

        print(f"Processing file: {filename}")

        # Convert document
        result = self.converter.convert(str(file_path))

        chunks = []
        chunk_sequence_id = 0
        overlap_size = int(max_length_limit * 0.1)  # 10% overlap

        # Extract text from document elements
        if hasattr(result.document, 'texts'):
            paragraphs = self._parse_text_elements(result.document.texts)
        else:
            # Alternative method: get all text as string
            full_text = str(result.document)
            paragraphs = [{
                'text': full_text,
                'title': 'Full_Document',
                'element_type': 'Document',
                'order': 0
            }]

        print(f"Found {len(paragraphs)} text elements for processing")

        for para_data in paragraphs:
            paragraph = para_data['text']
            section_title = para_data['title']
            element_type = para_data['element_type']

            if not paragraph.strip():
                continue

            # Process each paragraph
            if len(paragraph) <= max_length_limit:
                # Paragraph fits in one chunk
                chunk_data = {
                    "chunk_id": chunk_sequence_id,
                    "section_title": section_title,
                    "element_type": element_type,
                    "text": paragraph,
                    "chunk_size": len(paragraph),
                    "is_split": False,
                    "original_paragraph_size": len(paragraph)
                }
                chunks.append(chunk_data)
                chunk_sequence_id += 1
            else:
                # Paragraph needs to be split
                paragraph_chunks = self._recursively_split_text(
                    paragraph, max_length_limit, overlap_size
                )

                for i, chunk_text in enumerate(paragraph_chunks):
                    chunk_data = {
                        "chunk_id": chunk_sequence_id,
                        "section_title": section_title,
                        "element_type": element_type,
                        "text": chunk_text,
                        "chunk_size": len(chunk_text),
                        "is_split": True,
                        "split_part": f"{i+1}/{len(paragraph_chunks)}",
                        "original_paragraph_size": len(paragraph)
                    }
                    chunks.append(chunk_data)
                    chunk_sequence_id += 1

        # Save result
        output_filename = f"{Path(filename).stem}_chunks.json"
        output_path = self.output_chunks_path / output_filename

        output_data = {
            "source_file": filename,
            "max_chunk_size": max_length_limit,
            "overlap_size": overlap_size,
            "total_chunks": len(chunks),
            "chunks": chunks
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"Processing completed. Created {len(chunks)} chunks.")
        print(f"Result saved to: {output_path}")

        return chunks

    def calculate_chunk_metrics(self, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Returns statistical metrics for the created chunks.
        """
        if not chunks:
            return {}

        total_size = sum(chunk['chunk_size'] for chunk in chunks)
        split_chunks = [chunk for chunk in chunks if chunk.get('is_split', False)]

        return {
            "total_chunks": len(chunks),
            "split_chunks": len(split_chunks),
            "avg_chunk_size": total_size / len(chunks),
            "min_chunk_size": min(chunk['chunk_size'] for chunk in chunks),
            "max_chunk_size": max(chunk['chunk_size'] for chunk in chunks),
            "split_percentage": (len(split_chunks) / len(chunks)) * 100 if chunks else 0
        }
