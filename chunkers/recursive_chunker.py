"""
Recursive character text splitter implementation.
This chunker recursively splits text on a list of separators.
"""

from typing import List

from .base_chunker import BaseChunker

class RecursiveCharacterTextSplitter(BaseChunker):
    """Chunker that recursively tries to split text on a list of separators."""
    
    def __init__(self, 
                 chunk_size: int = 400, 
                 chunk_overlap: int = 0,
                 separators: List[str] = ["\n\n", "\n", ".", "?", "!", " ", ""]):
        """
        Initialize the RecursiveCharacterTextSplitter.
        
        Args:
            chunk_size: The target size of each chunk in tokens
            chunk_overlap: The number of tokens to overlap between chunks
            separators: A list of separators to split on, tried in order
        """
        super().__init__(chunk_size, chunk_overlap)
        self.separators = separators
        
        # Estimate average characters per token (for rough approximate sizing)
        self.chars_per_token = 4
        
        # Convert chunk size from tokens to characters
        self.char_size = self.chunk_size * self.chars_per_token
        self.char_overlap = self.chunk_overlap * self.chars_per_token
        
    def split_text(self, text: str) -> List[str]:
        """
        Split the text recursively using the separators.
        
        Args:
            text: The text to split
            
        Returns:
            A list of text chunks
        """
        # If the text is short enough, return it as a single chunk
        if len(text) <= self.char_size:
            return [text]
        
        # Try each separator in order
        for separator in self.separators:
            if separator == "":  # Last resort: split on character boundaries
                return self._split_by_chars(text)
                
            if separator in text:
                # Split on this separator
                splits = text.split(separator)
                
                # Process each split recursively
                final_chunks = []
                current_chunk = []
                current_length = 0
                
                for split in splits:
                    # If adding this split to the current chunk would make it too long,
                    # add the current chunk to the final chunks and start a new one
                    split_with_sep = split + separator if separator != "" else split
                    if current_length + len(split_with_sep) > self.char_size and current_chunk:
                        # Join the current chunk and add it to the final chunks
                        final_chunks.append(separator.join(current_chunk))
                        
                        # Start a new chunk, potentially with overlap
                        if self.char_overlap > 0 and current_chunk:
                            # Calculate how many of the last splits to keep for overlap
                            overlap_splits = []
                            overlap_length = 0
                            
                            for s in reversed(current_chunk):
                                s_len = len(s) + (len(separator) if separator != "" else 0)
                                if overlap_length + s_len <= self.char_overlap:
                                    overlap_splits.insert(0, s)
                                    overlap_length += s_len
                                else:
                                    break
                                    
                            current_chunk = overlap_splits
                            current_length = overlap_length
                        else:
                            current_chunk = []
                            current_length = 0
                    
                    # Add the current split to the current chunk
                    current_chunk.append(split)
                    current_length += len(split_with_sep)
                
                # Add the final chunk if it's not empty
                if current_chunk:
                    final_chunks.append(separator.join(current_chunk))
                
                return final_chunks
                
        # If we get here, none of the separators were found
        return [text]
    
    def _split_by_chars(self, text: str) -> List[str]:
        """
        Split the text by characters if no other separator works.
        
        Args:
            text: The text to split
            
        Returns:
            A list of text chunks
        """
        chunks = []
        start_idx = 0
        
        while start_idx < len(text):
            end_idx = min(start_idx + self.char_size, len(text))
            chunks.append(text[start_idx:end_idx])
            
            if end_idx == len(text):
                break
                
            start_idx = end_idx - self.char_overlap
            
        return chunks