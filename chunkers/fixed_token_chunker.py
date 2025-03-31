"""
Fixed token chunker implementation for text splitting.
"""

import re
from typing import List

from .base_chunker import BaseChunker

class FixedTokenChunker(BaseChunker):
    """Chunker that splits text into fixed-size token chunks."""
    
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 0, 
                 encoding_name: str = "cl100k_base"):
        """
        Initialize the FixedTokenChunker.
        
        Args:
            chunk_size: The target size of each chunk in tokens
            chunk_overlap: The number of tokens to overlap between chunks
            encoding_name: The name of the tiktoken encoding to use
        """
        super().__init__(chunk_size, chunk_overlap)
        
        # Initialize the tokenizer from tiktoken
        try:
            import tiktoken
            self.encoding = tiktoken.get_encoding(encoding_name)
        except ImportError:
            print("tiktoken not installed, falling back to simple tokenization")
            self.encoding = None
            
    def split_text(self, text: str) -> List[str]:
        """
        Split the text into chunks of approximately chunk_size tokens,
        with chunk_overlap tokens of overlap between chunks.
        
        Args:
            text: The text to split
            
        Returns:
            A list of text chunks
        """
        if self.encoding is None:
            # Fallback to simple word-based tokenization if tiktoken is not available
            return self._split_by_words(text)
        
        # Tokenize the text
        tokens = self.encoding.encode(text)
        
        # Split into chunks
        chunks = []
        start_idx = 0
        
        while start_idx < len(tokens):
            # Calculate end index for this chunk
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            
            # Get token IDs for this chunk
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode back to text
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # If we've reached the end, break
            if end_idx == len(tokens):
                break
                
            # Move to next position, accounting for overlap
            start_idx = end_idx - self.chunk_overlap
            
        return chunks
    
    def _split_by_words(self, text: str) -> List[str]:
        """
        Fallback method to split by words if tiktoken is not available.
        
        Args:
            text: The text to split
            
        Returns:
            A list of text chunks
        """
        # Split by words
        words = re.findall(r'\S+', text)
        
        # Create chunks
        chunks = []
        start_idx = 0
        
        while start_idx < len(words):
            end_idx = min(start_idx + self.chunk_size, len(words))
            chunk = ' '.join(words[start_idx:end_idx])
            chunks.append(chunk)
            
            if end_idx == len(words):
                break
                
            start_idx = end_idx - self.chunk_overlap
            
        return chunks