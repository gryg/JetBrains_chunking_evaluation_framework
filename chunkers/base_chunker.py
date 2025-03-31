"""
Base chunker class for text splitting.
All chunking strategies must inherit from this class.
"""

from typing import List

class BaseChunker:
    """Base class for text chunking strategies."""
    
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 0):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: The target size of each chunk in tokens
            chunk_overlap: The number of tokens to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """
        Split the text into chunks.
        
        Args:
            text: The text to split
            
        Returns:
            A list of text chunks
        """
        raise NotImplementedError("Subclasses must implement split_text method")