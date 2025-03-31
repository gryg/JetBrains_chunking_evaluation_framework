"""
Sentence chunker implementation that splits text by sentences.
"""

import re
from typing import List

from .base_chunker import BaseChunker

# Check if spaCy is available
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    spacy = None
    SPACY_AVAILABLE = False

class SentenceChunker(BaseChunker):
    """A chunker that splits text by sentences and then groups into chunks."""
    
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 0):
        """
        Initialize the SentenceChunker.
        
        Args:
            chunk_size: The target size of each chunk in tokens
            chunk_overlap: The number of tokens to overlap between chunks
        """
        super().__init__(chunk_size, chunk_overlap)
        # Load spaCy model for sentence splitting
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])
                self.nlp.max_length = 10000000  # Set high max length to handle long texts
            except Exception as e:
                print(f"Could not load spaCy model: {e}, falling back to regex")
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text by sentences and then group into chunks.
        
        Args:
            text: The text to split
            
        Returns:
            A list of text chunks
        """
        # Split into sentences
        if self.nlp:
            # Use spaCy for better sentence splitting
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
        else:
            # Fallback to regex-based sentence splitting
            sentences = re.split(r'(?<=[.!?])\s+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        # Group sentences into chunks
        chunks = []
        current_chunk = []
        current_token_count = 0
        
        for sentence in sentences:
            # Approximate token count (rough estimate)
            sentence_token_count = len(sentence.split())
            
            # If adding this sentence would exceed the chunk size and we already have content,
            # finish the current chunk and start a new one
            if current_token_count + sentence_token_count > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                
                # Handle overlap: keep some sentences for the next chunk
                if self.chunk_overlap > 0:
                    # Find sentences to keep for overlap
                    overlap_sentences = []
                    overlap_token_count = 0
                    
                    for s in reversed(current_chunk):
                        s_token_count = len(s.split())
                        if overlap_token_count + s_token_count <= self.chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_token_count += s_token_count
                        else:
                            break
                    
                    current_chunk = overlap_sentences
                    current_token_count = overlap_token_count
                else:
                    current_chunk = []
                    current_token_count = 0
            
            # Add the sentence to the current chunk
            current_chunk.append(sentence)
            current_token_count += sentence_token_count
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks