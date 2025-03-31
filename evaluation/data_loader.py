"""
Data loader utilities for loading corpora and questions.
"""

import json
import pandas as pd
from typing import List, Dict, Any, Optional

class DataLoader:
    """Class for loading and processing evaluation data."""
    
    @staticmethod
    def load_corpus(corpus_path: str) -> str:
        """
        Load a corpus from a file.
        
        Args:
            corpus_path: Path to the corpus file
            
        Returns:
            The corpus text
        """
        with open(corpus_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def load_questions(questions_path: str, corpus_id: str = None) -> List[Dict[str, Any]]:
        """
        Load questions from a CSV file.
        
        Args:
            questions_path: Path to the questions CSV file
            corpus_id: Optional corpus ID to filter questions
            
        Returns:
            A list of question dictionaries
        """
        questions_df = pd.read_csv(questions_path)
        
        # Process references (parse JSON strings)
        questions_df['references'] = questions_df['references'].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
        
        # Filter by corpus ID if provided
        if corpus_id:
            questions_df = questions_df[questions_df['corpus_id'] == corpus_id]
            
        # Convert to list of dictionaries
        questions = questions_df.to_dict('records')
        
        return questions