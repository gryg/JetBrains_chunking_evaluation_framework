"""
Embedding provider using HuggingFace transformer models.
"""

import numpy as np
from typing import List

from .base_provider import BaseEmbeddingProvider

class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider using any HuggingFace transformer model."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        """
        Initialize the HuggingFace embedding provider.
        
        Args:
            model_name: Name of the HuggingFace model to use
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            
            # Store torch for later use
            self.torch = torch
            
            print(f"Successfully loaded HuggingFace model: {model_name} on {self.device}")
        except ImportError:
            print("Transformers not installed. Install with: pip install transformers torch")
            print("Falling back to simple embedding method.")
        except Exception as e:
            print(f"Error loading HuggingFace model: {e}")
            print("Falling back to simple embedding method.")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using HuggingFace transformers."""
        if self.model is None or self.tokenizer is None:
            return self._simple_embed_texts(texts)
        
        # Process in batches to avoid memory issues
        batch_size = 8  # Smaller batch size as transformer models use more memory
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                # Tokenize the batch
                inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
                
                # Generate embeddings
                with self.torch.no_grad():
                    outputs = self.model(**inputs)
                    
                # Use the [CLS] token embedding or mean pooling
                if hasattr(outputs, "last_hidden_state"):
                    # Mean pooling
                    attention_mask = inputs["attention_mask"]
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    embeddings = self.torch.sum(token_embeddings * input_mask_expanded, 1) / self.torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                else:
                    # Fallback to CLS token
                    embeddings = outputs[0][:, 0, :]
                
                # Convert to numpy and normalize
                batch_embeddings = embeddings.cpu().numpy()
                batch_embeddings = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                
                all_embeddings.append(batch_embeddings)
            except Exception as e:
                print(f"Error generating embeddings for batch: {e}")
                # Fall back to simple embeddings for this batch
                simple_embeddings = self._simple_embed_texts(batch)
                all_embeddings.append(simple_embeddings)
        
        # Combine all batches
        if len(all_embeddings) == 1:
            return all_embeddings[0]
        else:
            return np.vstack(all_embeddings)