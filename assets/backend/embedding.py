import os
import json
import hashlib
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class EmbeddingCache:
    def __init__(self, cache_dir: str = "embedding_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_cache_key(self, text: str, model_name: str) -> str:
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{model_name}_{text_hash}"
        
    def get(self, text: str, model_name: str) -> Optional[np.ndarray]:
        cache_key = self._get_cache_key(text, model_name)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.npy")
        
        if os.path.exists(cache_path):
            return np.load(cache_path)
        return None
        
    def set(self, text: str, model_name: str, embedding: np.ndarray):
        cache_key = self._get_cache_key(text, model_name)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.npy")
        np.save(cache_path, embedding)

def get_embeddings(
    texts: List[str],
    model_name: str = 'all-MiniLM-L6-v2',
    instruction: str = "Represent this text for semantic search:",
    use_cache: bool = True
) -> np.ndarray:
    """
    Generate embeddings for a list of texts using instruction-tuned models with caching.
    
    Args:
        texts: List of texts to embed
        model_name: Name of the SentenceTransformer model to use
        instruction: Instruction prefix for better embedding quality
        use_cache: Whether to use embedding cache
        
    Returns:
        Numpy array of embeddings
    """
    cache = EmbeddingCache() if use_cache else None
    
    try:
        model = SentenceTransformer(model_name)
        embeddings = []
        
        for text in texts:
            if cache:
                cached_embedding = cache.get(text, model_name)
                if cached_embedding is not None:
                    embeddings.append(cached_embedding)
                    continue
            
            # Add instruction prefix for better quality
            instructed_text = f"{instruction} {text}"
            embedding = model.encode(instructed_text, convert_to_tensor=True)
            
            if cache:
                cache.set(text, model_name, embedding)
            embeddings.append(embedding)
            
        return np.vstack(embeddings)
        
    except Exception as e:
        print(f"Error using SentenceTransformer: {e}")
        print("Falling back to TF-IDF vectorizer...")
        
        # Fallback to TF-IDF if SentenceTransformers not available
        vectorizer = TfidfVectorizer(
            max_features=384,  # Match embedding dimension of MiniLM
            stop_words='english'
        )
        return vectorizer.fit_transform(texts).toarray()

def get_hierarchical_embeddings(
    texts: List[str],
    model_name: str = 'all-MiniLM-L6-v2'
) -> Dict[str, np.ndarray]:
    """
    Generate hierarchical embeddings at different granularities.
    
    Args:
        texts: List of texts to embed
        model_name: Name of the SentenceTransformer model to use
        
    Returns:
        Dictionary with embeddings at different levels
    """
    # Split texts into sentences
    from nltk.tokenize import sent_tokenize
    try:
        import nltk
        nltk.download('punkt', quiet=True)
    except:
        # Fallback to simple splitting if NLTK not available
        sent_tokenize = lambda x: x.split('.')
        
    # Generate embeddings at different granularities
    document_embeddings = get_embeddings(texts, model_name)
    
    sentence_lists = [sent_tokenize(text) for text in texts]
    all_sentences = [sent for sents in sentence_lists for sent in sents]
    sentence_embeddings = get_embeddings(all_sentences, model_name)
    
    # Create sentence embedding index mapping
    sentence_map = {}
    current_idx = 0
    for doc_idx, sentences in enumerate(sentence_lists):
        sentence_map[doc_idx] = sentence_embeddings[current_idx:current_idx + len(sentences)]
        current_idx += len(sentences)
    
    return {
        'document': document_embeddings,
        'sentence': sentence_map
    } 