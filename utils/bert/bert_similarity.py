"""
SapBERT Semantic Similarity Engine

A sophisticated semantic similarity calculator for medical terms using SapBERT embeddings.
Designed for clarity, performance, and maintainability with clean architecture principles.

Requirements:
    pip install python-dotenv requests numpy

Configuration:
    Create a .env file with:
        SAPBERT_API_URL=https://<your-endpoint-id>.huggingface.cloud
        HF_TOKEN=hf_<your_token_with_permissions>
"""

import os
import time
import random  # Agregar import para jitter
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Union
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ModelConfig:
    """Immutable configuration for SapBERT model parameters."""
    embedding_dimension: int = 768
    request_timeout: int = 120
    normalization_epsilon: float = 1e-12
    zero_norm_threshold: float = 1e-9
    initial_wait: float = 5.0  # Espera inicial en segundos
    backoff_factor: float = 1.0  # Factor multiplicador para cada reintento

@dataclass(frozen=True)
class ApiCredentials:
    """Secure API credentials container."""
    url: str
    token: str
    
    @classmethod
    def from_environment(cls) -> 'ApiCredentials':
        """Load credentials from environment variables."""
        load_dotenv()
        
        url = os.getenv("SAPBERT_API_URL")
        token = os.getenv("HF_TOKEN")
        
        if not url or not token:
            raise EnvironmentError(
                "SAPBERT_API_URL and HF_TOKEN must be defined in environment or .env file"
            )
        
        return cls(url=url, token=token)
    
    @property
    def headers(self) -> Dict[str, str]:
        """Generate HTTP headers for API requests."""
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CORE EMBEDDING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class EmbeddingClient:
    """High-level client for SapBERT embedding operations."""
    
    def __init__(self, credentials: ApiCredentials, config: ModelConfig = ModelConfig()):
        self._credentials = credentials
        self._config = config
        self._processor = EmbeddingProcessor(config)
        self._is_warmed_up = False
    
    def warm_up(self) -> bool:
        """
        Warm up the HuggingFace endpoint before processing.
        
        Returns:
            bool: True if endpoint is ready, False otherwise
        """
        if self._is_warmed_up:
            return True
            
        print("Warming up SapBERT endpoint...")
        test_text = ["test"]
        
        max_attempts = 5
        wait_time = 5.0
        
        for attempt in range(max_attempts):
            try:
                response = requests.post(
                    self._credentials.url,
                    headers=self._credentials.headers,
                    json={"inputs": test_text, "options": {"wait_for_model": True}},
                    timeout=30
                )
                
                if response.status_code == 200:
                    print("SapBERT endpoint ready.")
                    self._is_warmed_up = True
                    return True
                elif response.status_code == 503:
                    print(f"Endpoint starting up... waiting {wait_time}s (attempt {attempt + 1}/{max_attempts})")
                    time.sleep(wait_time)
                    wait_time += 1  # Increase wait time
                else:
                    print(f"Unexpected response: {response.status_code} - {response.text[:100]}")
                    return False
                    
            except Exception as e:
                print(f"Error during warm-up: {type(e).__name__}: {str(e)}")
                if attempt < max_attempts - 1:
                    time.sleep(wait_time)
                    
        print("Failed to warm up endpoint after maximum attempts")
        return False
    
    def get_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        Retrieve normalized CLS embeddings for input texts.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            Array of shape (len(texts), embedding_dim) or None if API fails
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._config.embedding_dimension)
        
        # Ensure endpoint is warmed up
        if not self._is_warmed_up:
            if not self.warm_up():
                return None
        
        raw_response = self._fetch_raw_embeddings(texts)
        if raw_response is None:
            return None
            
        return self._processor.extract_cls_embeddings(raw_response, texts)
    
    def _fetch_raw_embeddings(self, texts: List[str]) -> Optional[List[List[List[float]]]]:
        """Communicate with HuggingFace endpoint for raw token embeddings."""
        
        # Validate and clean inputs
        if not texts:
            return []
        
        # Clean texts
        clean_texts = []
        for i, text in enumerate(texts):
            if not text or not text.strip():
                clean_texts.append("Unknown condition")
            else:
                clean_text = text.strip()[:512]  # Limit length
                clean_texts.append(clean_text)
        
        payload = {
            "inputs": clean_texts,
            "options": {"wait_for_model": True}
        }
        
        try:
            response = requests.post(
                self._credentials.url,
                headers=self._credentials.headers,
                json=payload,
                timeout=self._config.request_timeout
            )
            
            if response.status_code == 200:
                return response.json()
            
            # Log errors concisely
            if response.status_code == 400:
                error_msg = response.json().get('error', response.text[:100])
                print(f"API Error 400: {error_msg}")
            elif response.status_code == 503:
                print(f"⚠️  Endpoint unavailable (503). Please run warm_up() first.")
            else:
                print(f"API Error {response.status_code}: {response.text[:100]}")
                
            return None
                
        except requests.exceptions.Timeout:
            print(f"⏱️  Request timeout after {self._config.request_timeout}s")
            return None
                    
        except requests.exceptions.RequestException as error:
            print(f"🌐 Network error: {type(error).__name__}")
            return None


class EmbeddingProcessor:
    """Processes raw API responses into normalized embedding vectors."""
    
    def __init__(self, config: ModelConfig):
        self._config = config
    
    def extract_cls_embeddings(
        self, 
        raw_embeddings: List[List[List[float]]], 
        original_texts: List[str]
    ) -> np.ndarray:
        """
        Convert raw token embeddings to normalized CLS vectors.
        
        Args:
            raw_embeddings: Raw API response with token-level embeddings
            original_texts: Original input texts for error context
            
        Returns:
            Normalized CLS embedding matrix
        """
        if len(raw_embeddings) != len(original_texts):
            self._log_dimension_mismatch(len(raw_embeddings), len(original_texts))
            return self._create_zero_matrix(len(original_texts))
        
        cls_vectors = []
        for raw_tokens, text in zip(raw_embeddings, original_texts):
            cls_vector = self._process_single_embedding(raw_tokens, text)
            cls_vectors.append(cls_vector)
        
        return np.array(cls_vectors, dtype=np.float32)
    
    def _process_single_embedding(self, token_embeddings: List[List[float]], text: str) -> np.ndarray:
        """Extract and normalize CLS token from single text embedding."""
        try:
            embedding_array = np.asarray(token_embeddings, dtype=np.float32)
            
            # Handle batch dimension if present
            if embedding_array.ndim == 3 and embedding_array.shape[0] == 1:
                embedding_array = embedding_array[0]
            
            if not self._is_valid_embedding_shape(embedding_array):
                self._log_invalid_shape(text, embedding_array.shape)
                return self._create_zero_vector()
            
            cls_token = embedding_array[0]  # First token is [CLS]
            return self._normalize_vector(cls_token)
            
        except Exception as error:
            print(f"ERROR: Processing embedding for '{text}': {error}")
            return self._create_zero_vector()
    
    def _is_valid_embedding_shape(self, array: np.ndarray) -> bool:
        """Validate embedding array dimensions."""
        return (
            array.ndim == 2 and 
            array.shape[0] >= 1 and 
            array.shape[1] == self._config.embedding_dimension
        )
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Apply L2 normalization to embedding vector."""
        norm = np.linalg.norm(vector)
        
        if norm < self._config.zero_norm_threshold:
            return np.zeros_like(vector)
        
        return vector / (norm + self._config.normalization_epsilon)
    
    def _create_zero_vector(self) -> np.ndarray:
        """Create zero vector with correct dimensions."""
        return np.zeros(self._config.embedding_dimension, dtype=np.float32)
    
    def _create_zero_matrix(self, num_texts: int) -> np.ndarray:
        """Create zero matrix for error cases."""
        return np.zeros((num_texts, self._config.embedding_dimension), dtype=np.float32)
    
    def _log_dimension_mismatch(self, embeddings_count: int, texts_count: int) -> None:
        """Log dimension mismatch errors."""
        print(f"ERROR: Embedding count ({embeddings_count}) != text count ({texts_count})")
    
    def _log_invalid_shape(self, text: str, shape: tuple) -> None:
        """Log invalid embedding shape errors."""
        expected_dim = self._config.embedding_dimension
        print(f"ERROR: Invalid embedding shape for '{text}': {shape}, expected: (*, {expected_dim})")


# ═══════════════════════════════════════════════════════════════════════════════
# SIMILARITY COMPUTATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class SimilarityCalculator:
    """Core similarity computation with optimized cross-comparison."""
    
    def __init__(self, embedding_client: EmbeddingClient):
        self._client = embedding_client
    
    def calculate_cross_similarity(
        self,
        input_a: Union[str, List[str]],
        input_b: Union[str, List[str]]
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """
        Calculate semantic similarity between all pairs of terms.
        
        Args:
            input_a: Single term or list of terms (source)
            input_b: Single term or list of terms (target)
            
        Returns:
            Nested dictionary with similarity scores for all pairs
        """
        # Normalize inputs to lists while preserving order
        terms_a = self._normalize_to_list(input_a)
        terms_b = self._normalize_to_list(input_b)
        
        # Initialize result structure
        results = self._initialize_result_matrix(terms_a, terms_b)
        
        if not terms_a or not terms_b:
            return results
        
        # Get embeddings for all unique terms efficiently
        embedding_map = self._build_embedding_map(terms_a, terms_b)
        
        if not embedding_map:
            return results  # All values remain None
        
        # Compute all pairwise similarities
        return self._compute_pairwise_similarities(terms_a, terms_b, embedding_map, results)
    
    def _normalize_to_list(self, input_data: Union[str, List[str]]) -> List[str]:
        """Convert input to list format while preserving order."""
        return [input_data] if isinstance(input_data, str) else list(input_data)
    
    def _initialize_result_matrix(
        self, 
        terms_a: List[str], 
        terms_b: List[str]
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """Pre-initialize result structure with None values."""
        return {
            term_a: {term_b: None for term_b in terms_b} 
            for term_a in terms_a
        }
    
    def _build_embedding_map(
        self, 
        terms_a: List[str], 
        terms_b: List[str]
    ) -> Dict[str, np.ndarray]:
        """Build efficient mapping from terms to their embeddings."""
        unique_terms = sorted(set(terms_a) | set(terms_b))
        
        if not unique_terms:
            return {}
        
        embeddings = self._client.get_embeddings(unique_terms)
        
        if embeddings is None or embeddings.shape[0] != len(unique_terms):
            return {}
        
        return {
            term: embeddings[i] 
            for i, term in enumerate(unique_terms)
        }
    
    def _compute_pairwise_similarities(
        self,
        terms_a: List[str],
        terms_b: List[str],
        embedding_map: Dict[str, np.ndarray],
        results: Dict[str, Dict[str, Optional[float]]]
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """Compute cosine similarities for all term pairs."""
        for term_a in terms_a:
            vector_a = embedding_map.get(term_a)
            if vector_a is None:
                continue
                
            for term_b in terms_b:
                vector_b = embedding_map.get(term_b)
                if vector_b is None:
                    continue
                
                similarity = self._calculate_cosine_similarity(vector_a, vector_b)
                results[term_a][term_b] = similarity
        
        return results
    
    def _calculate_cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute cosine similarity between L2-normalized vectors."""
        if np.all(vec_a == 0) or np.all(vec_b == 0):
            return 0.0
        
        return float(np.dot(vec_a, vec_b))

# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

# Global instances for convenience
_credentials = ApiCredentials.from_environment()
_embedding_client = EmbeddingClient(_credentials)
_similarity_calculator = SimilarityCalculator(_embedding_client)


def warm_up_endpoint() -> bool:
    """
    Warm up the SapBERT endpoint before processing.
    
    This should be called once before batch processing to ensure the endpoint is ready.
    
    Returns:
        bool: True if endpoint is ready, False otherwise
        
    Example:
        >>> from utils.bert import warm_up_endpoint, calculate_semantic_similarity
        >>> if warm_up_endpoint():
        ...     results = calculate_semantic_similarity("heart attack", "myocardial infarction")
    """
    return _embedding_client.warm_up()


def calculate_semantic_similarity(
    input_a: Union[str, List[str]],
    input_b: Union[str, List[str]]
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Calculate semantic similarity between medical terms with cross-comparison.
    
    Performs many-to-many comparison between elements of input_a and input_b
    using SapBERT embeddings and cosine similarity.
    
    Note: For best performance, call warm_up_endpoint() once before batch processing.
    
    Args:
        input_a: Medical term(s) - string or list of strings
        input_b: Medical term(s) - string or list of strings
        
    Returns:
        Nested dictionary where outer keys are terms from input_a,
        inner keys are terms from input_b, and values are similarity
        scores (0.0-1.0) or None if computation failed.
        
    Examples:
        >>> calculate_semantic_similarity("heart attack", "myocardial infarction")
        {'heart attack': {'myocardial infarction': 0.95}}
        
        >>> calculate_semantic_similarity("covid-19", ["sars-cov-2", "influenza"])
        {'covid-19': {'sars-cov-2': 0.88, 'influenza': 0.65}}
        
        >>> calculate_semantic_similarity(
        ...     ["heart attack", "stroke"],
        ...     ["myocardial infarction", "cerebrovascular accident"]
        ... )
        {
            'heart attack': {
                'myocardial infarction': 0.95,
                'cerebrovascular accident': 0.32
            },
            'stroke': {
                'myocardial infarction': 0.31,
                'cerebrovascular accident': 0.92
            }
        }
    """
    return _similarity_calculator.calculate_cross_similarity(input_a, input_b)
