import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Union
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

from utils.bert import calculate_semantic_similarity


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION SUITE
# ═══════════════════════════════════════════════════════════════════════════════

class SimilarityLevel(Enum):
    """Semantic similarity interpretation levels."""
    VERY_HIGH = (0.90, "🟡 MUY ALTA")
    HIGH = (0.75, "🟢 ALTA")
    MEDIUM = (0.50, "🟠 MEDIA")
    LOW = (0.30, "🔴 BAJA")
    VERY_LOW = (0.15, "⚫ MUY BAJA")
    MINIMAL = (0.00, "⚫⚫ BAJÍSIMA")
    
    def __init__(self, threshold: float, label: str):
        self.threshold = threshold
        self.label = label

class SimilarityVisualizer:
    """Elegant visualization utilities for similarity results."""
    
    @staticmethod
    def create_progress_bar(score: Optional[float], width: int = 25) -> str:
        """Create visual progress bar representation of similarity score."""
        if score is None:
            return "[ ERROR AL OBTENER SCORE ]"
        
        normalized_score = max(0.0, min(1.0, score))
        filled_blocks = int(round(normalized_score * width))
        empty_blocks = width - filled_blocks
        
        return "█" * filled_blocks + "░" * empty_blocks
    
    @staticmethod
    def get_similarity_level(score: Optional[float]) -> str:
        """Determine semantic similarity level with emoji indicators."""
        if score is None:
            return "N/A (Error)"
        
        for level in SimilarityLevel:
            if score >= level.threshold:
                return level.label
        
        return SimilarityLevel.MINIMAL.label
    
    @staticmethod
    def format_score(score: Optional[float]) -> str:
        """Format similarity score for display."""
        return f"{score:.3f}" if score is not None else "N/A"

class DemoRunner:
    """Comprehensive demonstration of SapBERT similarity capabilities."""
    
    def __init__(self):
        self.visualizer = SimilarityVisualizer()
    
    def run_complete_demo(self) -> None:
        """Execute full demonstration suite."""
        self._print_header()
        
        demo_cases = [
            self._demo_raw_outputs,
            self._demo_string_vs_string,
            self._demo_string_vs_list,
            self._demo_list_vs_string,
            self._demo_list_vs_list,
            self._demo_empty_inputs,
            self._demo_api_failure
        ]
        
        for demo_case in demo_cases:
            demo_case()
        
        self._print_footer()
    
    def _print_header(self) -> None:
        """Print demonstration header."""
        print("\n" + "═" * 70)
        print("🔬 SapBERT SEMANTIC SIMILARITY DEMONSTRATION 🔬")
        print("═" * 70)
    
    def _print_footer(self) -> None:
        """Print demonstration footer."""
        print("\n" + "═" * 70)
        print("✨ Demo Complete ✨")
        print("═" * 70)
    
    def _demo_raw_outputs(self) -> None:
        """Demonstrate raw output formats."""
        print("\n--- Raw Output Examples ---")
        
        examples = [
            ("myocardial infarction", "heart attack"),
            ("myocardial infarction", ["heart attack", "stroke"]),
            (["myocardial infarction", "stroke"], "heart attack"),
            (["myocardial infarction", "stroke"], ["heart attack", "stroke"])
        ]
        
        for input_a, input_b in examples:
            result = calculate_semantic_similarity(input_a, input_b)
            print(f"{input_a} vs {input_b}:")
            print(f"  {result}")
    
    def _demo_string_vs_string(self) -> None:
        """Demonstrate single term comparison."""
        print("\n--- String vs String Comparison ---")
        
        term_a, term_b = "myocardial infarction", "heart attack"
        results = calculate_semantic_similarity(term_a, term_b)
        
        print(f"Comparing '{term_a}' vs '{term_b}':")
        self._display_results(results)
    
    def _demo_string_vs_list(self) -> None:
        """Demonstrate one-to-many comparison."""
        print("\n--- String vs List Comparison ---")
        
        source = "covid-19"
        targets = ["sars-cov-2 infection", "influenza", "common cold", ""]
        
        start_time = time.perf_counter()
        results = calculate_semantic_similarity(source, targets)
        duration = time.perf_counter() - start_time
        
        print(f"Computation time: {duration:.2f} seconds")
        print(f"Comparing '{source}' with multiple targets:")
        self._display_results(results)
    
    def _demo_list_vs_string(self) -> None:
        """Demonstrate many-to-one comparison."""
        print("\n--- List vs String Comparison ---")
        
        sources = ["type 2 diabetes mellitus", "hypertension", ""]
        target = "high blood pressure"
        
        start_time = time.perf_counter()
        results = calculate_semantic_similarity(sources, target)
        duration = time.perf_counter() - start_time
        
        print(f"Computation time: {duration:.2f} seconds")
        print(f"Comparing multiple sources with '{target}':")
        self._display_results(results)
    
    def _demo_list_vs_list(self) -> None:
        """Demonstrate many-to-many comparison."""
        print("\n--- List vs List Comparison ---")
        
        list_a = ["acute respiratory distress syndrome", "pneumonia"]
        list_b = ["ARDS", "lung injury", "broken leg", "veryrareconditionxyz123"]
        
        start_time = time.perf_counter()
        results = calculate_semantic_similarity(list_a, list_b)
        duration = time.perf_counter() - start_time
        
        print(f"Computation time: {duration:.2f} seconds")
        print("Cross-comparison matrix:")
        self._display_results(results, hierarchical=True)
    
    def _demo_empty_inputs(self) -> None:
        """Demonstrate edge cases with empty inputs."""
        print("\n--- Empty Input Edge Cases ---")
        
        empty_list = []
        sample_list = ["influenza", "common cold"]
        
        test_cases = [
            ("Empty A vs Non-empty B", empty_list, sample_list),
            ("Non-empty A vs Empty B", sample_list, empty_list),
            ("Empty A vs Empty B", empty_list, empty_list)
        ]
        
        for description, input_a, input_b in test_cases:
            print(f"\n{description}:")
            result = calculate_semantic_similarity(input_a, input_b)
            print(f"  Result: {result}")
    
    def _demo_api_failure(self) -> None:
        """Simulate API failure scenario."""
        print("\n--- API Failure Simulation ---")
        
        # Test with potentially problematic inputs that might cause issues
        print("Testing with simulated API failure scenarios...")
        
        # Test with empty strings
        results = calculate_semantic_similarity("", "")
        print("Results with empty strings:")
        self._display_results(results)
        
        # Test with very long or unusual strings
        unusual_input = "x" * 1000  # Very long string
        results = calculate_semantic_similarity(unusual_input, "normal term")
        print(f"\nResults with unusual input length ({len(unusual_input)} chars):")
        self._display_results(results)
    
    def _display_results(self, results: Dict[str, Dict[str, Optional[float]]], hierarchical: bool = False) -> None:
        """Display results with visual formatting."""
        for term_a, comparisons in results.items():
            if hierarchical:
                print(f"  Comparisons for '{term_a}':")
                indent = "    "
            else:
                indent = "  "
            
            for term_b, similarity in comparisons.items():
                score_str = self.visualizer.format_score(similarity)
                progress_bar = self.visualizer.create_progress_bar(similarity)
                level_str = self.visualizer.get_similarity_level(similarity)
                
                print(f"{indent}'{term_a}' vs '{term_b}': {score_str} {progress_bar} {level_str}")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo_runner = DemoRunner()
    demo_runner.run_complete_demo()