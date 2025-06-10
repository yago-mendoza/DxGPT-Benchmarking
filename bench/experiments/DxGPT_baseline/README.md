This visualization platform analyzes the performance of DxGPT, a diagnostic AI model, by evaluating two key dimensions:

- **Semantic similarity**: How well the model's differential diagnoses (DDX) match the ground truth diagnoses (GDX) based on BERT embeddings
- **Severity assessment**: How accurately the model predicts disease severity levels

# Semantics vs Severity

## Semantic score (SapBERT)

The key objective is for the correct diagnosis to be present, regardless of its position. In cases with comorbidity, retrieving at least one relevant diagnosis is sufficient to consider the outcome successful, prioritizing coverage over ranking.

The severity score measures how far the model's severity predictions deviate from the ground truth:

1. For each DDX, calculate the absolute distance from its GDX: `|S_DDX - S_GDX|`
2. Normalize by the maximum possible distance:
   - If S_GDX ≤ 5: max distance = 10 - S_GDX
   - If S_GDX > 5: max distance = S_GDX
3. Average these normalized distances across all 5 DDX
4. Result: severity score ∈ [0,1] where 0 = perfect match

## Severity score (LLM-as-judge)

In this analysis, we consider all generated diagnoses, as they are presented simultaneously within the application and their combined presence influences the patient's experience. Focusing solely on the top-ranked result would underestimate the potential negative impact.

It is important to note that the severity score represents a **distance**; thus, a score of 0 indicates the ideal outcome. This metric quantifies how much the model's severity predictions deviate from the ground truth.

1. For each DDX, calculate the absolute distance from its GDX: `|S_DDX - S_GDX|`
2. Normalize by the maximum possible distance:
   - If S_GDX ≤ 5: max distance = 10 - S_GDX
   - If S_GDX > 5: max distance = S_GDX
3. Average these normalized distances across all 5 DDX
4. Result: severity score ∈ [0,1] where 0 = perfect match (no distance)

**Note:** in cases with multiple GDX (ground truth diagnoses), the one with the highest semantic similarity to any of the DDXs is used for the severity calculation.