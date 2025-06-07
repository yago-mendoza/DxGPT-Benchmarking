# RAMEDIS Baseline Experiment

This experiment evaluates diagnostic LLMs on medical datasets using both semantic similarity and severity matching metrics.

## Structure

```
ramedis_baseline/
├── config.yaml              # Experiment configuration
├── run.py                   # Main execution script
├── eval-prompts/           # Prompt templates
│   ├── candidate_prompt.txt # Prompt for generating diagnoses
│   ├── judge_prompt.txt    # Prompt for severity assignment
│   └── output_schemas.py   # Output schemas for structured responses
├── config-parser/       # Utilities
│   └── llm_factory.py     # LLM instantiation from config
└── results/               # Generated results (timestamped)
```

## Data Workflow & Methodology

### 1. **Input Data Structure**
Each case in the dataset contains:
- `id`: Unique case identifier
- `case`: Clinical presentation text (symptoms, history, etc.)
- `complexity`: Case complexity rating (C5-C10)
- `diagnoses`: Golden diagnoses (GDX) with severity levels

### 2. **Candidate Generation Phase**
- **Input**: Clinical case description
- **Process**: Candidate LLM generates 5 differential diagnoses (DDX)
- **Output**: `candidate_responses.json` - Raw DDX for each case

### 3. **Semantic Evaluation Phase**
- **Method**: SapBERT embeddings + cosine similarity
- **Process**:
  1. For each DDX diagnosis, calculate similarity with ALL GDX diagnoses
  2. Take the MAXIMUM similarity score as the semantic match
  3. Early stopping: If perfect match (1.0) found, skip remaining comparisons
- **Output**: `semantic_evaluation.json` - Contains:
  - Individual DDX scores against each GDX
  - Maximum score for the case
  - Average semantic score across all cases

### 4. **Severity Assignment Phase**
- **Purpose**: Create a reusable severity lookup table
- **Process**:
  1. Collect ALL unique diagnoses (from both DDX and GDX)
  2. Batch diagnoses (100 per batch) to Judge LLM
  3. Assign severity levels S0-S10 based on:
     - Mortality risk without treatment
     - Long-term morbidity/sequelae
     - Treatment urgency
     - Quality of life impact
- **Output**: `severity_assignments.json` - Intermediate cache product
  - Prevents redundant API calls for repeated diagnoses
  - Ensures consistency across evaluations

### 4. **Severity Evaluation Phase**
- **Method**: Asymmetric penalty calculation
- **Process**:
  1. Look up severities for DDX and GDX from assignments
  2. Calculate penalties for each DDX-GDX pair:
     - **Underdiagnosis** (DDX < GDX): penalty = |diff| × 2.0
     - **Overdiagnosis** (DDX > GDX): penalty = diff × 1.5
  3. Take MINIMUM penalty (best match) and convert to 0-1 score
- **Output**: `severity_evaluation.json` - Final severity scores

## Configuration

Edit `config.yaml` to:
- Select datasets to evaluate
- Configure LLM models and parameters
- Adjust evaluation weights (underdiagnosis/overdiagnosis penalties)
- Control visualization outputs

## Running the Experiment

1. Ensure environment variables are set:
   ```bash
   export AZURE_OPENAI_ENDPOINT="your-endpoint"
   export AZURE_OPENAI_API_KEY="your-key"
   export SAPBERT_API_URL="your-sapbert-url"
   export HF_TOKEN="your-huggingface-token"
   ```

2. Run the experiment:
   ```bash
   cd src/experiments/ramedis_baseline
   python run.py
   ```

## Output Files Explained

Results are saved in `results/YYYY-MM-DD_HH-MM-SS/`:

1. **`config_copy.yaml`** - Snapshot of configuration used (for reproducibility)

2. **`candidate_responses.json`** - Raw DDX output from candidate LLM
   ```json
   {
     "metadata": {...},
     "responses": [
       {"case_id": "001", "ddx": ["diagnosis1", "diagnosis2", ...]}
     ]
   }
   ```

3. **`semantic_evaluation.json`** - Detailed semantic similarity analysis
   ```json
   {
     "evaluations": [{
       "case_id": "001",
       "score": 0.87,  // Max similarity found
       "gdx": ["actual diagnosis 1", "actual diagnosis 2"],
       "ddx_scores": {
         "diagnosis1": [0.87, 0.65],  // Similarity with each GDX
         "diagnosis2": [0.45, 0.52]
       }
     }]
   }
   ```

4. **`severity_assignments.json`** - Reusable severity lookup table (intermediate)
   ```json
   {
     "severities": {
       "Hypertension": "S3",
       "Myocardial infarction": "S8",
       // ... all unique diagnoses
     }
   }
   ```

5. **`severity_evaluation.json`** - Final severity matching scores
   ```json
   {
     "evaluations": [
       {"case_id": "001", "score": 0.75}  // Based on best severity match
     ]
   }
   ```

6. **`plots/`** - Visualizations
   - Scatter plot: Severity vs Semantic scores
   - Histograms: Score distributions

7. **`execution.log`** - Detailed runtime log with timestamps

## Scoring Methodology Details

### Semantic Score Calculation
```
For each case:
  For each DDX diagnosis:
    similarities = [similarity(ddx, gdx1), similarity(ddx, gdx2), ...]
    ddx_score = max(similarities)
  case_score = max(all ddx_scores)
```

### Severity Score Calculation
```
For each case:
  penalties = []
  For each DDX severity:
    For each GDX severity:
      diff = DDX_severity - GDX_severity
      if diff < 0:  # Underdiagnosis
        penalty = |diff| × 2.0
      else:         # Overdiagnosis  
        penalty = diff × 1.5
      penalties.append(penalty)
  
  best_penalty = min(penalties)
  score = 1 - (best_penalty / 10)
```

## Key Design Decisions

1. **Why cache severities?** 
   - Many diagnoses repeat across cases
   - Reduces API calls and costs
   - Ensures consistent severity assignment

2. **Why asymmetric penalties?**
   - Missing a serious condition (underdiagnosis) is more dangerous
   - Overdiagnosis has costs but is less critical

3. **Why maximum similarity?**
   - A diagnosis is correct if it matches ANY golden diagnosis
   - Rewards finding at least one correct diagnosis

4. **Why batch processing?**
   - Efficient API usage
   - Faster execution
   - Better error handling

## Customization

To create a new experiment:
1. Copy this folder to a new experiment name
2. Modify prompts in `eval-prompts/`
3. Adjust configuration in `config.yaml`
4. Extend `run.py` for custom evaluation logic