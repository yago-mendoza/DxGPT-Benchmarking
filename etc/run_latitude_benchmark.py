# ==============================================================================
# SCRIPT: Latitude Benchmark with Custom Synonym Judge
# ==============================================================================
#
# PURPOSE:
# 1. Sets up a project, dataset, and model in Latitude.
# 2. Uploads clinical cases from a CSV to the dataset.
# 3. Runs an evaluation using a specified LLM and prompt.
# 4. Waits for the Latitude evaluation to complete.
# 5. Fetches the results (LLM outputs and golden truths) from the evaluation.
# 6. Applies a custom Python "judge" to score each result based on synonymy.
# 7. (Optional, if async SDK features are accessible) Pushes these custom scores
#    back to Latitude using the 'annotate' feature.
#
# REQUIREMENTS:
# - Python 3.9+
# - pandas: `pip install pandas`
# - python-dotenv: `pip install python-dotenv`
# - latitude-sdk: `pip install latitude-sdk` (ensure it's the version compatible
#   with both the synchronous operations and potentially async 'annotate' or
#   be prepared to use two client instances if they are from different SDKs)
# - A .env file with your LATITUDE_API_KEY.
# - A CSV file (e.g., 'clinical_dataset.csv') with 'id', 'case', 'golden_diagnosis'.
# - (For annotation) An "Evaluator" pre-configured in your Latitude project UI,
#   and its UUID.
#
# ==============================================================================

# --- Standard Library Imports ---
import os
import json # For parsing LLM's JSON output
import time # For polling evaluation status
import asyncio # For the annotation part, if using async SDK features

# --- Third-party Library Imports ---
import pandas as pd
from dotenv import load_dotenv

# --- Latitude SDK Imports ---
# This is the primary SDK used for most operations, based on your initial script.
from latitude import Latitude
from latitude.models.project import CreateProjectRequestBody
from latitude.models.dataset import CreateDatasetRequestBody
from latitude.models.test_case import CreateTestCaseRequestBody, TestCase # For type hinting if needed
from latitude.models.model import CreateModelRequestBody
from latitude.models.evaluation import CreateEvaluationRequestBody, Evaluation # For type hinting
from latitude.models.test_run import TestRun # To inspect results

# For the annotation part, we might need a different client or way to call it
# This import is based on the *second* documentation snippet you provided.
# If 'latitude_sdk' is a separate package or namespace from 'latitude':
# from latitude_sdk import Latitude as LatitudeAsyncClient, AnnotateEvaluationOptions
# For now, let's assume the 'latitude' package might provide access, or we'll clarify.
# For simplicity in this first pass, we'll try to use the same 'client' if possible,
# but be mindful this part might need adjustment based on actual SDK capabilities.
# The documentation mentions `sdk.evaluations.annotate`, implying it's part of the
# same SDK object, but the `async` nature needs to be handled.

# ==============================================================================
# --- Configuration Section ---
# ==============================================================================
print("--- Script Configuration ---")

# Load API Key from .env file
load_dotenv()
LATITUDE_API_KEY = os.getenv("LATITUDE_API_KEY")
if not LATITUDE_API_KEY:
    raise ValueError(
        "LATITUDE_API_KEY not found in .env file or environment variables. "
        "Please create a .env file with LATITUDE_API_KEY='your_key_here'."
    )
print("LATITUDE_API_KEY loaded.")

# --- File and Naming Configuration ---
CSV_FILE_PATH = "clinical_dataset.csv" # Path to your input dataset
PROJECT_NAME = "Benchmark Diagnostico Medico V2"
DATASET_NAME = "Casos Clinicos Mayo 2024 - Con Juez"
MODEL_NAME = "GPT3.5 Turbo - Diagnostico 5 Opciones - Juez"
EVALUATION_NAME = "Evaluacion Mayo 2024 - Con Juez Personalizado"

# --- Prompt and Model Configuration ---
# The variable {{CASO_CLINICO}} will be replaced by the 'case' column from your CSV.
PROMPT_TEMPLATE = "Dado este caso clinico {{CASO_CLINICO}} genera una lista JSON de 5 posibles diagnosticos. El formato JSON debe ser una lista de strings, por ejemplo: [\"diagnostico 1\", \"diagnostico 2\", \"diagnostico 3\", \"diagnostico 4\", \"diagnostico 5\"]"
LLM_PROVIDER_MODEL_ID = "gpt-3.5-turbo" # e.g., "gpt-4", "claude-3-opus-20240229", etc.

# --- Custom Judge and Annotation Configuration ---
# This is the UUID of the *Evaluator* you create in the Latitude UI
# for your custom synonym score.
# Example: You go to Latitude UI -> Project -> Evaluators -> Create New
# Name it "Custom Synonym Score", Type "Numeric". Latitude gives it a UUID.
# Replace with your actual Evaluator/Metric UUID from Latitude UI.
CUSTOM_EVALUATOR_UUID_IN_LATITUDE = "your_custom_evaluator_uuid_from_latitude_ui_here"
# Set to None if you don't want to attempt annotation
# CUSTOM_EVALUATOR_UUID_IN_LATITUDE = None


# ==============================================================================
# --- Custom Synonym Judge Function ---
# ==============================================================================
def custom_synonym_judge(predicted_diagnoses_json_str: str, golden_diagnosis_str: str) -> dict:
    """
    Compares a list of predicted diagnoses (from LLM, as JSON string)
    with a golden diagnosis string.

    Args:
        predicted_diagnoses_json_str (str): LLM output, expected to be a JSON list of strings.
        golden_diagnosis_str (str): The ground truth diagnosis.

    Returns:
        dict: A dictionary containing:
            'score' (int): index + 1 of the best match, or 0 if no synonym found.
            'match_type' (str): 'exact_match', 'broad_match', 'no_match'.
            'matched_prediction' (str | None): The predicted diagnosis that matched.
            'reason' (str): Explanation of the scoring.
    """
    # --- Default result ---
    result = {
        'score': 0,
        'match_type': 'no_match',
        'matched_prediction': None,
        'reason': 'No valid predicted diagnoses or no match found.'
    }

    # --- 1. Parse LLM's predicted diagnoses ---
    try:
        # The prompt asks for a list of strings.
        # Example: ["Unspecified fracture...", "Otitis externa..."]
        predicted_list = json.loads(predicted_diagnoses_json_str)
        if not isinstance(predicted_list, list):
            result['reason'] = "Predicted output is not a valid JSON list."
            return result
    except json.JSONDecodeError:
        result['reason'] = f"Failed to parse predicted_diagnoses_json_str as JSON: {predicted_diagnoses_json_str}"
        return result
    except Exception as e:
        result['reason'] = f"Unexpected error parsing JSON: {e}"
        return result
        
    if not predicted_list:
        result['reason'] = "Predicted list is empty."
        return result

    # --- 2. Normalize golden diagnosis (simple normalization) ---
    norm_golden = golden_diagnosis_str.strip().lower()

    # --- 3. Iterate through predicted diagnoses and check for synonyms ---
    best_match_found = False
    for index, pred_diag_str in enumerate(predicted_list):
        if not isinstance(pred_diag_str, str):
            # Skip if an item in the list is not a string
            continue

        norm_pred = pred_diag_str.strip().lower()

        # --- 3a. Exact Match (after normalization) ---
        # This is a simple exact match.
        # For "Exact Synonym", you'd need a more sophisticated synonym lookup.
        if norm_pred == norm_golden:
            result['score'] = index + 1
            result['match_type'] = 'exact_match'
            result['matched_prediction'] = pred_diag_str
            result['reason'] = f"Exact match found at index {index} (score {index+1})."
            best_match_found = True
            break # Found best possible match (exact)

        # --- 3b. Broad Synonym Match (Placeholder - Needs real implementation) ---
        # This is where your sophisticated synonym logic would go.
        # For example, using a medical thesaurus, word embeddings, or another LLM call.
        #
        # Placeholder: Check if golden diagnosis is a substring of prediction OR
        # prediction is a substring of golden diagnosis (very basic "broad" check).
        # And ensure it's not an exact match we already checked.
        # THIS IS A VERY CRUDE APPROXIMATION OF "BROAD SYNONYM"
        if (norm_golden in norm_pred or norm_pred in norm_golden) and not best_match_found:
            # More specific checks can be added here
            # E.g., if norm_golden == "otitis externa" and norm_pred == "unspecified otitis externa, right ear"
            # this could be a broad match.
            # For now, we'll just take the first such "broad" match.
            result['score'] = index + 1
            result['match_type'] = 'broad_match' # Change if you have stricter broad definition
            result['matched_prediction'] = pred_diag_str
            result['reason'] = f"Potential broad match found at index {index} (score {index+1}). Review logic."
            best_match_found = True
            # Don't break here, an earlier exact match would be better.
            # If we want the *first* broad match, then break.
            # Current logic: an exact match later will override a broad match earlier.
            # To prioritize earlier items: if an exact match is found, it breaks.
            # If a broad match is found, it sets values and continues, but a later exact match will overwrite.
            # If we want the *first* match (exact or broad), we break after any match.
            # Let's stick to: find the *best type* of match at the *lowest index*.
            # The current loop does this if exact matches break.
            # For now, let's assume the first broad match is fine if no exact match is found later.
            break # Taking the first 'broad' match for simplicity.

    if not best_match_found and predicted_list:
         result['reason'] = f"No exact or broad synonym match found for '{golden_diagnosis_str}' in predictions: {predicted_list}."
    elif not predicted_list:
         result['reason'] = "Predicted list was empty or invalid after parsing."


    return result

# ==============================================================================
# --- Main Script Logic ---
# ==============================================================================
def main_sync_operations():
    """Handles the synchronous part of creating entities and running evaluation."""
    print("\n--- Initializing Latitude Client (Synchronous Operations) ---")
    # This client is for the main project/dataset/model/evaluation setup
    client = Latitude(api_key=LATITUDE_API_KEY)
    print("Latitude client initialized.")

    # --- 1. Create or Retrieve Project ---
    # A Project is a container for your datasets, models, and evaluations.
    print(f"\n--- Step 1: Ensuring Project '{PROJECT_NAME}' Exists ---")
    try:
        project = client.projects.create_or_retrieve(
            name=PROJECT_NAME,
            body=CreateProjectRequestBody(name=PROJECT_NAME) # Pass the name in the body as well
        )
        print(f"Project '{project.name}' (ID: {project.id}) obtained/created successfully.")
    except Exception as e:
        print(f"ERROR: Could not create/retrieve project '{PROJECT_NAME}'. Exception: {e}")
        return None, None # Return None for project and client to stop execution

    # --- 2. Load Data and Create/Retrieve Dataset with Test Cases ---
    # A Dataset holds your test cases. Each test case has an input and a ground truth.
    print(f"\n--- Step 2: Loading CSV and Managing Dataset '{DATASET_NAME}' ---")
    try:
        # --- 2a. Read clinical cases from CSV ---
        print(f"Reading dataset from: {CSV_FILE_PATH}")
        df = pd.read_csv(CSV_FILE_PATH)
        print(f"Successfully read {len(df)} rows from '{CSV_FILE_PATH}'.")
        # Ensure required columns are present
        required_cols = ['id', 'case', 'golden_diagnosis']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"ERROR: CSV file is missing required columns: {missing}")
            return project, client # Return project to potentially clean up later

        # --- 2b. Create or retrieve the dataset ---
        dataset = client.datasets.create_or_retrieve(
            project_id=project.id, # Associate with our project
            name=DATASET_NAME,
            body=CreateDatasetRequestBody(name=DATASET_NAME, project_id=project.id)
        )
        print(f"Dataset '{dataset.name}' (ID: {dataset.id}) obtained/created.")

        # --- 2c. Upload test cases to the dataset ---
        print(f"Processing and uploading/verifying {len(df)} test cases...")
        test_cases_in_latitude = []
        for index, row in df.iterrows():
            # `external_id` helps Latitude identify unique test cases, preventing duplicates
            # if you run the script multiple times.
            case_external_id = str(row['id'])
            clinical_case_text = str(row['case'])
            golden_diagnosis_text = str(row['golden_diagnosis'])

            # The 'input' to your prompt. The key ('CASO_CLINICO') MUST match
            # the placeholder in your PROMPT_TEMPLATE (e.g., {{CASO_CLINICO}}).
            input_data_for_prompt = {"CASO_CLINICO": clinical_case_text}

            try:
                # Check if test case with this external_id already exists
                existing_tcs_response = client.test_cases.list(dataset_id=dataset.id, external_id=case_external_id)
                
                if existing_tcs_response and existing_tcs_response.data and len(existing_tcs_response.data) > 0:
                    # Test case already exists
                    existing_tc = existing_tcs_response.data[0]
                    print(f"Test case with external_id '{case_external_id}' (ID: {existing_tc.id}) already exists. Verifying content...")
                    # Optionally, update if content differs (more complex, SDK might offer `update`)
                    # For now, we assume if it exists, it's correct.
                    test_cases_in_latitude.append(existing_tc)
                else:
                    # Test case does not exist, so create it
                    print(f"Creating test case with external_id '{case_external_id}'...")
                    test_case_body = CreateTestCaseRequestBody(
                        dataset_id=dataset.id,
                        input=input_data_for_prompt,
                        ground_truth=golden_diagnosis_text, # This is what LLM output is compared against by Latitude's default scorers
                        external_id=case_external_id
                    )
                    created_tc = client.test_cases.create(body=test_case_body)
                    print(f"Test case '{case_external_id}' created (ID: {created_tc.id}).")
                    test_cases_in_latitude.append(created_tc)

            except Exception as e_tc:
                print(f"ERROR creating/retrieving test case external_id '{case_external_id}': {e_tc}")
                # Decide if you want to 'continue' to the next test case or 'raise e_tc' to stop
                continue
        
        print(f"Finished processing {len(df)} test cases. {len(test_cases_in_latitude)} test cases now in dataset '{DATASET_NAME}'.")

    except FileNotFoundError:
        print(f"ERROR: CSV file not found at '{CSV_FILE_PATH}'.")
        return project, client
    except Exception as e_dataset:
        print(f"ERROR processing dataset or test cases: {e_dataset}")
        return project, client

    # --- 3. Create or Retrieve Model (Prompt Configuration) ---
    # A Model in Latitude defines the prompt, the LLM provider, and its parameters.
    print(f"\n--- Step 3: Ensuring Model '{MODEL_NAME}' Exists ---")
    try:
        model_body = CreateModelRequestBody(
            project_id=project.id,
            name=MODEL_NAME,
            provider_model_id=LLM_PROVIDER_MODEL_ID, # e.g., "gpt-3.5-turbo"
            prompt_template=PROMPT_TEMPLATE,
            # You can add model_parameters like temperature, max_tokens, etc.
            # model_parameters={"temperature": 0.5, "max_tokens": 300}
        )
        model = client.models.create_or_retrieve(
            project_id=project.id,
            name=MODEL_NAME, # Used for retrieval if it exists
            body=model_body   # Used for creation if it doesn't exist
        )
        print(f"Model '{model.name}' (ID: {model.id}) using '{LLM_PROVIDER_MODEL_ID}' obtained/created.")
    except Exception as e_model:
        print(f"ERROR creating/retrieving model: {e_model}")
        return project, client

    # --- 4. Create and Run Evaluation ---
    # An Evaluation runs a Model (or multiple models) against a Dataset.
    print(f"\n--- Step 4: Creating and Running Evaluation '{EVALUATION_NAME}' ---")
    try:
        evaluation_body = CreateEvaluationRequestBody(
            project_id=project.id,
            dataset_id=dataset.id,
            models_id=[model.id], # List of model IDs to evaluate
            name=EVALUATION_NAME
            # Latitude has built-in 'scorers'.
            # e.g., scorers=[{"type": "exact_match", "name": "Exact Ground Truth Match"}]
            # Our custom judge runs *after* this Latitude evaluation.
        )
        # Note: `create_or_retrieve` is not typically available for evaluations as they represent a 'run'.
        # We always create a new one.
        evaluation = client.evaluations.create(body=evaluation_body)
        print(f"Evaluation '{evaluation.name}' (ID: {evaluation.id}) created and execution initiated.")
        print(f"View progress: https://app.latitude.so/projects/{project.id}/evaluations/{evaluation.id}")

        # --- 4a. Wait for Evaluation to Complete ---
        print("Waiting for evaluation to complete (polling every 30 seconds)...")
        max_wait_time_seconds = 1800 # 30 minutes
        start_time = time.time()
        while True:
            current_time = time.time()
            if current_time - start_time > max_wait_time_seconds:
                print("ERROR: Max wait time exceeded for evaluation completion.")
                break # Exit loop, evaluation might still be running

            eval_status_response = client.evaluations.get(evaluation_id=evaluation.id)
            status = eval_status_response.status
            print(f"Current evaluation status: {status} (Elapsed: {int(current_time - start_time)}s)")

            if status in ["completed", "failed", "error"]:
                print(f"Evaluation finished with status: {status}")
                break
            
            time.sleep(30) # Wait before checking again
        
        if evaluation.status != "completed": # Re-fetch latest status
            final_eval_status = client.evaluations.get(evaluation_id=evaluation.id)
            if final_eval_status.status != "completed":
                 print(f"WARNING: Evaluation did not complete successfully (status: {final_eval_status.status}). Custom judging might be based on partial results or fail.")
                 # return project, client # Optionally stop if eval failed

        return project, client, evaluation # Return all key objects

    except Exception as e_eval:
        print(f"ERROR creating or running evaluation: {e_eval}")
        return project, client, None # Return None for evaluation

async def apply_judge_and_annotate(sync_client: Latitude, project_id: str, evaluation_id: str):
    """
    Fetches evaluation results, applies the custom judge, and (optionally) annotates.
    This function is async because 'annotate' is often an async operation.
    """
    if not evaluation_id:
        print("Skipping judge and annotation as evaluation_id is not available.")
        return

    print(f"\n--- Step 5: Fetching Results for Evaluation ID: {evaluation_id} ---")
    all_test_runs = []
    page = 1
    limit = 50 # Adjust as needed, check API limits
    while True:
        try:
            # TestRuns link TestCase inputs, Model outputs, and Evaluation context.
            test_runs_response = sync_client.test_runs.list(
                evaluation_id=evaluation_id,
                page=page,
                limit=limit
                # You might also filter by project_id if necessary, though eval_id should be specific enough
            )
            if test_runs_response and test_runs_response.data:
                print(f"Fetched page {page} with {len(test_runs_response.data)} test runs.")
                all_test_runs.extend(test_runs_response.data)
                if len(test_runs_response.data) < limit: # Last page
                    break
                page += 1
            else: # No more data or empty response
                print(f"No more test runs found after page {page-1 if page > 1 else 1}.")
                break
        except Exception as e_fetch:
            print(f"ERROR fetching test runs for evaluation '{evaluation_id}', page {page}: {e_fetch}")
            break # Stop fetching on error
    
    if not all_test_runs:
        print(f"No test runs found for evaluation ID {evaluation_id}. Cannot apply custom judge.")
        return

    print(f"Total test runs fetched: {len(all_test_runs)}")

    # --- Step 6: Apply Custom Judge to each Test Run ---
    print(f"\n--- Step 6: Applying Custom Judge to {len(all_test_runs)} Test Runs ---")
    judged_results = []
    for test_run in all_test_runs:
        # A TestRun object should have:
        # - test_run.id (unique ID for this specific run of a test case)
        # - test_run.output (the LLM's raw output string)
        # - test_run.test_case_id (to link back to the original TestCase for ground_truth)
        #   OR test_run.input and test_run.ground_truth directly
        
        # We need the ground_truth. Let's fetch the TestCase if not directly on TestRun
        try:
            # The `test_run.input` should be the dict we provided e.g. {"CASO_CLINICO": "text"}
            # The `test_run.output` is what the LLM generated.
            # The `test_run.ground_truth` should be available if the dataset was structured correctly.
            # Let's check the attributes of a `TestRun` object.
            # According to docs (if available) or by inspecting `test_run`:
            # It should have `test_run.output` and `test_run.ground_truth`.
            # And `test_run.id` is the unique ID for *this execution*.

            if not hasattr(test_run, 'output') or not hasattr(test_run, 'ground_truth'):
                print(f"WARNING: Test run {test_run.id} is missing 'output' or 'ground_truth'. Skipping.")
                # Fallback: try to get TestCase details
                # test_case_details = sync_client.test_cases.get(test_case_id=test_run.test_case_id)
                # llm_output = test_run.output
                # golden_truth = test_case_details.ground_truth
                continue # Skip if essential data is missing

            llm_output_str = test_run.output # This should be the JSON string from the LLM
            golden_truth_str = test_run.ground_truth
            
            print(f"\nJudging Test Run ID: {test_run.id}")
            print(f"  Golden Diagnosis: {golden_truth_str[:100]}...") # Print snippet
            print(f"  LLM Output: {llm_output_str[:100]}...") # Print snippet

            judge_result = custom_synonym_judge(llm_output_str, golden_truth_str)
            
            print(f"  Judge Result: Score={judge_result['score']}, Type='{judge_result['match_type']}', Matched='{judge_result['matched_prediction']}', Reason='{judge_result['reason']}'")
            
            judged_results.append({
                "test_run_id": test_run.id, # This is the 'conversation-uuid' equivalent for annotation
                "test_case_id": test_run.test_case_id, # Original test case ID
                "model_output": llm_output_str,
                "golden_diagnosis": golden_truth_str,
                "custom_score": judge_result['score'],
                "custom_match_type": judge_result['match_type'],
                "custom_reason": judge_result['reason']
            })

        except Exception as e_judge:
            print(f"ERROR applying judge to test_run {test_run.id if hasattr(test_run, 'id') else 'UNKNOWN'}: {e_judge}")
            continue
    
    # --- Step 7: (Optional) Annotate with Custom Scores in Latitude ---
    # This part uses the 'annotate' feature, which is often async and might
    # belong to a slightly different SDK interface or require async handling.
    if CUSTOM_EVALUATOR_UUID_IN_LATITUDE and judged_results:
        print(f"\n--- Step 7: Annotating {len(judged_results)} Test Runs with Custom Scores ---")
        print(f"Using Custom Evaluator Metric UUID (from Latitude UI): {CUSTOM_EVALUATOR_UUID_IN_LATITUDE}")

        # The `annotate` method is described in the newer async SDK docs.
        # We need to ensure our `sync_client` can call this, or instantiate an async client.
        # Let's assume for now that the `latitude-sdk` provides an `evaluations.annotate`
        # that can be called within an asyncio event loop.

        # from latitude.models.evaluation import AnnotateEvaluationOptions # Check if this exists
        # A simple dict might work if AnnotateEvaluationOptions is not found for the sync SDK
        # For the async SDK, it was `from latitude_sdk import AnnotateEvaluationOptions`

        # The method signature from the docs was:
        # latitude.evaluations.annotate(uuid: str, score: int, evaluation_uuid: str, options: AnnotateEvaluationOptions)
        # - uuid: The 'conversation-uuid', which is our `test_run.id`.
        # - score: Our `custom_score`.
        # - evaluation_uuid: The UUID of the *metric/evaluator* in Latitude UI.
        # - options: Contains `reason`.

        # If `sync_client.evaluations.annotate` is not async, we don't need the asyncio.gather
        # If it IS async, then we do. The SDK docs will be key here.
        # The new docs explicitly show `await latitude.evaluations.annotate(...)`

        # Because the structure of `latitude-sdk` (the one used for `client.projects` etc.)
        # does not seem to directly expose an async `annotate` method on `client.evaluations`,
        # this part is the most speculative and might require:
        # 1. A different client initialization (e.g., `LatitudeAsyncClient`).
        # 2. Confirmation that `latitude-sdk` has unified these.
        #
        # For now, let's *attempt* to call it, prepared for it to be async.
        # If your `latitude-sdk`'s `annotate` is synchronous, remove `async` and `await`.
        # If it requires a separate async client, you'd initialize that here.

        # Placeholder: Assume we need to create an async client if different
        # latitude_async_client = LatitudeAsyncClient(api_key=LATITUDE_API_KEY)

        annotations_sent = 0
        for item in judged_results:
            try:
                print(f"Attempting to annotate Test Run ID: {item['test_run_id']} with score: {item['custom_score']}")
                
                # Construct options for annotation.
                # The exact model for options depends on the SDK version.
                # For `latitude_sdk` (async one), it was `AnnotateEvaluationOptions(reason=item['custom_reason'])`.
                # If using the `latitude-sdk` (sync one), it might be a dict or a different model.
                # Let's try with a dictionary as a common fallback.
                annotation_options_payload = {"reason": item['custom_reason']}

                # This is the critical call.
                # IF `sync_client.evaluations.annotate` IS ASYNC (as per new docs):
                # await sync_client.evaluations.annotate( # Or latitude_async_client
                #     uuid=item['test_run_id'],
                #     score=int(item['custom_score']), # Ensure score is int
                #     evaluation_uuid=CUSTOM_EVALUATOR_UUID_IN_LATITUDE,
                #     options=annotation_options_payload # or AnnotateEvaluationOptions(reason=...)
                # )
                #
                # IF `sync_client.evaluations.annotate` IS SYNCHRONOUS:
                # sync_client.evaluations.annotate(
                #     uuid=item['test_run_id'],
                #     score=int(item['custom_score']),
                #     evaluation_uuid=CUSTOM_EVALUATOR_UUID_IN_LATITUDE,
                #     options=annotation_options_payload # or AnnotateEvaluationOptions(reason=...)
                # )
                #
                # For now, due to uncertainty on how the installed `latitude-sdk` handles this,
                # this part is commented out. You'll need to verify the exact method signature
                # and whether it's async from your SDK's documentation or by trying it.

                print(f"  INFO: Annotation for Test Run ID {item['test_run_id']} would be sent here.")
                print(f"    uuid='{item['test_run_id']}', score={int(item['custom_score'])}, evaluation_uuid='{CUSTOM_EVALUATOR_UUID_IN_LATITUDE}', reason='{item['custom_reason']}'")
                # To actually run it, uncomment one of the blocks above and ensure SDK compatibility.
                # Example (if it were async and part of the same client, which is unlikely for the older SDK structure):
                # await sync_client.evaluations.annotate(uuid=item['test_run_id'], score=int(item['custom_score']), evaluation_uuid=CUSTOM_EVALUATOR_UUID_IN_LATITUDE, options={"reason": item['custom_reason']})

                annotations_sent += 1
                # Add a small delay to avoid overwhelming the API if sending many
                if annotations_sent % 10 == 0: # Every 10 annotations
                    await asyncio.sleep(1) # if in async context
                    # time.sleep(1) # if in sync context

            except AttributeError as ae:
                print(f"  ERROR: The 'annotate' method might not be available or is structured differently on your client. Error: {ae}")
                print("  Skipping further annotations. Please check your Latitude SDK version and documentation for 'annotate'.")
                break # Stop trying to annotate
            except Exception as e_annotate:
                print(f"  ERROR annotating test_run_id {item['test_run_id']}: {e_annotate}")
                # Continue to next item or break, depending on error handling preference
                # break

        if annotations_sent > 0:
            print(f"Successfully sent (or would have sent) {annotations_sent} annotations to Latitude.")
        elif judged_results:
            print("Annotation step was configured but no annotations were actually sent (code is illustrative).")

    elif not CUSTOM_EVALUATOR_UUID_IN_LATITUDE:
        print("\n--- Step 7: Annotation Skipped ---")
        print("CUSTOM_EVALUATOR_UUID_IN_LATITUDE is not set. No scores will be pushed to Latitude.")
    elif not judged_results:
        print("\n--- Step 7: Annotation Skipped ---")
        print("No judged results available to annotate.")
        
    print("\n--- Custom Judging and Annotation Process Complete ---")
    # You can save `judged_results` to a local CSV/JSON if needed
    if judged_results:
        try:
            judged_df = pd.DataFrame(judged_results)
            judged_output_path = "judged_evaluation_results.csv"
            judged_df.to_csv(judged_output_path, index=False)
            print(f"Custom judged results saved to: {judged_output_path}")
        except Exception as e_save:
            print(f"Error saving judged results to CSV: {e_save}")


# --- Main Execution ---
if __name__ == "__main__":
    print("==============================================")
    print("=== Latitude Benchmark Script Initializing ===")
    print("==============================================")

    # --- Part 1: Synchronous operations (Setup and Run Evaluation) ---
    project, sync_latitude_client, evaluation_run = main_sync_operations()

    # --- Part 2: Asynchronous operations (Apply Judge and Annotate) ---
    # This part is wrapped in asyncio.run because 'annotate' is likely async.
    if project and sync_latitude_client and evaluation_run and evaluation_run.status == "completed":
        # Only proceed if evaluation completed and we have the necessary objects
        asyncio.run(apply_judge_and_annotate(
            sync_client=sync_latitude_client,
            project_id=project.id,
            evaluation_id=evaluation_run.id
        ))
    elif evaluation_run:
        print(f"Skipping custom judge and annotation because evaluation did not complete successfully (status: {evaluation_run.status}).")
    else:
        print("Skipping custom judge and annotation due to errors in earlier stages.")

    print("\n==============================================")
    print("=== Latitude Benchmark Script Finished     ===")
    print("==============================================")
    if project:
        print(f"Review your project in Latitude: https://app.latitude.so/projects/{project.id}")