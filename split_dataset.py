import os
from datasets import load_dataset, DatasetDict
from huggingface_hub import HfApi, whoami

# --- Configuration ---
# The original dataset we want to load and split
ORIGINAL_REPO_ID = "Thermostatic/NeuralTranslate-mt-en-es-v2"

# The suffix to add to the new repository name
NEW_REPO_SUFFIX = "-Splits"

# Define the split ratios
TRAIN_SIZE = 0.8  # 80% for training
# The remaining 20% will be split equally between validation and test (10% each)
VALIDATION_TEST_SIZE = 0.5 # 50% of the remaining 20% for validation

# A random seed for reproducibility of the splits
RANDOM_SEED = 42

def get_hf_username():
    """Gets the Hugging Face username of the logged-in user."""
    try:
        user_info = whoami()
        return user_info['name']
    except (OSError, KeyError):
        print("Could not retrieve Hugging Face username.")
        print("Please ensure you are logged in using 'huggingface-cli login' or have set the HUGGING_FACE_HUB_TOKEN environment variable.")
        return None

def main():
    """Main function to load, split, and upload the dataset."""
    
    # 1. Get user's Hugging Face username to create the new repo under their namespace
    username = get_hf_username()
    if not username:
        return # Exit if user is not logged in

    # Construct the new repository name
    repo_name = os.path.basename(ORIGINAL_REPO_ID)
    new_repo_id = f"{username}/{repo_name}{NEW_REPO_SUFFIX}"
    
    print(f"--- Starting Dataset Processing ---")
    print(f"Original dataset: {ORIGINAL_REPO_ID}")
    print(f"New repository will be created at: {new_repo_id}")

    # 2. Load the original dataset from the Hugging Face Hub
    print("\n[Step 1/4] Loading original dataset...")
    try:
        original_dataset = load_dataset(ORIGINAL_REPO_ID)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
        
    print("Dataset loaded successfully.")
    print("Original dataset structure:")
    print(original_dataset)

    # The dataset has a single 'train' split, which we will split further.
    # If it had multiple splits, you would choose which one to process, e.g., original_dataset['train']
    if 'train' not in original_dataset:
        print(f"Error: The dataset does not contain a 'train' split to process.")
        return
        
    dataset_to_split = original_dataset['train']

    # 3. Create the 80% train split and a 20% temporary split (for validation and test)
    print("\n[Step 2/4] Splitting data into train (80%) and temp (20%)...")
    train_test_split = dataset_to_split.train_test_split(
        train_size=TRAIN_SIZE,
        seed=RANDOM_SEED
    )
    # This gives us:
    # train_test_split['train'] -> 80% of the data (our final training set)
    # train_test_split['test']  -> 20% of the data (to be split further)

    # 4. Split the 20% temporary set into a 50% validation and 50% test set
    print("[Step 3/4] Splitting temp data into validation (10%) and test (10%)...")
    temp_dataset = train_test_split['test']
    validation_test_split = temp_dataset.train_test_split(
        test_size=VALIDATION_TEST_SIZE, # 50% of this chunk for the test set
        seed=RANDOM_SEED
    )
    # This gives us:
    # validation_test_split['train'] -> 50% of the 20% (our final validation set)
    # validation_test_split['test']  -> 50% of the 20% (our final test set)

    # 5. Create the final DatasetDict with 'train', 'validation', and 'test' splits
    final_dataset = DatasetDict({
        'train': train_test_split['train'],
        'validation': validation_test_split['train'],
        'test': validation_test_split['test']
    })

    print("\nNew dataset structure with splits:")
    print(final_dataset)

    # 6. Upload the new dataset to the Hugging Face Hub
    print(f"\n[Step 4/4] Uploading the new dataset to '{new_repo_id}'...")
    try:
        # Create the repo first (optional, but good practice)
        api = HfApi()
        api.create_repo(
            repo_id=new_repo_id,
            repo_type="dataset",
            exist_ok=True # Don't raise an error if the repo already exists
        )
        print(f"Repository '{new_repo_id}' created or already exists.")
        
        # Push the dataset to the Hub
        final_dataset.push_to_hub(
            repo_id=new_repo_id,
            # private=True # Uncomment to make the new dataset private
        )
        print("\n--- Success! ---")
        print(f"Dataset successfully uploaded to: https://huggingface.co/datasets/{new_repo_id}")

    except Exception as e:
        print(f"\nAn error occurred during upload: {e}")
        print("Please ensure you are logged in and have write permissions.")

if __name__ == "__main__":
    main()