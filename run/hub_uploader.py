from huggingface_hub import HfApi, create_repo

def upload_checkpoint_folder(local_dir, repo_id, ignore_patterns=None):
    """
    Uploads a folder to a specific Hugging Face repository.
    """
    if ignore_patterns is None:
        ignore_patterns = []
        
    try:
        # Create repo if it doesn't exist
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print(f"Uploading contents of '{local_dir}' to '{repo_id}'...")
        
        api = HfApi()
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=ignore_patterns
        )
        print(f"Epoch checkpoint successfully uploaded to {repo_id}")
    except Exception as e:
        print(f"An error occurred during upload: {e}")