import os
import sys
from dotenv import load_dotenv

# --- Debugging: Print the current working directory ---
# This helps confirm where the script is looking for the .env file.
# The .env file should be in this directory.
print(f"Current Working Directory: {os.getcwd()}")

# --- Step 1: Load Environment Variables ---
# This line looks for a .env file in the current directory and loads
# its content into the environment, making it accessible via os.getenv().
load_dotenv()
print("Attempting to load variables from .env file...")


def get_paths_from_env():
    """
    Loads and verifies file paths from the environment.
    
    Returns:
        tuple: A tuple containing the two paths (shapenet_path, dino_path).
               The script will exit if any variable is not found.
    """
    # --- Step 2: Read the variables using os.getenv() ---
    # These variables are now available just as if you had exported them in your shell.
    shapenet_path = os.getenv("SHAPENET_DATASET_PATH")
    dino_path = os.getenv("DINO_PROJECT_PATH")

    # --- Step 3: Verify the paths to prevent TypeErrors ---
    # The following checks are CRITICAL. If os.getenv() cannot find a variable,
    # it returns None. Using a None value in a function like os.path.join()
    # will cause the exact "TypeError: ... not NoneType" you are seeing.
    # By checking for None here, we can provide a clear error message and exit safely.

    if not shapenet_path:
        print("❌ ERROR: 'SHAPENET_DATASET_PATH' not found in .env file or environment.")
    else:
        print(f"✅ SUCCESS: Loaded ShapeNet path: {shapenet_path}")
        if not os.path.isdir(shapenet_path):
            print(f"   - Warning: The path '{shapenet_path}' does not exist or is not a directory.")

    if not dino_path:
        print("❌ ERROR: 'DINO_PROJECT_PATH' not found in .env file or environment.")
    else:
        print(f"✅ SUCCESS: Loaded DINO project path: {dino_path}")
        if not os.path.isdir(dino_path):
            print(f"   - Warning: The path '{dino_path}' does not exist or is not a directory.")

    # Exit if any of the essential paths are missing. This prevents the program
    # from continuing with a `None` value and crashing later.
    if not all([shapenet_path, dino_path]):
        print("\nOne or more required paths are missing. Exiting to prevent errors.")
        sys.exit(1)

    return shapenet_path, dino_path


def main():
    """
    Main function to demonstrate loading and using the paths from .env.
    """
    shapenet_dataset_path, dino_project_path = get_paths_from_env()

    print("\n--- Your script can now use these paths safely ---")
    print(f"Path to ShapeNet Dataset: {shapenet_dataset_path}")
    print(f"Path to DINO Project: {dino_project_path}")
    
    # Example of how you might use these paths:
    potential_file = os.path.join(dino_project_path, 'models', 'final_model.pth')
    print(f"\nExample of a constructed file path: {potential_file}")


if __name__ == "__main__":
    main()

