import os
import hashlib
import shutil

def compute_md5(filename):
    """
    Computes the MD5 hash of a given filename.
    """
    md5_hash = hashlib.md5(filename.encode())
    return md5_hash.hexdigest()

def main():
    # Get the current directory
    current_directory = os.getcwd()
    
    # List all files in the current directory
    for filename in os.listdir(current_directory):
        # Check if it's a file (not a directory)
        if os.path.isfile(filename):
            # Get the file's extension
            file_extension = os.path.splitext(filename)[1]
            
            # Compute the MD5 hash of the filename (without extension)
            new_filename = compute_md5(filename) + file_extension
            
            # Create a copy of the file with the new name
            shutil.copy(filename, new_filename)
            print(f"Copied {filename} to {new_filename}")

if __name__ == "__main__":
    main()