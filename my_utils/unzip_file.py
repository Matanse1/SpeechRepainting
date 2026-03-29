import zipfile
from pathlib import Path

def unzip_file(zip_file_path, extract_to):
    # Ensure the output directory exists
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)
    
    # Unzip the file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f'Unzipped {zip_file_path} to {extract_to}')

if __name__ == "__main__":
    # Example usage
    zip_file = '/dsi/gannot-lab/gannot-lab1/datasets/FSD50K/files-archive'  # Path to your zip file
    output_dir = '/dsi/gannot-lab/gannot-lab1/datasets/FSD50K'  # Directory where you want to extract the files
    unzip_file(zip_file, output_dir)
