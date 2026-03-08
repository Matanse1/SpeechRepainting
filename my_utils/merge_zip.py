import zipfile
import os

def merge_zip_files(parts, output_file):
    with open(output_file, 'wb') as outfile:
        for part in parts:
            with open(part, 'rb') as infile:
                outfile.write(infile.read())
    print(f"Merged files into {output_file}")

def unzip_file(zip_file_path, extract_to):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"Unzipped {zip_file_path} to {extract_to}")

def main():
    # Path to the directory containing the split zip files
    split_zip_dir = '/dsi/gannot-lab1/datasets/FSD50K'  # Change this to your path

    # List of split files (make sure to include them in the correct order)
    split_files = [
        'FSD50K.dev_audio.z01',
        'FSD50K.dev_audio.z02',
        'FSD50K.dev_audio.z03',
        'FSD50K.dev_audio.z04',
        'FSD50K.dev_audio.z05',
        'FSD50K.dev_audio.zip'
    ]

    # Full paths to split files
    split_files = [os.path.join(split_zip_dir, f) for f in split_files]

    # Output file name
    merged_zip = os.path.join(split_zip_dir, 'unsplit.zip')

    # Merge split files
    merge_zip_files(split_files, merged_zip)

    # Directory to unzip
    extract_dir = os.path.join(split_zip_dir, 'extracted_files')
    os.makedirs(extract_dir, exist_ok=True)

    # Unzip the merged file
    unzip_file(merged_zip, extract_dir)

if __name__ == "__main__":
    main()
