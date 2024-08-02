import os
import glob

def combine_md_files(docs_dir, output_file):
    subfolders = ['core-concepts', 'getting-started', 'how-to', 'tools']
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for subfolder in subfolders:
            folder_path = os.path.join(docs_dir, subfolder)
            if not os.path.exists(folder_path):
                print(f"Warning: {folder_path} does not exist. Skipping...")
                continue
            md_files = glob.glob(os.path.join(folder_path, '*.md'))
            for file in sorted(md_files):
                with open(file, 'r', encoding='utf-8') as infile:
                    outfile.write(f"\n\n# {os.path.basename(file)}\n\n")
                    outfile.write(infile.read())
                    outfile.write("\n\n---\n")

def main():
    print("This script combines markdown files from the crewAI docs folder.")
    print("The docs folder should contain subfolders like 'core-concepts', 'getting-started', etc.")
    docs_dir = input("Please enter the full path to the crewAI docs folder: ").strip()
    
    # Validate the input path
    if not os.path.exists(docs_dir):
        print("Error: The specified path does not exist.")
        return
    
    # Check if it's likely the correct folder
    if not any(os.path.exists(os.path.join(docs_dir, subfolder)) for subfolder in ['core-concepts', 'getting-started', 'how-to', 'tools']):
        print("Warning: The specified folder doesn't seem to contain the expected subfolders.")
        confirm = input("Do you want to continue anyway? (y/n): ").strip().lower()
        if confirm != 'y':
            return
    
    # Set the output file path to the current working directory
    output_file = os.path.join(os.getcwd(), 'combined_crewai_docs.md')
    
    # Combine the markdown files
    combine_md_files(docs_dir, output_file)
    print(f"Combined documentation created at: {output_file}")

if __name__ == "__main__":
    main()