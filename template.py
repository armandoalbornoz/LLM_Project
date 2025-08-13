import os 
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# Define the name of the main project package
project_name = "datascience"

# List of all necessary file paths for the project structure
list_of_files = [
    ".github/workflows/.gitkeep",  # Keeps the GitHub workflows directory in version control
    f"src/{project_name}/__init__.py", 
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",  
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",  
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py", 
    f"src/{project_name}/constants/__init__.py",  
    f"src/{project_name}/cloud/__init__.py",  
    "data/"
    "config/config.yaml",     
    "params.yaml", 
    ".env",           
    "schema.yaml",            
    "main.py",                
    "Dockerfile",             
    "setup.py",               
    "research/research.ipynb", 
    "templates/index.html",     
]

# Loop over all file paths to create directories and files
for filepath in list_of_files:
    filepath = Path(filepath)  # Convert string path to Path object
    filedir, filename = os.path.split(filepath)  # Split into directory and file name

    try:
        # If a directory path is specified, create it (if it doesn't already exist)
        if filedir != "":
            os.makedirs(filedir, exist_ok=True)
            logging.info(f"Creating directory {filedir} for file: {filename}")

        # Create the file if it doesn't exist or is empty
        if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
            with open(filepath, "w") as f:
                pass  # Create an empty file
            logging.info(f"Creating empty file: {filepath}")
        else:
            logging.info(f"{filename} already exists")  # File already present and not empty

    except Exception as e:
        logging.error(f" Error creating {filepath}: {e}")
