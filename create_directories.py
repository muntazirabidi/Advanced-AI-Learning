import os

def create_directory_structure():
    # Base directory structure as a dictionary
    structure = {
        "1_Core_AI": {
            "Fundamentals": ["Linear_Algebra", "Probability_Stats", "Optimization"],
            "Deep_Learning_Basics": ["PyTorch_Basics", "TensorFlow_Basics", "Neural_Net_Architectures"]
        },
        "2_Generative_Methods": {
            "Diffusion_Models": ["Theory", "Implementations"],
            "GANs": {},
            "Normalizing_Flows": {},
            "Autoencoders": {},
            "Hybrid_Models": {}
        },
        "3_Probabilistic_ML": {
            "Gaussian_Processes": ["Theory", "Libraries"],
            "Bayesian_Neural_Nets": {},
            "Variational_Inference": {},
            "Bayesian_Optimization": {}
        },
        "4_Reinforcement_Learning": {
            "Theory": {},
            "Projects": {}
        },
        "5_Applications": {
            "Customer_Projects": {},
            "Scientific_AI": {},
            "Industry_Case_Studies": {}
        },
        "6_Mathematics": {
            "Calculus": {},
            "Probability": {},
            "Information_Theory": {}
        },
        "7_Tools": {
            "ML_Ops": {},
            "Open_Source": {},
            "Visualization": {}
        },
        "8_Papers": {
            "Generative_Models": {},
            "Bayesian_Methods": {},
            "Recent_Advances_2023+": {}
        },
        "9_Career_&_Projects": {
            "Interview_Prep": {
                "Secondmind": {},
                "General_ML_Questions": {}
            },
            "Project_Showcase": {}
        },
        "Resources": {
            "Courses": {},
            "Blogs": {},
            "Cheatsheets": {}
        }
    }

    # Base directory name
    base_dir = "Advanced-AI-Learning/"
    
    # Create base directory
    os.makedirs(base_dir, exist_ok=True)

    def create_directories(parent_path, structure):
        for dir_name, contents in structure.items():
            # Create current directory
            current_path = os.path.join(parent_path, dir_name)
            os.makedirs(current_path, exist_ok=True)
            print(f"Created directory: {current_path}")

            # If contents is a dictionary, recurse
            if isinstance(contents, dict):
                create_directories(current_path, contents)
            # If contents is a list, create those directories
            elif isinstance(contents, list):
                for subdir in contents:
                    subdir_path = os.path.join(current_path, subdir)
                    os.makedirs(subdir_path, exist_ok=True)
                    print(f"Created directory: {subdir_path}")

    # Create the directory structure
    create_directories(base_dir, structure)

    # Create additional files
    files_to_create = {
        "environment.yml": "name: ai-learning\ndependencies:\n  - python=3.9\n  - pytorch\n  - tensorflow\n  - jupyter",
        "LEARNING_JOURNAL.md": "# Learning Journal\n\n## Daily Progress Log\n\n### Date: [Today]\n- Topics covered:\n- Resources used:\n- Notes:",
        "README.md": "# Advanced AI Learning\n\n## Overview\nThis repository contains my learning journey in advanced AI concepts.\n\n## Structure\n[Directory structure description]\n\n## Learning Roadmap\n1. Core AI Fundamentals\n2. Generative Methods\n3. Probabilistic Machine Learning\n[Continue with other sections...]"
    }

    for filename, content in files_to_create.items():
        file_path = os.path.join(base_dir, filename)
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Created file: {file_path}")

if __name__ == "__main__":
    create_directory_structure()
    print("\nDirectory structure created successfully!")