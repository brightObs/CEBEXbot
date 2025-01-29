import os

def create_project_structure(base_dir="crypto_signal_bot"):
    # Directory structure
    structure = {
        "bot": ["__init__.py", "bot.py", "signal.py", "scheduler.py"],
        "config": ["__init__.py", "settings.py"],
        "data": {
            "logs": ["app.log"],
            "__init__.py": None,
        },
        "tests": ["__init__.py", "test_bot.py", "test_signal.py"],
    }

    # Create base directory
    os.makedirs(base_dir, exist_ok=True)

    # Helper function to create directories and files
    def create_dirs_and_files(parent, content):
        if isinstance(content, list):  # If list, it's a directory with files
            for file in content:
                file_path = os.path.join(parent, file)
                open(file_path, "a").close()  # Create an empty file
        elif isinstance(content, dict):  # If dict, it's a nested directory
            for subdir, subcontent in content.items():
                subdir_path = os.path.join(parent, subdir)
                os.makedirs(subdir_path, exist_ok=True)
                if subcontent:
                    create_dirs_and_files(subdir_path, subcontent)

    # Iterate through structure
    for dir_name, files_or_dirs in structure.items():
        dir_path = os.path.join(base_dir, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        create_dirs_and_files(dir_path, files_or_dirs)

    # Create top-level files
    with open(os.path.join(base_dir, "requirements.txt"), "a") as f:
        f.write("python-telegram-bot\nrequests\nta\n")
    with open(os.path.join(base_dir, "README.md"), "a") as f:
        f.write("# Crypto Signal Bot\n\nThis project is a Telegram bot that provides cryptocurrency signals.")
    open(os.path.join(base_dir, "main.py"), "a").close()

    print(f"Project structure created at: {os.path.abspath(base_dir)}")

# Run the function
if __name__ == "__main__":
    create_project_structure()
