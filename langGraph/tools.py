from langchain_core.tools import tool
import os
import traceback
import subprocess

@tool
def create_file(file_name: str, file_contents: str):
    """
    Create a new file with the provided contents at a given path in the workspace.
    
    Args:
        file_name (str): Name of the file to be created (required, non-empty)
        file_contents (str): The content to write to the file (required)
    """
    try:
        if not file_name or not file_contents:
            return {"error": "file_name and file_contents must be non-empty strings"}
        file_path = os.path.join(os.getcwd(), file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as file:  # 加 encoding
            file.write(file_contents)
        return {"message": f"Successfully created file at {file_path}"}
    except Exception as e:
        return {"error": str(e)}

@tool
def str_replace(file_name: str, old_str: str, new_str: str):
    """
    Replace specific text in a file.
    
    Args:
        file_name (str): Name of the target file (required, non-empty)
        old_str (str): Text to be replaced (must appear exactly once, required, non-empty)
        new_str (str): Replacement text (required, non-empty)
    """
    try:
        if not all([file_name, old_str, new_str]):
            return {"error": "All parameters must be non-empty strings"}
        file_path = os.path.join(os.getcwd(), file_name)
        with open(file_path, "r", encoding='utf-8') as file:
            content = file.read()
        new_content = content.replace(old_str, new_str, 1)
        with open(file_path, "w", encoding='utf-8') as file:
            file.write(new_content)
        return {"message": f"Successfully replaced '{old_str}' with '{new_str}' in {file_path}"}  # 修复：直接 f-string
    except Exception as e:
        return {"error": f"Error replacing '{old_str}' with '{new_str}' in {file_path}: {str(e)}"}

@tool
def send_message(message: str):
    """
    Send a message to the user.
    
    Args:
        message (str): The message to send to the user (required, non-empty)
    """
    if not message:
        return {"error": "Message must be non-empty"}
    return {"message": message}  # 统一返回 dict

@tool
def shell_exec(command: str) -> dict:
    """
    Execute a command in the specified shell session.

    Args:
        command (str): The shell command to execute (required, non-empty)

    Returns:
        dict: Contains 'stdout' and 'stderr'
    """
    try:
        if not command:
            return {"error": {"stderr": "Command must be non-empty"}}
        result = subprocess.run(
            command,
            shell=True,          
            cwd=os.getcwd(),        
            capture_output=True,
            text=True,    
            check=False,
            encoding='utf-8',
            errors='ignore'
        )
        return {"message": {"stdout": result.stdout, "stderr": result.stderr}}
    except Exception as e:
        return {"error": {"stderr": str(e)}}