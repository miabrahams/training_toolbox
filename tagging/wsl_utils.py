import os
import subprocess
import re

def is_wsl():
    """Check if running under Windows Subsystem for Linux"""
    if os.path.exists('/proc/version'):
        with open('/proc/version', 'r') as f:
            return 'microsoft' in f.read().lower()
    return False

def wsl_path_to_windows(path):
    """Convert a WSL path to Windows format

    Example: /mnt/c/Users -> C:\\Users
    """
    if not is_wsl():
        return path

    try:
        return subprocess.check_output(['wslpath', '-w', path]).decode().strip()
    except subprocess.CalledProcessError:
        # Fallback method if wslpath fails
        if path.startswith('/mnt/'):
            drive = path[5:6].upper()
            rest = path[7:].replace('/', '\\')
            return f"{drive}:\\{rest}"
        return path

def windows_path_to_wsl(path):
    """Convert a Windows path to WSL format

    Example: C:\\Users -> /mnt/c/Users
    """
    if not is_wsl():
        return path

    try:
        return subprocess.check_output(['wslpath', path]).decode().strip()
    except subprocess.CalledProcessError:
        # Fallback method if wslpath fails
        if ':' in path:
            drive = path[0].lower()
            rest = path[2:].replace('\\', '/')
            return f"/mnt/{drive}/{rest}"
        return path

def convert_path_if_needed(path, target="wsl"):
    """Convert path to target format if needed

    Args:
        path: Path to convert
        target: Either "wsl" or "windows"

    Returns:
        Converted path
    """
    if not is_wsl():
        return path

    if target == "windows":
        if path.startswith('/mnt/'):
            return wsl_path_to_windows(path)
    elif target == "wsl":
        if re.match(r'^[a-zA-Z]:', path):
            return windows_path_to_wsl(path)

    return path
