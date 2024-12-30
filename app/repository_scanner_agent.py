from typing import Dict, List, Optional, Set
import os
import pathlib
import magic
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from langchain_core.tools import tool

@dataclass
class RepoStructure:
    """Data class to store repository structure information"""
    path: str
    total_size: int
    file_count: int
    directory_count: int
    file_types: Dict[str, int]
    large_files: List[str]
    binary_files: List[str]
    important_files: List[str]

class RepositoryScannerAgent:
    """Agent responsible for scanning and analyzing repository structure"""
    
    def __init__(self, base_path: str, size_threshold: int = 10_000_000):
        """
        Initialize the Repository Scanner Agent
        
        Args:
            base_path: Root path of the repository
            size_threshold: File size threshold in bytes (default 10MB)
        """
        self.base_path = os.path.abspath(base_path)
        self.size_threshold = size_threshold
        self.mime = magic.Magic(mime=True)
        self.ignored_dirs = {'.git', 'node_modules', 'venv', '__pycache__', '.idea', '.vscode'}
        self.important_file_patterns = {'README', 'LICENSE', 'requirements.txt', 'setup.py', 
                                      'package.json', 'Dockerfile', '.gitignore'}
        
    @tool
    def scan_repository(self) -> RepoStructure:
        """
        Perform a complete scan of the repository
        
        Returns:
            RepoStructure object containing repository information
        """
        total_size = 0
        file_count = 0
        directory_count = 0
        file_types: Dict[str, int] = {}
        large_files: List[str] = []
        binary_files: List[str] = []
        important_files: List[str] = []
        
        for root, dirs, files in os.walk(self.base_path):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if d not in self.ignored_dirs]
            
            directory_count += len(dirs)
            
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.base_path)
                
                try:
                    # Get file stats
                    stats = os.stat(file_path)
                    total_size += stats.st_size
                    file_count += 1
                    
                    # Check file type
                    file_type = self._get_file_type(file_path)
                    file_types[file_type] = file_types.get(file_type, 0) + 1
                    
                    # Check for large files
                    if stats.st_size > self.size_threshold:
                        large_files.append(rel_path)
                    
                    # Check for binary files
                    if self._is_binary(file_path):
                        binary_files.append(rel_path)
                    
                    # Check for important files
                    if any(pattern in file.upper() for pattern in self.important_file_patterns):
                        important_files.append(rel_path)
                        
                except Exception as e:
                    logging.warning(f"Error processing file {file_path}: {str(e)}")
        
        return RepoStructure(
            path=self.base_path,
            total_size=total_size,
            file_count=file_count,
            directory_count=directory_count,
            file_types=file_types,
            large_files=large_files,
            binary_files=binary_files,
            important_files=important_files
        )
    
    @tool
    def get_file_content(self, file_path: str, max_size: int = 1_000_000) -> Optional[str]:
        """
        Safely read and return file content
        
        Args:
            file_path: Path to the file
            max_size: Maximum file size to read (default 1MB)
            
        Returns:
            File content as string or None if file is too large or binary
        """
        abs_path = os.path.join(self.base_path, file_path)
        if not os.path.exists(abs_path):
            return None
            
        if os.path.getsize(abs_path) > max_size:
            return None
            
        if self._is_binary(abs_path):
            return None
            
        try:
            with open(abs_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logging.warning(f"Error reading file {abs_path}: {str(e)}")
            return None
    
    def _get_file_type(self, file_path: str) -> str:
        """Determine file type using file extension and mime type"""
        try:
            mime_type = self.mime.from_file(file_path)
            return mime_type
        except:
            return pathlib.Path(file_path).suffix or 'unknown'
    
    def _is_binary(self, file_path: str) -> bool:
        """Check if a file is binary"""
        try:
            mime_type = self.mime.from_file(file_path)
            return not mime_type.startswith(('text/', 'application/json', 'application/xml'))
        except:
            return True  # Assume binary if we can't determine
    
    @tool
    def get_directory_structure(self, max_depth: int = 3) -> Dict:
        """
        Get a hierarchical view of the repository structure
        
        Args:
            max_depth: Maximum depth to traverse
            
        Returns:
            Dictionary representing the directory structure
        """
        def _scan_dir(path: str, current_depth: int) -> Dict:
            if current_depth > max_depth:
                return {"type": "directory", "truncated": True}
                
            result = {}
            try:
                for item in os.listdir(path):
                    item_path = os.path.join(path, item)
                    if os.path.isdir(item_path):
                        if item not in self.ignored_dirs:
                            result[item] = _scan_dir(item_path, current_depth + 1)
                    else:
                        result[item] = {"type": "file", "size": os.path.getsize(item_path)}
            except Exception as e:
                logging.warning(f"Error scanning directory {path}: {str(e)}")
                
            return result
            
        return _scan_dir(self.base_path, 1)
