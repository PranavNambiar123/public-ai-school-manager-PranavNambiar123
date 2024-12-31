from typing import Dict, List, Optional, Set, Union
import os
import ast
import logging
import tempfile
from git import Repo
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from langchain_core.tools import tool
import radon.complexity as radon_cc
import radon.metrics as radon_metrics
from pylint.lint import Run
from pylint.reporters import JSONReporter
from tree_sitter import Language, Parser

@dataclass
class CodeMetrics:
    """Data class to store code metrics"""
    complexity: float
    maintainability_index: float
    loc: int
    comments: int
    blank_lines: int

@dataclass
class FunctionInfo:
    """Data class to store function information"""
    name: str
    docstring: Optional[str]
    parameters: List[str]
    return_type: Optional[str]
    complexity: float
    start_line: int
    end_line: int

@dataclass
class ClassInfo:
    """Data class to store class information"""
    name: str
    docstring: Optional[str]
    methods: List[FunctionInfo]
    base_classes: List[str]
    complexity: float
    start_line: int
    end_line: int

@dataclass
class FileAnalysis:
    """Data class to store file analysis results"""
    path: str
    metrics: CodeMetrics
    imports: List[str]
    classes: List[ClassInfo]
    functions: List[FunctionInfo]
    dependencies: Set[str]
    issues: List[str]

class CodeAnalyzerAgent:
    """Agent responsible for deep code analysis"""
    
    def __init__(self, repo_url: str):
        """
        Initialize the Code Analyzer Agent
        
        Args:
            repo_url: URL of the GitHub repository to analyze
        """
        self.repo_url = repo_url
        self.base_path = self._clone_repository()
        self.parser = self._setup_parser()
    
    def _clone_repository(self) -> str:
        """Clone the GitHub repository to a temporary directory"""
        temp_dir = tempfile.mkdtemp()
        try:
            Repo.clone_from(self.repo_url, temp_dir)
            return temp_dir
        except Exception as e:
            logging.error(f"Error cloning repository {self.repo_url}: {str(e)}")
            raise

    def _cleanup(self):
        """Clean up temporary directory"""
        import shutil
        if os.path.exists(self.base_path):
            shutil.rmtree(self.base_path)
        
    def _setup_parser(self) -> Parser:
        """Setup tree-sitter parser"""
        parser = Parser()
        # You'll need to build the language first using tree-sitter-cli
        # Language.build_library('build/languages.so', ['tree-sitter-python'])
        # PY_LANGUAGE = Language('build/languages.so', 'python')
        # parser.set_language(PY_LANGUAGE)
        return parser
    
    @tool
    def analyze_file(self, file_path: str) -> Optional[FileAnalysis]:
        """
        Perform deep analysis of a Python file
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            FileAnalysis object containing analysis results
        """
        abs_path = os.path.join(self.base_path, file_path)
        if not os.path.exists(abs_path) or not file_path.endswith('.py'):
            return None
            
        try:
            with open(abs_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse AST
            tree = ast.parse(content)
            
            # Collect metrics
            metrics = self._get_metrics(content)
            
            # Analyze code structure
            imports = self._get_imports(tree)
            classes = self._get_classes(tree)
            functions = self._get_functions(tree)
            dependencies = self._get_dependencies(imports)
            
            # Get code quality issues
            issues = self._get_code_issues(abs_path)
            
            return FileAnalysis(
                path=file_path,
                metrics=metrics,
                imports=imports,
                classes=classes,
                functions=functions,
                dependencies=dependencies,
                issues=issues
            )
        except Exception as e:
            logging.error(f"Error analyzing file {abs_path}: {str(e)}")
            return None
    
    def _get_metrics(self, content: str) -> CodeMetrics:
        """Calculate code metrics using radon"""
        cc = radon_cc.cc_visit(content)
        mi = radon_metrics.mi_visit(content, multi=True)
        raw = radon_metrics.raw_metrics(content)
        
        return CodeMetrics(
            complexity=sum(block.complexity for block in cc),
            maintainability_index=mi,
            loc=raw.loc,
            comments=raw.comments,
            blank_lines=raw.blank
        )
    
    def _get_imports(self, tree: ast.AST) -> List[str]:
        """Extract imports from AST"""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(n.name for n in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                imports.extend(f"{module}.{n.name}" for n in node.names)
        return imports
    
    def _get_classes(self, tree: ast.AST) -> List[ClassInfo]:
        """Extract class information from AST"""
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append(self._function_to_info(item))
                
                classes.append(ClassInfo(
                    name=node.name,
                    docstring=ast.get_docstring(node),
                    methods=methods,
                    base_classes=[base.id for base in node.bases if isinstance(base, ast.Name)],
                    complexity=radon_cc.cc_visit_ast(node),
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno
                ))
        return classes
    
    def _get_functions(self, tree: ast.AST) -> List[FunctionInfo]:
        """Extract function information from AST"""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not isinstance(node.parent, ast.ClassDef):
                functions.append(self._function_to_info(node))
        return functions
    
    def _function_to_info(self, node: ast.FunctionDef) -> FunctionInfo:
        """Convert AST FunctionDef to FunctionInfo"""
        return FunctionInfo(
            name=node.name,
            docstring=ast.get_docstring(node),
            parameters=[arg.arg for arg in node.args.args],
            return_type=self._get_return_type(node),
            complexity=radon_cc.cc_visit_ast(node),
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno
        )
    
    def _get_return_type(self, node: ast.FunctionDef) -> Optional[str]:
        """Extract return type annotation if present"""
        if node.returns:
            return ast.unparse(node.returns)
        return None
    
    def _get_dependencies(self, imports: List[str]) -> Set[str]:
        """Extract external package dependencies from imports"""
        return {imp.split('.')[0] for imp in imports}
    
    def _get_code_issues(self, file_path: str) -> List[str]:
        """Run pylint and get code issues"""
        issues = []
        try:
            reporter = JSONReporter()
            Run([file_path], reporter=reporter, do_exit=False)
            
            for message in reporter.messages:
                issues.append(f"{message['type']} ({message['symbol']}): {message['message']} at line {message['line']}")
                
        except Exception as e:
            logging.error(f"Error running pylint: {str(e)}")
            
        return issues
    
    @tool
    def get_complexity_report(self, file_path: str) -> Dict[str, Union[float, List[Dict]]]:
        """
        Generate a detailed complexity report for a file
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Dictionary containing complexity metrics and hotspots
        """
        abs_path = os.path.join(self.base_path, file_path)
        if not os.path.exists(abs_path) or not file_path.endswith('.py'):
            return {}
            
        try:
            with open(abs_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Get complexity blocks
            blocks = radon_cc.cc_visit(content)
            
            # Calculate maintainability index
            mi = radon_metrics.mi_visit(content, multi=True)
            
            # Identify complexity hotspots
            hotspots = [
                {
                    'name': block.name,
                    'type': block.type,
                    'complexity': block.complexity,
                    'line': block.lineno
                }
                for block in blocks
                if block.complexity > 10  # Threshold for complex code
            ]
            
            return {
                'maintainability_index': mi,
                'average_complexity': sum(b.complexity for b in blocks) / len(blocks) if blocks else 0,
                'hotspots': hotspots
            }
        except Exception as e:
            logging.error(f"Error generating complexity report for {abs_path}: {str(e)}")
            return {}
