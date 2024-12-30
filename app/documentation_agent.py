from typing import Dict, List, Optional, Set, Union
import os
import re
import logging
import networkx as nx
from dataclasses import dataclass, field
from datetime import datetime
from langchain_core.tools import tool
import mistune
from docstring_parser import parse as parse_docstring

@dataclass
class DocSection:
    """Data class to store documentation section information"""
    title: str
    content: str
    subsections: List['DocSection'] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)

@dataclass
class DocumentationNode:
    """Data class to store documentation node information"""
    path: str
    type: str  # 'file', 'class', 'function', 'module'
    name: str
    description: str
    related_nodes: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)

@dataclass
class DocumentationAnalysis:
    """Data class to store documentation analysis results"""
    coverage_score: float
    quality_score: float
    missing_docs: List[str]
    improvement_suggestions: List[str]
    knowledge_graph: nx.DiGraph

class DocumentationAgent:
    """Agent responsible for documentation analysis and generation"""
    
    def __init__(self, base_path: str):
        """
        Initialize the Documentation Agent
        
        Args:
            base_path: Root path of the repository
        """
        self.base_path = os.path.abspath(base_path)
        self.markdown_parser = mistune.create_markdown(renderer=mistune.AstRenderer())
        self.knowledge_graph = nx.DiGraph()
        
    @tool
    def analyze_documentation(self, file_path: str) -> Optional[DocumentationAnalysis]:
        """
        Analyze documentation quality and coverage
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            DocumentationAnalysis object containing analysis results
        """
        abs_path = os.path.join(self.base_path, file_path)
        if not os.path.exists(abs_path):
            return None
            
        try:
            with open(abs_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse file content
            docstrings = self._extract_docstrings(content)
            readme_content = self._find_related_readme(file_path)
            
            # Analyze documentation
            coverage = self._calculate_coverage(content, docstrings)
            quality = self._assess_quality(docstrings)
            missing = self._identify_missing_docs(content, docstrings)
            suggestions = self._generate_suggestions(coverage, quality, missing)
            
            # Update knowledge graph
            self._update_knowledge_graph(file_path, docstrings, readme_content)
            
            return DocumentationAnalysis(
                coverage_score=coverage,
                quality_score=quality,
                missing_docs=missing,
                improvement_suggestions=suggestions,
                knowledge_graph=self.knowledge_graph
            )
        except Exception as e:
            logging.error(f"Error analyzing documentation for {abs_path}: {str(e)}")
            return None
    
    @tool
    def generate_documentation(self, file_path: str, output_format: str = 'markdown') -> Optional[str]:
        """
        Generate documentation for a Python file
        
        Args:
            file_path: Path to the Python file
            output_format: Format of the output ('markdown' or 'rst')
            
        Returns:
            Generated documentation as string
        """
        abs_path = os.path.join(self.base_path, file_path)
        if not os.path.exists(abs_path):
            return None
            
        try:
            with open(abs_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract information
            docstrings = self._extract_docstrings(content)
            structure = self._analyze_file_structure(content)
            examples = self._extract_examples(content)
            
            # Generate documentation
            doc = self._format_documentation(
                file_path=file_path,
                structure=structure,
                docstrings=docstrings,
                examples=examples,
                output_format=output_format
            )
            
            return doc
        except Exception as e:
            logging.error(f"Error generating documentation for {abs_path}: {str(e)}")
            return None
    
    def _extract_docstrings(self, content: str) -> Dict[str, str]:
        """Extract and parse docstrings from code"""
        docstrings = {}
        module_pattern = r'"""(.*?)"""'
        
        # Extract module docstring
        module_match = re.search(module_pattern, content, re.DOTALL)
        if module_match:
            docstrings['module'] = module_match.group(1).strip()
        
        # Use AST to extract other docstrings (implementation details omitted)
        return docstrings
    
    def _find_related_readme(self, file_path: str) -> Optional[str]:
        """Find and parse related README files"""
        dir_path = os.path.dirname(os.path.join(self.base_path, file_path))
        readme_path = os.path.join(dir_path, 'README.md')
        
        if os.path.exists(readme_path):
            with open(readme_path, 'r', encoding='utf-8') as f:
                return f.read()
        return None
    
    def _calculate_coverage(self, content: str, docstrings: Dict[str, str]) -> float:
        """Calculate documentation coverage score"""
        # Implementation would analyze ratio of documented to total elements
        return len(docstrings) / (len(re.findall(r'def\s+\w+', content)) + 1)
    
    def _assess_quality(self, docstrings: Dict[str, str]) -> float:
        """Assess documentation quality"""
        if not docstrings:
            return 0.0
            
        total_score = 0
        for doc in docstrings.values():
            parsed = parse_docstring(doc)
            # Score based on presence of description, parameters, returns, etc.
            score = 0
            if parsed.short_description:
                score += 0.3
            if parsed.long_description:
                score += 0.2
            if parsed.params:
                score += 0.3
            if parsed.returns:
                score += 0.2
            total_score += score
            
        return total_score / len(docstrings)
    
    def _identify_missing_docs(self, content: str, docstrings: Dict[str, str]) -> List[str]:
        """Identify missing documentation"""
        missing = []
        # Implementation would identify functions/classes without docstrings
        return missing
    
    def _generate_suggestions(self, coverage: float, quality: float, missing: List[str]) -> List[str]:
        """Generate documentation improvement suggestions"""
        suggestions = []
        if coverage < 0.8:
            suggestions.append("Increase documentation coverage")
        if quality < 0.7:
            suggestions.append("Improve documentation quality")
        if missing:
            suggestions.append(f"Add documentation for: {', '.join(missing)}")
        return suggestions
    
    def _update_knowledge_graph(self, file_path: str, docstrings: Dict[str, str], readme: Optional[str]):
        """Update documentation knowledge graph"""
        # Add nodes and edges based on documentation relationships
        file_node = os.path.basename(file_path)
        self.knowledge_graph.add_node(file_node, type='file')
        
        for name, doc in docstrings.items():
            self.knowledge_graph.add_node(name, type='docstring')
            self.knowledge_graph.add_edge(file_node, name)
    
    def _analyze_file_structure(self, content: str) -> Dict:
        """Analyze file structure for documentation"""
        # Implementation would extract classes, functions, and their relationships
        return {}
    
    def _extract_examples(self, content: str) -> List[str]:
        """Extract code examples from docstrings"""
        examples = []
        # Implementation would extract example code blocks
        return examples
    
    def _format_documentation(self, file_path: str, structure: Dict,
                            docstrings: Dict[str, str], examples: List[str],
                            output_format: str) -> str:
        """Format documentation in specified output format"""
        if output_format == 'markdown':
            return self._format_markdown(file_path, structure, docstrings, examples)
        elif output_format == 'rst':
            return self._format_rst(file_path, structure, docstrings, examples)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _format_markdown(self, file_path: str, structure: Dict,
                        docstrings: Dict[str, str], examples: List[str]) -> str:
        """Format documentation in Markdown"""
        lines = [
            f"# {os.path.basename(file_path)}",
            "",
            docstrings.get('module', 'No module documentation available.'),
            "",
            "## Structure",
            "",
            "## Documentation",
            "",
            "## Examples",
            *examples,
            "",
            f"*Generated by DocumentationAgent on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        ]
        return '\n'.join(lines)
    
    def _format_rst(self, file_path: str, structure: Dict,
                   docstrings: Dict[str, str], examples: List[str]) -> str:
        """Format documentation in reStructuredText"""
        # Implementation for RST format
        return ""
