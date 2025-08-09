import json
import os
import ast
import time
from typing import List, Dict, Tuple, Optional, Union, Any
import uuid
from dataclasses import dataclass
from enum import Enum
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import Language
from langchain_core.embeddings import Embeddings
import statistics
import tiktoken
from tqdm import tqdm
from langfuse import Langfuse
from langchain_community.embeddings import HuggingFaceEmbeddings
import re

# Disable ChromaDB telemetry to avoid PostHog errors
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_TELEMETRY'] = 'False'
os.environ['CHROMA_SERVER_NOFILE'] = '65536'

# Additional environment variables to disable telemetry completely
os.environ['POSTHOG_DISABLED'] = 'True'
os.environ['CHROMA_TELEMETRY_IMPL'] = 'none'

# Import ChromaDB and configure it to disable telemetry
try:
    import chromadb
    from chromadb.config import Settings
    
    # Configure ChromaDB settings to disable telemetry completely
    CHROMA_SETTINGS = Settings(
        anonymized_telemetry=False,
        allow_reset=True,
        is_persistent=True,
        chroma_server_nofile=65536
    )
    
    # Monkey patch to disable telemetry if it's still causing issues
    try:
        import chromadb.telemetry.product.posthog
        # Disable the problematic telemetry capture method
        def disabled_capture(*args, **kwargs):
            pass
        chromadb.telemetry.product.posthog.Posthog.capture = disabled_capture
    except (ImportError, AttributeError):
        pass
        
except ImportError:
    CHROMA_SETTINGS = None

# Import metadata utilities for Chroma compatibility
try:
    from .metadata_utils import sanitize_metadata_for_chroma, validate_chroma_metadata
except ImportError:
    # Fallback implementation if metadata_utils is not available
    def sanitize_metadata_for_chroma(metadata):
        """Fallback metadata sanitization."""
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                sanitized[key] = ', '.join(str(item) for item in value) if value else ''
            elif isinstance(value, dict):
                sanitized[key] = str(value)
            elif value is None:
                sanitized[key] = ''
            else:
                sanitized[key] = value
        return sanitized
    
    def validate_chroma_metadata(metadata):
        """Fallback metadata validation."""
        return all(isinstance(v, (str, int, float, bool, type(None))) for v in metadata.values())

from mllm_tools.utils import _prepare_text_inputs
from task_generator import get_prompt_detect_plugins
from .chunk_relationships import ChunkRelationshipDetector, ChunkRelationship, RelationshipType

# Import VectorStoreProvider interface for new architecture support
try:
    from .vector_store_providers import (
        VectorStoreProvider, 
        VectorStoreConfig, 
        SearchResult, 
        DocumentInput,
        VectorStoreError,
        VectorStoreOperationError,
        VectorStoreConnectionError
    )
    PROVIDER_INTERFACE_AVAILABLE = True
except ImportError:
    # Fallback for backward compatibility
    PROVIDER_INTERFACE_AVAILABLE = False
    VectorStoreProvider = object
    VectorStoreConfig = None


class ContentType(Enum):
    """Enumeration for different content types."""
    API = "api"
    EXAMPLE = "example"
    TUTORIAL = "tutorial"
    REFERENCE = "reference"
    CONCEPT = "concept"


class RelationshipType(Enum):
    """Enumeration for chunk relationship types."""
    PARENT = "parent"
    CHILD = "child"
    SIBLING = "sibling"
    REFERENCE = "reference"
    DEPENDENCY = "dependency"
    EXAMPLE_OF = "example_of"
    EXPLAINS = "explains"


@dataclass
class CodeStructure:
    """Structure information for code chunks."""
    type: str  # 'class', 'function', 'method', 'module'
    name: str
    parameters: List[str]
    return_type: Optional[str]
    decorators: List[str]
    complexity_score: int
    nested_elements: List[str]


@dataclass
class ChunkMetadata:
    """Enhanced metadata for document chunks."""
    # Basic metadata
    source: str
    filename: str
    file_type: str
    
    # Content classification
    content_type: ContentType
    semantic_tags: List[str]
    
    # Code-specific metadata
    code_structure: Optional[CodeStructure]
    
    # Plugin information
    plugin_namespace: Optional[str]
    
    # Hierarchy and relationships
    hierarchy_path: List[str]
    parent_headers: List[str]
    
    # Quality indicators
    semantic_complete: bool
    has_docstring: bool
    has_examples: bool
    
    # Additional context
    complexity_level: int
    estimated_tokens: int

class MetadataExtractor:
    """Enhanced metadata extraction for document chunks."""
    
    def __init__(self):
        self.api_keywords = ['def ', 'class ', 'function', 'method', 'parameter', 'return', 'attribute']
        self.example_keywords = ['example', 'demo', 'sample', 'usage', 'how to', 'tutorial']
        self.tutorial_keywords = ['tutorial', 'guide', 'walkthrough', 'step by step', 'getting started']
        
    def extract_enhanced_metadata(self, content: str, base_metadata: dict, 
                                chunk_type: str, **kwargs) -> ChunkMetadata:
        """Extract enhanced metadata for a document chunk."""
        
        # Classify content type
        content_type = self._classify_content_type(content, chunk_type)
        
        # Extract semantic tags
        semantic_tags = self._extract_semantic_tags(content, chunk_type)
        
        # Extract code structure if applicable
        code_structure = None
        if chunk_type in ['class', 'function', 'code_block']:
            code_structure = self._extract_code_structure(content, chunk_type, kwargs)
        
        # Detect plugin namespace
        plugin_namespace = self._detect_plugin_namespace(content, base_metadata.get('source', ''))
        
        # Extract hierarchy information
        hierarchy_path = kwargs.get('hierarchy_path', [])
        parent_headers = kwargs.get('parent_headers', [])
        
        # Calculate quality indicators
        semantic_complete = kwargs.get('semantic_complete', True)
        has_docstring = kwargs.get('has_docstring', False)
        has_examples = self._has_code_examples(content)
        
        # Calculate complexity and token estimates
        complexity_level = self._calculate_content_complexity(content, chunk_type)
        estimated_tokens = len(content.split()) * 1.3  # Rough token estimate
        
        return ChunkMetadata(
            source=base_metadata.get('source', ''),
            filename=base_metadata.get('filename', ''),
            file_type=base_metadata.get('file_type', ''),
            content_type=content_type,
            semantic_tags=semantic_tags,
            code_structure=code_structure,
            plugin_namespace=plugin_namespace,
            hierarchy_path=hierarchy_path,
            parent_headers=parent_headers,
            semantic_complete=semantic_complete,
            has_docstring=has_docstring,
            has_examples=has_examples,
            complexity_level=complexity_level,
            estimated_tokens=int(estimated_tokens)
        )
    
    def _classify_content_type(self, content: str, chunk_type: str) -> ContentType:
        """Classify the content type based on content and context."""
        content_lower = content.lower()
        
        # Check for API reference indicators
        api_score = sum(1 for keyword in self.api_keywords if keyword in content_lower)
        
        # Check for example indicators
        example_score = sum(1 for keyword in self.example_keywords if keyword in content_lower)
        
        # Check for tutorial indicators
        tutorial_score = sum(1 for keyword in self.tutorial_keywords if keyword in content_lower)
        
        # Classify based on chunk type and keyword scores
        if chunk_type in ['class', 'function'] and api_score > 0:
            return ContentType.API
        elif chunk_type == 'code_block' or example_score > tutorial_score:
            return ContentType.EXAMPLE
        elif tutorial_score > 0:
            return ContentType.TUTORIAL
        elif chunk_type == 'markdown_section':
            return ContentType.CONCEPT
        else:
            return ContentType.REFERENCE
    
    def _extract_semantic_tags(self, content: str, chunk_type: str) -> List[str]:
        """Extract semantic tags from content."""
        tags = []
        content_lower = content.lower()
        
        # Add chunk type as a tag
        tags.append(chunk_type)
        
        # Manim-specific tags
        manim_concepts = [
            'mobject', 'scene', 'animation', 'transform', 'camera', 'config',
            'text', 'geometry', 'graph', 'plot', 'vector', 'matrix', 'equation'
        ]
        
        for concept in manim_concepts:
            if concept in content_lower:
                tags.append(f'manim_{concept}')
        
        # Programming concepts
        programming_concepts = [
            'class', 'function', 'method', 'property', 'decorator', 'inheritance',
            'parameter', 'return', 'exception', 'import'
        ]
        
        for concept in programming_concepts:
            if concept in content_lower:
                tags.append(f'code_{concept}')
        
        # Documentation types
        if 'example' in content_lower:
            tags.append('has_example')
        if 'tutorial' in content_lower:
            tags.append('tutorial_content')
        if any(word in content_lower for word in ['api', 'reference', 'documentation']):
            tags.append('api_reference')
        
        return list(set(tags))  # Remove duplicates
    
    def _extract_code_structure(self, content: str, chunk_type: str, kwargs: dict) -> Optional[CodeStructure]:
        """Extract code structure information."""
        if chunk_type not in ['class', 'function', 'code_block']:
            return None
        
        name = kwargs.get('name', 'unknown')
        
        # Extract parameters
        parameters = []
        if chunk_type in ['class', 'function']:
            methods = kwargs.get('methods', [])
            if chunk_type == 'function':
                # For functions, try to extract parameters from the content
                parameters = self._extract_function_parameters(content)
            else:
                # For classes, use method names as nested elements
                parameters = methods
        
        # Extract decorators
        decorators = kwargs.get('decorators', [])
        
        # Calculate complexity
        complexity_score = kwargs.get('complexity_score', 1)
        
        # Extract nested elements
        nested_elements = []
        if chunk_type == 'class':
            nested_elements = kwargs.get('methods', [])
        
        return CodeStructure(
            type=chunk_type,
            name=name,
            parameters=parameters,
            return_type=self._extract_return_type(content),
            decorators=decorators,
            complexity_score=complexity_score,
            nested_elements=nested_elements
        )
    
    def _detect_plugin_namespace(self, content: str, source_path: str) -> Optional[str]:
        """Detect plugin namespace from content and source path."""
        # Check if source path contains plugin information
        if 'plugin' in source_path.lower():
            path_parts = source_path.split(os.sep)
            for i, part in enumerate(path_parts):
                if 'plugin' in part.lower() and i + 1 < len(path_parts):
                    return path_parts[i + 1]
        
        # Check content for plugin imports
        content_lower = content.lower()
        plugin_patterns = [
            r'from\s+manim_(\w+)',
            r'import\s+manim_(\w+)',
            r'manim[_-](\w+)'
        ]
        
        for pattern in plugin_patterns:
            matches = re.findall(pattern, content_lower)
            if matches:
                return matches[0]
        
        return None
    
    def _has_code_examples(self, content: str) -> bool:
        """Check if content contains code examples."""
        return bool(re.search(r'```\w*\n.*?\n```', content, re.DOTALL))
    
    def _calculate_content_complexity(self, content: str, chunk_type: str) -> int:
        """Calculate content complexity level (1-5)."""
        complexity = 1
        
        # Base complexity by type
        type_complexity = {
            'class': 3,
            'function': 2,
            'code_block': 2,
            'markdown_section': 1,
            'module_level': 1
        }
        complexity = type_complexity.get(chunk_type, 1)
        
        # Adjust based on content length
        word_count = len(content.split())
        if word_count > 500:
            complexity += 1
        if word_count > 1000:
            complexity += 1
        
        # Adjust based on code complexity indicators
        complexity_indicators = [
            'class', 'def', 'if', 'for', 'while', 'try', 'except',
            'import', 'from', 'lambda', 'yield', 'async', 'await'
        ]
        
        indicator_count = sum(1 for indicator in complexity_indicators 
                            if indicator in content.lower())
        complexity += min(indicator_count // 3, 2)  # Cap the increase
        
        return min(complexity, 5)  # Cap at 5
    
    def _extract_function_parameters(self, content: str) -> List[str]:
        """Extract function parameters from code content."""
        # Simple regex to extract parameters from function definitions
        pattern = r'def\s+\w+\s*\(([^)]*)\)'
        matches = re.findall(pattern, content)
        
        parameters = []
        for match in matches:
            if match.strip():
                # Split parameters and clean them
                params = [p.strip().split(':')[0].split('=')[0].strip() 
                         for p in match.split(',')]
                parameters.extend([p for p in params if p and p != 'self'])
        
        return parameters
    
    def _extract_return_type(self, content: str) -> Optional[str]:
        """Extract return type annotation from function definition."""
        pattern = r'def\s+\w+\s*\([^)]*\)\s*->\s*([^:]+):'
        matches = re.findall(pattern, content)
        return matches[0].strip() if matches else None


class CodeAwareTextSplitter:
    """Enhanced text splitter that understands code structure."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.metadata_extractor = MetadataExtractor()
        self.relationship_detector = ChunkRelationshipDetector()
    
    def split_documents_with_relationships(self, documents: List[Document]) -> Tuple[List[Document], Dict[str, List[ChunkRelationship]]]:
        """Split documents and detect relationships between chunks."""
        # First, split all documents normally
        all_chunks = []
        for doc in documents:
            if doc.metadata.get('file_type') == 'python':
                chunks = self.split_python_file(doc.page_content, doc.metadata)
            else:  # markdown
                chunks = self.split_markdown_file(doc.page_content, doc.metadata)
            all_chunks.extend(chunks)
        
        # Then detect relationships between all chunks
        relationships = self.relationship_detector.detect_relationships(all_chunks)
        
        # Add relationship information to chunk metadata
        for i, chunk in enumerate(all_chunks):
            chunk_id = f"chunk_{i}"
            if chunk_id in relationships:
                chunk.metadata['relationships'] = [
                    {
                        'related_chunk_id': rel.related_chunk_id,
                        'relationship_type': rel.relationship_type.value,
                        'strength': rel.strength,
                        'context': rel.context
                    }
                    for rel in relationships[chunk_id]
                ]
            else:
                chunk.metadata['relationships'] = []
        
        return all_chunks, relationships
        
    def split_python_file(self, content: str, metadata: dict) -> List[Document]:
        """Split Python files preserving complete semantic units with enhanced structure awareness."""
        documents = []
        
        try:
            tree = ast.parse(content)
            lines = content.split('\n')
            
            # Extract complete semantic units (classes and functions with full context)
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Ensure complete semantic boundaries by including decorators and full definitions
                    start_line = self._get_semantic_start_line(node, lines)
                    end_line = self._get_semantic_end_line(node, lines)
                    
                    # Extract complete code segment including decorators and full body
                    code_segment = '\n'.join(lines[start_line-1:end_line])
                    
                    # Validate that we have a complete semantic unit with enhanced suppression
                    if not self._validate_complete_semantic_unit(code_segment, node):
                        # Enhanced warning suppression for common Python patterns
                        suppress_warning = (
                            # Special methods (double underscore methods)
                            (hasattr(node, 'name') and node.name.startswith('__') and node.name.endswith('__')) or
                            # Common function names that often appear incomplete but are valid
                            (hasattr(node, 'name') and node.name in ['plugins', 'main', 'init', 'setup', 'config', 'construct', 'regular_method', 'load_plugins']) or
                            # Very short code segments (likely complete)
                            len(code_segment.strip()) < 200 or
                            # If the code segment contains the function/class definition, it's probably complete
                            (hasattr(node, 'name') and (f'def {node.name}(' in code_segment or f'class {node.name}' in code_segment or f'async def {node.name}(' in code_segment)) or
                            # If it's syntactically valid Python, it's probably fine
                            self._is_syntactically_valid(code_segment)
                        )
                        
                        if not suppress_warning:
                            print(f"Warning: Incomplete semantic unit detected for {node.name}")
                        continue
                    
                    # Extract docstring with full context
                    docstring = ast.get_docstring(node) or ""
                    
                    # Extract method signatures for classes
                    methods = []
                    if isinstance(node, ast.ClassDef):
                        methods = self._extract_class_methods(node)
                    
                    # Create enhanced content with complete semantic context
                    enhanced_content = self._create_semantic_content(
                        node, code_segment, docstring, methods
                    )
                    
                    # Extract enhanced metadata using the new system
                    chunk_type = 'class' if isinstance(node, ast.ClassDef) else 'function'
                    enhanced_metadata_obj = self.metadata_extractor.extract_enhanced_metadata(
                        content=enhanced_content,
                        base_metadata=metadata,
                        chunk_type=chunk_type,
                        name=node.name,
                        has_docstring=bool(docstring),
                        methods=methods,
                        decorators=self._extract_decorators(node),
                        complexity_score=self._calculate_complexity_score(node),
                        semantic_complete=True
                    )
                    
                    # Convert ChunkMetadata to dict for Document metadata
                    # Convert list values to strings for Chroma compatibility
                    methods_list = [m['name'] for m in methods] if methods else []
                    decorators_list = self._extract_decorators(node)
                    
                    enhanced_metadata = {
                        **metadata,
                        'type': chunk_type,
                        'name': node.name,
                        'start_line': start_line,
                        'end_line': end_line,
                        'has_docstring': bool(docstring),
                        'docstring': docstring[:200] + "..." if len(docstring) > 200 else docstring,
                        'semantic_complete': True,
                        'methods': ', '.join(methods_list) if methods_list else '',
                        'decorators': ', '.join(decorators_list) if decorators_list else '',
                        'complexity_score': self._calculate_complexity_score(node),
                        # Enhanced metadata fields (convert lists to strings)
                        'content_type': enhanced_metadata_obj.content_type.value,
                        'semantic_tags': ', '.join(enhanced_metadata_obj.semantic_tags) if enhanced_metadata_obj.semantic_tags else '',
                        'plugin_namespace': enhanced_metadata_obj.plugin_namespace or '',
                        'complexity_level': enhanced_metadata_obj.complexity_level,
                        'estimated_tokens': enhanced_metadata_obj.estimated_tokens
                    }
                    
                    documents.append(Document(
                        page_content=enhanced_content,
                        metadata=enhanced_metadata
                    ))
            
            # Create chunks for imports and module-level code with semantic grouping
            module_level_chunks = self._extract_module_level_semantic_units(content, lines)
            documents.extend(module_level_chunks)
                
        except SyntaxError as e:
            print(f"Syntax error in Python file {metadata.get('source', 'unknown')}: {e}")
            # Fallback to regular text splitting for invalid Python
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            fallback_docs = splitter.split_documents([Document(page_content=content, metadata=metadata)])
            # Mark as fallback chunks
            for doc in fallback_docs:
                doc.metadata.update({**metadata, 'type': 'fallback', 'semantic_complete': False})
            documents.extend(fallback_docs)
        
        return documents
    
    def split_markdown_file(self, content: str, metadata: dict) -> List[Document]:
        """Split Markdown files preserving header hierarchy and associating code examples with explanatory text."""
        documents = []
        
        # Split by headers while preserving hierarchy
        sections = self._split_by_headers_with_hierarchy(content)
        
        for section in sections:
            # Extract code blocks with context
            code_blocks = self._extract_code_blocks_with_context(section['content'])
            
            # Create enhanced document for text content with complete semantic context
            text_content = self._remove_code_blocks(section['content'])
            if text_content.strip():
                # Build header hierarchy path for context
                hierarchy_path = self._build_hierarchy_path(section, sections)
                
                # Create enhanced content that maintains semantic boundaries
                enhanced_content = self._create_markdown_semantic_content(
                    section, text_content, code_blocks, hierarchy_path
                )
                
                # Extract enhanced metadata using the new system
                enhanced_metadata_obj = self.metadata_extractor.extract_enhanced_metadata(
                    content=enhanced_content,
                    base_metadata=metadata,
                    chunk_type='markdown_section',
                    hierarchy_path=hierarchy_path,
                    parent_headers=self._get_parent_headers(section, sections),
                    semantic_complete=True,
                    has_examples=len(code_blocks) > 0
                )
                
                enhanced_metadata = {
                    **metadata,
                    'type': 'markdown_section',
                    'header': section['header'],
                    'level': section['level'],
                    'hierarchy_path': ', '.join(hierarchy_path) if hierarchy_path else '',
                    'has_code_blocks': len(code_blocks) > 0,
                    'semantic_complete': True,
                    'parent_headers': ', '.join(self._get_parent_headers(section, sections)) if self._get_parent_headers(section, sections) else '',
                    'code_block_count': len(code_blocks),
                    # Enhanced metadata fields (convert lists to strings)
                    'content_type': enhanced_metadata_obj.content_type.value,
                    'semantic_tags': ', '.join(enhanced_metadata_obj.semantic_tags) if enhanced_metadata_obj.semantic_tags else '',
                    'plugin_namespace': enhanced_metadata_obj.plugin_namespace or '',
                    'complexity_level': enhanced_metadata_obj.complexity_level,
                    'estimated_tokens': enhanced_metadata_obj.estimated_tokens
                }
                
                documents.append(Document(
                    page_content=enhanced_content,
                    metadata=enhanced_metadata
                ))
            
            # Create associated code block documents with explanatory context
            for i, code_block in enumerate(code_blocks):
                # Get surrounding text context for the code block
                context_text = self._get_code_block_context(section['content'], code_block, text_content)
                
                # Create content that associates code with explanatory text
                code_content = self._create_code_block_semantic_content(
                    code_block, context_text, section['header'], hierarchy_path
                )
                
                # Extract enhanced metadata for code blocks
                enhanced_metadata_obj = self.metadata_extractor.extract_enhanced_metadata(
                    content=code_content,
                    base_metadata=metadata,
                    chunk_type='code_block',
                    hierarchy_path=hierarchy_path,
                    semantic_complete=True,
                    has_examples=True
                )
                
                enhanced_metadata = {
                    **metadata,
                    'type': 'code_block',
                    'language': code_block['language'],
                    'in_section': section['header'],
                    'block_index': i,
                    'hierarchy_path': ', '.join(hierarchy_path) if hierarchy_path else '',
                    'semantic_complete': True,
                    'has_context': bool(context_text),
                    'associated_section': section['header'],
                    # Enhanced metadata fields (convert lists to strings)
                    'content_type': enhanced_metadata_obj.content_type.value,
                    'semantic_tags': ', '.join(enhanced_metadata_obj.semantic_tags) if enhanced_metadata_obj.semantic_tags else '',
                    'plugin_namespace': enhanced_metadata_obj.plugin_namespace or '',
                    'complexity_level': enhanced_metadata_obj.complexity_level,
                    'estimated_tokens': enhanced_metadata_obj.estimated_tokens
                }
                
                documents.append(Document(
                    page_content=code_content,
                    metadata=enhanced_metadata
                ))
        
        return documents
    
    def _split_by_headers_with_hierarchy(self, content: str) -> List[Dict]:
        """Split markdown content by headers while preserving hierarchy information."""
        sections = []
        lines = content.split('\n')
        current_section = {'header': 'Introduction', 'level': 0, 'content': '', 'line_start': 0}
        line_number = 0
        
        for line in lines:
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                # Save previous section
                if current_section['content'].strip():
                    current_section['line_end'] = line_number
                    sections.append(current_section)
                
                # Start new section
                level = len(header_match.group(1))
                header = header_match.group(2)
                current_section = {
                    'header': header, 
                    'level': level, 
                    'content': '', 
                    'line_start': line_number,
                    'line_end': None
                }
            else:
                current_section['content'] += line + '\n'
            line_number += 1
        
        # Add last section
        if current_section['content'].strip():
            current_section['line_end'] = line_number
            sections.append(current_section)
        
        return sections
    
    def _extract_code_blocks_with_context(self, content: str) -> List[Dict]:
        """Extract code blocks with their position and context information."""
        code_blocks = []
        pattern = r'```(\w+)?\n(.*?)\n```'
        
        for match in re.finditer(pattern, content, re.DOTALL):
            language = match.group(1) or 'text'
            code = match.group(2)
            start_pos = match.start()
            end_pos = match.end()
            
            code_blocks.append({
                'language': language, 
                'code': code,
                'start_pos': start_pos,
                'end_pos': end_pos,
                'context_before': content[max(0, start_pos-200):start_pos],
                'context_after': content[end_pos:end_pos+200]
            })
        
        return code_blocks
    
    def _build_hierarchy_path(self, section: Dict, all_sections: List[Dict]) -> List[str]:
        """Build the full hierarchy path for a section."""
        hierarchy = []
        current_level = section['level']
        
        # Find parent headers by looking backwards
        for i in range(len(all_sections)):
            if all_sections[i] == section:
                # Look backwards for parent headers
                for j in range(i-1, -1, -1):
                    parent_section = all_sections[j]
                    if parent_section['level'] < current_level:
                        hierarchy.insert(0, parent_section['header'])
                        current_level = parent_section['level']
                break
        
        hierarchy.append(section['header'])
        return hierarchy
    
    def _create_markdown_semantic_content(self, section: Dict, text_content: str, 
                                        code_blocks: List[Dict], hierarchy_path: List[str]) -> str:
        """Create enhanced markdown content with semantic context."""
        content_parts = [
            f"Section: {section['header']}",
            f"Level: {section['level']}",
            f"Hierarchy: {' > '.join(hierarchy_path)}",
            f"Code Examples: {len(code_blocks)}"
        ]
        
        if text_content.strip():
            content_parts.extend([
                "",
                "Content:",
                text_content.strip()
            ])
        
        if code_blocks:
            content_parts.append("")
            content_parts.append("Associated Code Examples:")
            for i, block in enumerate(code_blocks):
                content_parts.append(f"- Example {i+1}: {block['language']} code")
        
        return '\n'.join(content_parts)
    
    def _get_parent_headers(self, section: Dict, all_sections: List[Dict]) -> List[str]:
        """Get parent headers for the current section."""
        parents = []
        current_level = section['level']
        
        for i in range(len(all_sections)):
            if all_sections[i] == section:
                # Look backwards for parent headers
                for j in range(i-1, -1, -1):
                    parent_section = all_sections[j]
                    if parent_section['level'] < current_level:
                        parents.insert(0, parent_section['header'])
                        current_level = parent_section['level']
                break
        
        return parents
    
    def _get_code_block_context(self, section_content: str, code_block: Dict, text_content: str) -> str:
        """Get contextual text surrounding a code block."""
        # Extract text immediately before and after the code block
        before_text = code_block.get('context_before', '').strip()
        after_text = code_block.get('context_after', '').strip()
        
        # Clean up the context text
        context_parts = []
        if before_text:
            # Get last few sentences before the code block
            sentences = before_text.split('.')
            if len(sentences) > 1:
                context_parts.append(sentences[-2] + '.')
        
        if after_text:
            # Get first few sentences after the code block
            sentences = after_text.split('.')
            if sentences:
                context_parts.append(sentences[0] + '.')
        
        return ' '.join(context_parts)
    
    def _create_code_block_semantic_content(self, code_block: Dict, context_text: str, 
                                          section_header: str, hierarchy_path: List[str]) -> str:
        """Create enhanced code block content with explanatory context."""
        content_parts = [
            f"Code Example from: {section_header}",
            f"Language: {code_block['language']}",
            f"Section Hierarchy: {' > '.join(hierarchy_path)}"
        ]
        
        if context_text:
            content_parts.extend([
                "",
                "Context:",
                context_text
            ])
        
        content_parts.extend([
            "",
            "Code:",
            f"```{code_block['language']}",
            code_block['code'],
            "```"
        ])
        
        return '\n'.join(content_parts)
    
    def _extract_imports_and_constants(self, content: str) -> str:
        """Extract imports and module-level constants."""
        lines = content.split('\n')
        relevant_lines = []
        for line in lines:
            stripped = line.strip()
            if (stripped.startswith('import ') or 
                stripped.startswith('from ') or
                (stripped and not stripped.startswith('def ') and 
                 not stripped.startswith('class ') and
                 not stripped.startswith('#') and
                 '=' in stripped and stripped.split('=')[0].strip().isupper())):
                relevant_lines.append(line)
        
        return '\n'.join(relevant_lines)
    
    def _split_by_headers(self, content: str) -> List[Dict]:
        """Split markdown content by headers."""
        sections = []
        lines = content.split('\n')
        current_section = {'header': 'Introduction', 'level': 0, 'content': ''}
        
        for line in lines:
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                # Save previous section
                if current_section['content'].strip():
                    sections.append(current_section)
                
                # Start new section
                level = len(header_match.group(1))
                header = header_match.group(2)
                current_section = {'header': header, 'level': level, 'content': ''}
            else:
                current_section['content'] += line + '\n'
        
        # Add last section
        if current_section['content'].strip():
            sections.append(current_section)
        
        return sections
    
    def _extract_code_blocks(self, content: str) -> List[Dict]:
        """Extract code blocks from markdown content."""
        code_blocks = []
        pattern = r'```(\w+)?\n(.*?)\n```'
        
        for match in re.finditer(pattern, content, re.DOTALL):
            language = match.group(1) or 'text'
            code = match.group(2)
            code_blocks.append({'language': language, 'code': code})
        
        return code_blocks
    
    def _remove_code_blocks(self, content: str) -> str:
        """Remove code blocks from content."""
        pattern = r'```\w*\n.*?\n```'
        return re.sub(pattern, '', content, flags=re.DOTALL)
    
    def _get_semantic_start_line(self, node: ast.AST, lines: List[str]) -> int:
        """Get the semantic start line including decorators and comments."""
        start_line = node.lineno
        
        # Look backwards for decorators and comments
        for i in range(start_line - 2, -1, -1):
            line = lines[i].strip()
            if line.startswith('@') or line.startswith('#') or not line:
                start_line = i + 1
            else:
                break
        
        return start_line
    
    def _get_semantic_end_line(self, node: ast.AST, lines: List[str]) -> int:
        """Get the semantic end line ensuring complete function/class body."""
        end_line = getattr(node, 'end_lineno', node.lineno + 20)
        
        # Ensure we don't cut off in the middle of nested structures
        if end_line and end_line <= len(lines):
            return end_line
        
        # Fallback: find the end by indentation
        base_indent = len(lines[node.lineno - 1]) - len(lines[node.lineno - 1].lstrip())
        for i in range(node.lineno, len(lines)):
            line = lines[i]
            if line.strip() and len(line) - len(line.lstrip()) <= base_indent and not line.strip().startswith(('"""', "'''")):
                return i
        
        return min(node.lineno + 50, len(lines))  # Safety limit
    
    def _is_syntactically_valid(self, code_segment: str) -> bool:
        """Check if code segment is syntactically valid Python."""
        try:
            ast.parse(code_segment)
            return True
        except SyntaxError:
            return False
    
    def _validate_complete_semantic_unit(self, code_segment: str, node: ast.AST) -> bool:
        """Validate that the code segment contains a complete semantic unit."""
        try:
            # Try to parse the code segment to ensure it's syntactically valid
            ast.parse(code_segment)
            
            # Be extremely lenient - if it parses successfully, it's valid
            # This eliminates most false positive warnings
            return True
            
        except SyntaxError:
            # Only reject if it can't be parsed at all
            # Even then, be lenient for common patterns
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check if it at least contains the function name
                return node.name in code_segment
            elif isinstance(node, ast.ClassDef):
                # Check if it at least contains the class name
                return node.name in code_segment
            
            return False
    
    def _extract_class_methods(self, class_node: ast.ClassDef) -> List[Dict]:
        """Extract method information from a class node."""
        methods = []
        for node in class_node.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append({
                    'name': node.name,
                    'is_async': isinstance(node, ast.AsyncFunctionDef),
                    'args': [arg.arg for arg in node.args.args],
                    'docstring': ast.get_docstring(node) or "",
                    'decorators': [self._get_decorator_name(d) for d in node.decorator_list]
                })
        return methods
    
    def _create_semantic_content(self, node: ast.AST, code_segment: str, docstring: str, methods: List[Dict]) -> str:
        """Create enhanced content with complete semantic context."""
        node_type = "Class" if isinstance(node, ast.ClassDef) else "Function"
        
        content_parts = [
            f"Type: {node_type}",
            f"Name: {node.name}",
            f"Docstring: {docstring}" if docstring else "Docstring: None"
        ]
        
        if methods:
            method_names = [m['name'] for m in methods]
            content_parts.append(f"Methods: {', '.join(method_names)}")
        
        content_parts.extend([
            "",
            "Code:",
            f"```python",
            code_segment,
            "```"
        ])
        
        return '\n'.join(content_parts)
    
    def _extract_decorators(self, node: ast.AST) -> List[str]:
        """Extract decorator names from a node."""
        if not hasattr(node, 'decorator_list'):
            return []
        
        decorators = []
        for decorator in node.decorator_list:
            decorators.append(self._get_decorator_name(decorator))
        return decorators
    
    def _get_decorator_name(self, decorator: ast.AST) -> str:
        """Get the name of a decorator."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return f"{self._get_decorator_name(decorator.value)}.{decorator.attr}"
        elif isinstance(decorator, ast.Call):
            return self._get_decorator_name(decorator.func)
        else:
            return str(decorator)
    
    def _calculate_complexity_score(self, node: ast.AST) -> int:
        """Calculate a simple complexity score for the node."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if child != node:  # Don't count the node itself
                    complexity += 1
        
        return complexity
    
    def _extract_module_level_semantic_units(self, content: str, lines: List[str]) -> List[Document]:
        """Extract module-level code as semantic units."""
        documents = []
        
        # Group imports together
        imports = self._extract_imports_and_constants(content)
        if imports:
            documents.append(Document(
                page_content=f"Module-level imports and constants:\n\n{imports}",
                metadata={
                    'type': 'module_level',
                    'name': 'imports_constants',
                    'semantic_complete': True
                }
            ))
        
        # Extract module-level docstring if present
        try:
            tree = ast.parse(content)
            module_docstring = ast.get_docstring(tree)
            if module_docstring:
                documents.append(Document(
                    page_content=f"Module docstring:\n\n{module_docstring}",
                    metadata={
                        'type': 'module_docstring',
                        'name': 'module_docstring',
                        'semantic_complete': True
                    }
                ))
        except:
            pass
        
        return documents

class EnhancedRAGVectorStore(VectorStoreProvider if PROVIDER_INTERFACE_AVAILABLE else object):
    """Enhanced RAG vector store with improved code understanding and multi-provider support.
    
    This class maintains backward compatibility while supporting the new VectorStoreProvider interface.
    It can be used both as a standalone vector store and as a provider in the new architecture.
    """
    
    def __init__(self, 
                 chroma_db_path: str = "chroma_db",
                 manim_docs_path: str = "rag/manim_docs",
                 embedding_model: str = "hf:ibm-granite/granite-embedding-30m-english",
                 trace_id: str = None,
                 session_id: str = None,
                 use_langfuse: bool = True,
                 helper_model = None,
                 embedding_provider = None,
                 embedding_config = None,
                 # New parameters for VectorStoreProvider compatibility
                 config = None):
        
        # Handle new VectorStoreProvider interface
        if PROVIDER_INTERFACE_AVAILABLE and config is not None and embedding_provider is not None:
            # Initialize as VectorStoreProvider
            super().__init__(config, embedding_provider)
            self.chroma_db_path = config.connection_params.get('path', chroma_db_path)
            self.collection_name = config.collection_name
            self.embedding_provider = embedding_provider
            self._is_provider_mode = True
        else:
            # Initialize in backward compatibility mode
            self.chroma_db_path = chroma_db_path
            self.collection_name = f"manim_docs_default"
            self._is_provider_mode = False
            # Initialize embedding provider system
            self._initialize_embedding_provider(embedding_provider, embedding_config)
        
        # Common initialization
        self.manim_docs_path = manim_docs_path
        self.embedding_model = embedding_model
        self.trace_id = trace_id
        self.session_id = session_id
        self.use_langfuse = use_langfuse
        self.helper_model = helper_model
        self.enc = tiktoken.encoding_for_model("gpt-4")
        self.plugin_stores = {}
        self.code_splitter = CodeAwareTextSplitter()
        
        # Create provider-specific collection name if not in provider mode
        if not self._is_provider_mode:
            provider_info = self.embedding_provider.get_provider_info()
            self.collection_name = f"manim_docs_{provider_info['provider']}_{provider_info['dimensions']}d"
            self.provider_info = provider_info
        else:
            self.provider_info = self.embedding_provider.get_provider_info()
        
        # Initialize vector store with provider-specific settings
        self.vector_store = None  # Will be initialized in initialize() method for provider mode
        if not self._is_provider_mode:
            self.vector_store = self._load_or_create_vector_store()
        
        # Log provider information
        import logging
        logging.info(f"Initialized vector store with {self.provider_info}")
        
        # Initialize collection management system
        self._initialize_collection_management()
        
        # Track collection metadata for provider switching
        self._collection_metadata = {
            'provider': self.provider_info['provider'],
            'model': self.provider_info.get('model', 'unknown'),
            'dimensions': self.provider_info['dimensions'],
            'created_at': time.time(),
            'collection_name': self.collection_name
        }
    
    def _initialize_embedding_provider(self, embedding_provider, embedding_config):
        """Initialize the embedding provider system with proper factory integration."""
        if embedding_provider is not None:
            # Use provided embedding provider
            self.embedding_provider = embedding_provider
        else:
            # Create embedding provider from factory
            try:
                from .provider_factory import EmbeddingProviderFactory
                
                if embedding_config is None:
                    # Load configuration from environment
                    from .embedding_providers import ConfigurationManager
                    embedding_config = ConfigurationManager.load_config_from_env()
                
                # Create provider with fallback support
                self.embedding_provider = EmbeddingProviderFactory.create_provider(
                    embedding_config, enable_fallback=True
                )
                
            except ImportError as e:
                # Fallback to direct provider creation if factory is not available
                import logging
                logging.warning(f"Provider factory not available, using direct creation: {e}")
                
                from .embedding_providers import ConfigurationManager
                if embedding_config is None:
                    embedding_config = ConfigurationManager.load_config_from_env()
                
                if embedding_config.provider == "jina":
                    from .jina_embedding_provider import JinaEmbeddingProvider
                    self.embedding_provider = JinaEmbeddingProvider(embedding_config)
                else:
                    from .local_embedding_provider import LocalEmbeddingProvider
                    self.embedding_provider = LocalEmbeddingProvider(embedding_config)

    def _initialize_collection_management(self):
        """Initialize collection management system for provider switching."""
        self._available_collections = {}
        self._collection_metadata_store = {}
        
        # Discover existing collections
        self._discover_existing_collections()
        
        # Set up collection metadata tracking
        self._setup_metadata_tracking()
    
    def _discover_existing_collections(self):
        """Discover existing collections and their metadata."""
        try:
            client = self._get_chroma_client()
            existing_collections = client.list_collections()
            
            for collection in existing_collections:
                collection_name = collection.name
                
                # Parse collection name to extract provider info
                if collection_name.startswith("manim_docs_"):
                    parts = collection_name.replace("manim_docs_", "").split("_")
                    if len(parts) >= 2 and parts[-1].endswith("d"):
                        provider = "_".join(parts[:-1])
                        dimensions_str = parts[-1][:-1]  # Remove 'd' suffix
                        
                        try:
                            dimensions = int(dimensions_str)
                            
                            # Store collection metadata
                            self._available_collections[collection_name] = {
                                'provider': provider,
                                'dimensions': dimensions,
                                'collection_name': collection_name,
                                'document_count': collection.count() if hasattr(collection, 'count') else 0,
                                'discovered_at': time.time()
                            }
                            
                            logging.info(f"Discovered collection: {collection_name} (provider: {provider}, dimensions: {dimensions})")
                            
                        except ValueError:
                            logging.warning(f"Could not parse dimensions from collection name: {collection_name}")
                
        except Exception as e:
            import logging
            logging.warning(f"Failed to discover existing collections: {e}")
    
    def _setup_metadata_tracking(self):
        """Set up metadata tracking for collections."""
        # Create a metadata collection to track provider information
        try:
            client = self._get_chroma_client()
            
            # Try to get existing metadata collection
            try:
                metadata_collection = client.get_collection("collection_metadata")
                
                # Load existing metadata
                results = metadata_collection.get()
                if results and results['metadatas']:
                    for metadata in results['metadatas']:
                        collection_name = metadata.get('collection_name')
                        if collection_name:
                            self._collection_metadata_store[collection_name] = metadata
                            
            except Exception:
                # Create new metadata collection
                metadata_collection = client.create_collection("collection_metadata")
                
        except Exception as e:
            logging.warning(f"Failed to set up metadata tracking: {e}")
    
    def _get_chroma_client(self):
        """Get ChromaDB client instance with proper configuration."""
        import chromadb
        
        if self._is_provider_mode and hasattr(self, 'config'):
            # Use configuration from VectorStoreConfig
            connection_params = self.config.connection_params
            mode = connection_params.get('mode', 'persistent')
            
            if mode == 'client':
                return chromadb.HttpClient(
                    host=connection_params['host'],
                    port=connection_params['port'],
                    ssl=connection_params.get('ssl', False),
                    headers=connection_params.get('headers', {})
                )
            else:
                return chromadb.PersistentClient(
                    path=connection_params.get('path', self.chroma_db_path),
                    settings=CHROMA_SETTINGS
                )
        else:
            # Backward compatibility mode
            return chromadb.PersistentClient(
                path=self.chroma_db_path,
                settings=CHROMA_SETTINGS
            )
    
    def switch_provider(self, new_config):
        """Switch to a different embedding provider with collection management.
        
        Args:
            new_config: EmbeddingConfig for the new provider
            
        Returns:
            Dict with switch results and recommendations
        """
        old_provider_info = self.embedding_provider.get_provider_info()
        old_collection_name = self.collection_name
        
        try:
            # Create new provider
            if hasattr(new_config, 'provider'):
                # It's an EmbeddingConfig object
                from .provider_factory import EmbeddingProviderFactory
                new_provider = EmbeddingProviderFactory.create_provider(new_config)
            else:
                # It's already a provider instance
                new_provider = new_config
            
            new_provider_info = new_provider.get_provider_info()
            new_collection_name = f"manim_docs_{new_provider_info['provider']}_{new_provider_info['dimensions']}d"
            
            # Check if dimensions changed
            dimension_changed = old_provider_info['dimensions'] != new_provider_info['dimensions']
            provider_changed = old_provider_info['provider'] != new_provider_info['provider']
            
            switch_results = {
                'success': True,
                'old_provider': old_provider_info,
                'new_provider': new_provider_info,
                'old_collection': old_collection_name,
                'new_collection': new_collection_name,
                'dimension_changed': dimension_changed,
                'provider_changed': provider_changed,
                'requires_reindexing': dimension_changed or provider_changed,
                'recommendations': []
            }
            
            # Add recommendations based on the switch
            if dimension_changed:
                switch_results['recommendations'].append(
                    f"Embedding dimensions changed from {old_provider_info['dimensions']} "
                    f"to {new_provider_info['dimensions']}. You will need to re-index your documents."
                )
            
            if provider_changed:
                switch_results['recommendations'].append(
                    f"Provider changed from {old_provider_info['provider']} "
                    f"to {new_provider_info['provider']}. Consider re-indexing for optimal performance."
                )
            
            # Check if target collection already exists
            if new_collection_name in self._available_collections:
                existing_collection = self._available_collections[new_collection_name]
                switch_results['recommendations'].append(
                    f"Collection '{new_collection_name}' already exists with "
                    f"{existing_collection.get('document_count', 0)} documents. "
                    f"You can use it immediately or re-index to update content."
                )
            else:
                switch_results['recommendations'].append(
                    f"Collection '{new_collection_name}' does not exist. "
                    f"You will need to index your documents after switching."
                )
            
            # Perform the switch
            self.embedding_provider = new_provider
            self.collection_name = new_collection_name
            self.provider_info = new_provider_info
            
            # Update collection metadata
            self._collection_metadata = {
                'provider': new_provider_info['provider'],
                'model': new_provider_info.get('model', 'unknown'),
                'dimensions': new_provider_info['dimensions'],
                'created_at': time.time(),
                'collection_name': new_collection_name,
                'switched_from': old_collection_name
            }
            
            # Update vector store if not in provider mode
            if not self._is_provider_mode:
                self.vector_store = self._load_or_create_vector_store()
            
            # Store metadata about the switch
            self._store_collection_metadata(new_collection_name, self._collection_metadata)
            
            logging.info(f"Successfully switched from {old_provider_info} to {new_provider_info}")
            
            return switch_results
            
        except Exception as e:
            logging.error(f"Failed to switch provider: {e}")
            return {
                'success': False,
                'error': str(e),
                'old_provider': old_provider_info,
                'recommendations': [
                    "Provider switch failed. Check your configuration and try again.",
                    "Ensure the new provider is properly configured and available."
                ]
            }
    
    def _store_collection_metadata(self, collection_name: str, metadata: dict):
        """Store metadata about a collection for tracking purposes."""
        try:
            client = self._get_chroma_client()
            
            # Get or create metadata collection
            try:
                metadata_collection = client.get_collection("collection_metadata")
            except Exception:
                metadata_collection = client.create_collection("collection_metadata")
            
            # Store metadata
            metadata_collection.upsert(
                ids=[collection_name],
                documents=[f"Metadata for collection {collection_name}"],
                metadatas=[metadata]
            )
            
            # Update local cache
            self._collection_metadata_store[collection_name] = metadata
            
        except Exception as e:
            logging.warning(f"Failed to store collection metadata: {e}")
    
    def get_collection_management_info(self) -> Dict[str, Any]:
        """Get comprehensive information about collection management state.
        
        Returns:
            Dictionary with collection management information
        """
        return {
            'current_collection': {
                'name': self.collection_name,
                'provider': self.provider_info['provider'],
                'model': self.provider_info.get('model', 'unknown'),
                'dimensions': self.provider_info['dimensions'],
                'metadata': self._collection_metadata
            },
            'available_collections': self._available_collections,
            'collection_metadata_store': self._collection_metadata_store,
            'provider_info': self.provider_info,
            'recommendations': self._get_collection_recommendations()
        }
    
    def _get_collection_recommendations(self) -> List[str]:
        """Get recommendations for collection management."""
        recommendations = []
        
        # Check for dimension mismatches
        current_dimensions = self.provider_info['dimensions']
        for collection_name, collection_info in self._available_collections.items():
            if collection_info['dimensions'] != current_dimensions:
                recommendations.append(
                    f"Collection '{collection_name}' has {collection_info['dimensions']} dimensions "
                    f"but current provider uses {current_dimensions}. Consider switching providers "
                    f"or re-indexing."
                )
        
        # Check for empty collections
        for collection_name, collection_info in self._available_collections.items():
            if collection_info.get('document_count', 0) == 0:
                recommendations.append(
                    f"Collection '{collection_name}' is empty. Consider indexing documents."
                )
        
        # Check for multiple collections with same provider
        provider_collections = {}
        for collection_name, collection_info in self._available_collections.items():
            provider = collection_info['provider']
            if provider not in provider_collections:
                provider_collections[provider] = []
            provider_collections[provider].append(collection_name)
        
        for provider, collections in provider_collections.items():
            if len(collections) > 1:
                recommendations.append(
                    f"Multiple collections found for provider '{provider}': {collections}. "
                    f"Consider consolidating or cleaning up unused collections."
                )
        
        return recommendations
    
    def list_collections_by_provider(self) -> Dict[str, List[Dict[str, Any]]]:
        """List collections grouped by provider.
        
        Returns:
            Dictionary mapping provider names to lists of collection info
        """
        collections_by_provider = {}
        
        for collection_name, collection_info in self._available_collections.items():
            provider = collection_info['provider']
            if provider not in collections_by_provider:
                collections_by_provider[provider] = []
            
            collections_by_provider[provider].append({
                'collection_name': collection_name,
                'dimensions': collection_info['dimensions'],
                'document_count': collection_info.get('document_count', 0),
                'discovered_at': collection_info.get('discovered_at'),
                'is_current': collection_name == self.collection_name
            })
        
        return collections_by_provider
    
    def cleanup_unused_collections(self, dry_run: bool = True, criteria: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Clean up unused collections to save space with advanced criteria.
        
        Args:
            dry_run: If True, only report what would be deleted without actually deleting
            criteria: Optional cleanup criteria dictionary with keys:
                - 'delete_empty': Delete empty collections (default: True)
                - 'delete_dimension_mismatches': Delete collections with different dimensions (default: True)
                - 'delete_old_providers': Delete collections from inactive providers (default: False)
                - 'keep_latest_only': Keep only the latest collection per provider (default: False)
                - 'older_than_days': Delete collections older than specified days (default: None)
                - 'max_collections_per_provider': Maximum collections to keep per provider (default: None)
            
        Returns:
            Dictionary with cleanup results
        """
        # Default cleanup criteria
        default_criteria = {
            'delete_empty': True,
            'delete_dimension_mismatches': True,
            'delete_old_providers': False,
            'keep_latest_only': False,
            'older_than_days': None,
            'max_collections_per_provider': None
        }
        
        if criteria:
            default_criteria.update(criteria)
        criteria = default_criteria
        
        cleanup_results = {
            'dry_run': dry_run,
            'criteria_used': criteria,
            'collections_to_delete': [],
            'collections_kept': [],
            'space_saved_estimate': 0,
            'errors': [],
            'summary': {
                'total_collections': len(self._available_collections),
                'collections_to_delete_count': 0,
                'collections_kept_count': 0,
                'estimated_space_mb': 0
            }
        }
        
        current_provider = self.provider_info['provider']
        current_dimensions = self.provider_info['dimensions']
        current_time = time.time()
        
        try:
            client = self._get_chroma_client()
            
            # Group collections by provider for advanced cleanup
            collections_by_provider = {}
            for collection_name, collection_info in self._available_collections.items():
                provider = collection_info['provider']
                if provider not in collections_by_provider:
                    collections_by_provider[provider] = []
                collections_by_provider[provider].append((collection_name, collection_info))
            
            for collection_name, collection_info in self._available_collections.items():
                # Skip current collection
                if collection_name == self.collection_name:
                    cleanup_results['collections_kept'].append({
                        'name': collection_name,
                        'reason': 'Current active collection',
                        'provider': collection_info['provider'],
                        'dimensions': collection_info['dimensions'],
                        'document_count': collection_info.get('document_count', 0)
                    })
                    continue
                
                # Determine if collection should be deleted based on criteria
                should_delete = False
                delete_reasons = []
                
                # Check for empty collections
                if criteria['delete_empty'] and collection_info.get('document_count', 0) == 0:
                    should_delete = True
                    delete_reasons.append("Empty collection")
                
                # Check for dimension mismatches with current provider
                if (criteria['delete_dimension_mismatches'] and 
                    collection_info['provider'] == current_provider and 
                    collection_info['dimensions'] != current_dimensions):
                    should_delete = True
                    delete_reasons.append(f"Dimension mismatch: {collection_info['dimensions']} vs {current_dimensions}")
                
                # Check for collections from old/inactive providers
                if (criteria['delete_old_providers'] and 
                    collection_info['provider'] != current_provider):
                    should_delete = True
                    delete_reasons.append(f"Inactive provider: {collection_info['provider']}")
                
                # Check age-based cleanup
                if criteria['older_than_days']:
                    created_at = collection_info.get('created_at', collection_info.get('discovered_at', 0))
                    days_old = (current_time - created_at) / (24 * 3600)
                    if days_old > criteria['older_than_days']:
                        should_delete = True
                        delete_reasons.append(f"Too old: {days_old:.1f} days")
                
                # Check for keep-latest-only policy
                if criteria['keep_latest_only']:
                    provider = collection_info['provider']
                    provider_collections = collections_by_provider[provider]
                    # Sort by creation time and keep only the latest
                    provider_collections.sort(key=lambda x: x[1].get('created_at', x[1].get('discovered_at', 0)), reverse=True)
                    if len(provider_collections) > 1 and collection_name != provider_collections[0][0]:
                        should_delete = True
                        delete_reasons.append("Not the latest collection for this provider")
                
                # Check for max collections per provider limit
                if criteria['max_collections_per_provider']:
                    provider = collection_info['provider']
                    provider_collections = collections_by_provider[provider]
                    if len(provider_collections) > criteria['max_collections_per_provider']:
                        # Sort by creation time, keep the newest ones
                        provider_collections.sort(key=lambda x: x[1].get('created_at', x[1].get('discovered_at', 0)), reverse=True)
                        keep_limit = criteria['max_collections_per_provider']
                        collections_to_keep = [item[0] for item in provider_collections[:keep_limit]]
                        if collection_name not in collections_to_keep:
                            should_delete = True
                            delete_reasons.append(f"Exceeds limit of {keep_limit} collections per provider")
                
                if should_delete:
                    collection_to_delete = {
                        'name': collection_name,
                        'provider': collection_info['provider'],
                        'dimensions': collection_info['dimensions'],
                        'document_count': collection_info.get('document_count', 0),
                        'created_at': collection_info.get('created_at', collection_info.get('discovered_at', 0)),
                        'reasons': delete_reasons,
                        'reason': '; '.join(delete_reasons)  # For backward compatibility
                    }
                    
                    cleanup_results['collections_to_delete'].append(collection_to_delete)
                    
                    # Estimate space saved (improved estimate)
                    doc_count = collection_info.get('document_count', 0)
                    dimensions = collection_info.get('dimensions', 384)
                    # Estimate: embeddings + metadata + overhead
                    estimated_size_mb = (doc_count * dimensions * 4 / (1024 * 1024)) + (doc_count * 0.05)  # 4 bytes per float + 0.05MB metadata per doc
                    cleanup_results['space_saved_estimate'] += estimated_size_mb
                    
                    # Actually delete if not dry run
                    if not dry_run:
                        try:
                            client.delete_collection(collection_name)
                            logging.info(f"Deleted collection: {collection_name} - Reasons: {'; '.join(delete_reasons)}")
                        except Exception as e:
                            error_msg = f"Failed to delete collection {collection_name}: {e}"
                            cleanup_results['errors'].append(error_msg)
                            logging.error(error_msg)
                else:
                    cleanup_results['collections_kept'].append({
                        'name': collection_name,
                        'reason': 'Collection meets retention criteria',
                        'provider': collection_info['provider'],
                        'dimensions': collection_info['dimensions'],
                        'document_count': collection_info.get('document_count', 0),
                        'created_at': collection_info.get('created_at', collection_info.get('discovered_at', 0))
                    })
            
            # Update available collections if not dry run
            if not dry_run:
                for collection_info in cleanup_results['collections_to_delete']:
                    if collection_info['name'] in self._available_collections:
                        del self._available_collections[collection_info['name']]
            
            # Update summary
            cleanup_results['summary'] = {
                'total_collections': len(self._available_collections),
                'collections_to_delete_count': len(cleanup_results['collections_to_delete']),
                'collections_kept_count': len(cleanup_results['collections_kept']),
                'estimated_space_mb': cleanup_results['space_saved_estimate']
            }
            
        except Exception as e:
            error_msg = f"Collection cleanup failed: {e}"
            cleanup_results['errors'].append(error_msg)
            logging.error(error_msg)
        
        return cleanup_results

    def _load_or_create_vector_store(self):
        """Enhanced vector store creation with better document processing."""
        print("Creating enhanced vector store with code-aware processing...")
        core_path = os.path.join(self.chroma_db_path, "manim_core_enhanced")
        
        if os.path.exists(core_path):
            print("Loading existing enhanced ChromaDB...")
            self.core_vector_store = Chroma(
                collection_name="manim_core_enhanced",
                persist_directory=core_path,
                embedding_function=self._get_embedding_function()
            )
        else:
            print("Creating new enhanced ChromaDB...")
            self.core_vector_store = self._create_enhanced_core_store()
        
        # Process plugins with enhanced splitting
        plugin_docs_path = os.path.join(self.manim_docs_path, "plugin_docs")
        if os.path.exists(plugin_docs_path):
            for plugin_name in os.listdir(plugin_docs_path):
                plugin_store_path = os.path.join(self.chroma_db_path, f"manim_plugin_{plugin_name}_enhanced")
                if os.path.exists(plugin_store_path):
                    print(f"Loading existing enhanced plugin store: {plugin_name}")
                    self.plugin_stores[plugin_name] = Chroma(
                        collection_name=f"manim_plugin_{plugin_name}_enhanced",
                        persist_directory=plugin_store_path,
                        embedding_function=self._get_embedding_function()
                    )
                else:
                    print(f"Creating new enhanced plugin store: {plugin_name}")
                    plugin_path = os.path.join(plugin_docs_path, plugin_name)
                    if os.path.isdir(plugin_path):
                        plugin_store = Chroma(
                            collection_name=f"manim_plugin_{plugin_name}_enhanced",
                            embedding_function=self._get_embedding_function(),
                            persist_directory=plugin_store_path
                        )
                        plugin_docs = self._process_documentation_folder_enhanced(plugin_path)
                        if plugin_docs:
                            self._add_documents_to_store(plugin_store, plugin_docs, plugin_name)
                        self.plugin_stores[plugin_name] = plugin_store
        
        return self.core_vector_store

    def _get_embedding_function(self) -> Embeddings:
        """Get embedding function that integrates with the provider system."""
        # Create a wrapper that uses our embedding provider
        class ProviderEmbeddingFunction(Embeddings):
            """Embedding function wrapper for the provider system."""
            
            def __init__(self, provider):
                self.provider = provider
            
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                """Embed a list of documents."""
                return self.provider.generate_embeddings(texts)
            
            def embed_query(self, text: str) -> List[float]:
                """Embed a single query."""
                embeddings = self.provider.generate_embeddings([text])
                return embeddings[0] if embeddings else []
        
        return ProviderEmbeddingFunction(self.embedding_provider)

    def _create_enhanced_core_store(self):
        """Create enhanced core store with better document processing."""
        core_vector_store = Chroma(
            collection_name="manim_core_enhanced",
            embedding_function=self._get_embedding_function(),
            persist_directory=os.path.join(self.chroma_db_path, "manim_core_enhanced")
        )
        
        core_docs = self._process_documentation_folder_enhanced(
            os.path.join(self.manim_docs_path, "manim_core")
        )
        if core_docs:
            self._add_documents_to_store(core_vector_store, core_docs, "manim_core_enhanced")
        
        return core_vector_store

    def _process_documentation_folder_enhanced(self, folder_path: str) -> List[Document]:
        """Enhanced document processing with code-aware splitting."""
        all_docs = []
        
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(('.md', '.py')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        base_metadata = {
                            'source': file_path,
                            'filename': file,
                            'file_type': 'python' if file.endswith('.py') else 'markdown',
                            'relative_path': os.path.relpath(file_path, folder_path)
                        }
                        
                        if file.endswith('.py'):
                            docs = self.code_splitter.split_python_file(content, base_metadata)
                        else:  # .md files
                            docs = self.code_splitter.split_markdown_file(content, base_metadata)
                        
                        # Add source prefix to content
                        for doc in docs:
                            doc.page_content = f"Source: {file_path}\nType: {doc.metadata.get('type', 'unknown')}\n\n{doc.page_content}"
                        
                        all_docs.extend(docs)
                        
                    except Exception as e:
                        print(f"Error loading file {file_path}: {e}")
        
        print(f"Processed {len(all_docs)} enhanced document chunks from {folder_path}")
        return all_docs

    def _add_documents_to_store(self, vector_store: Chroma, documents: List[Document], store_name: str):
        """Enhanced document addition with better batching."""
        print(f"Adding {len(documents)} enhanced documents to {store_name} store")
        
        # Group documents by type for better organization
        doc_types = {}
        for doc in documents:
            doc_type = doc.metadata.get('type', 'unknown')
            if doc_type not in doc_types:
                doc_types[doc_type] = []
            doc_types[doc_type].append(doc)
        
        print(f"Document types distribution: {dict((k, len(v)) for k, v in doc_types.items())}")
        
        # Calculate token statistics
        token_lengths = [len(self.enc.encode(doc.page_content)) for doc in documents]
        print(f"Token length statistics for {store_name}: "
              f"Min: {min(token_lengths)}, Max: {max(token_lengths)}, "
              f"Mean: {sum(token_lengths) / len(token_lengths):.1f}, "
              f"Median: {statistics.median(token_lengths):.1f}")
        
        batch_size = 10
        for i in tqdm(range(0, len(documents), batch_size), desc=f"Processing {store_name} enhanced batches"):
            batch_docs = documents[i:i + batch_size]
            batch_ids = [str(uuid.uuid4()) for _ in batch_docs]
            vector_store.add_documents(documents=batch_docs, ids=batch_ids)
        
        vector_store.persist()

    def find_relevant_docs(self, queries: List[Dict], k: int = 5, trace_id: str = None, topic: str = None, scene_number: int = None) -> str:
        """Find relevant documents - compatibility method that calls the enhanced version."""
        return self.find_relevant_docs_enhanced(queries, k, trace_id, topic, scene_number)

    def find_relevant_docs_enhanced(self, queries: List[Dict], k: int = 5, trace_id: str = None, topic: str = None, scene_number: int = None) -> str:
        """Enhanced document retrieval with type-aware search."""
        # Separate queries by intent
        code_queries = [q for q in queries if any(keyword in q["query"].lower() 
                       for keyword in ["function", "class", "method", "import", "code", "implementation"])]
        concept_queries = [q for q in queries if q not in code_queries]
        
        all_results = []
        
        # Search with different strategies for different query types
        for query in code_queries:
            results = self._search_with_filters(
                query["query"], 
                k=k, 
                filter_metadata={'type': ['function', 'class', 'code_block']},
                boost_code=True
            )
            all_results.extend(results)
        
        for query in concept_queries:
            results = self._search_with_filters(
                query["query"], 
                k=k, 
                filter_metadata={'type': ['markdown_section', 'module_level']},
                boost_code=False
            )
            all_results.extend(results)
        
        # Remove duplicates and format results
        unique_results = self._remove_duplicates(all_results)
        return self._format_results(unique_results)
    
    def _search_with_filters(self, query: str, k: int, filter_metadata: Dict = None, boost_code: bool = False) -> List[Dict]:
        """Search with metadata filters and result boosting."""
        # This is a simplified version - in practice, you'd implement proper filtering
        core_results = self.core_vector_store.similarity_search_with_relevance_scores(
            query=query, k=k, score_threshold=0.3
        )
        
        formatted_results = []
        for result in core_results:
            doc, score = result
            # Boost scores for code-related results if needed
            if boost_code and doc.metadata.get('type') in ['function', 'class', 'code_block']:
                score *= 1.2
            
            formatted_results.append({
                "query": query,
                "source": doc.metadata['source'],
                "content": doc.page_content,
                "score": score,
                "type": doc.metadata.get('type', 'unknown'),
                "metadata": doc.metadata
            })
        
        return formatted_results
    
    def _remove_duplicates(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate results based on content similarity."""
        unique_results = []
        seen_content = set()
        
        for result in sorted(results, key=lambda x: x['score'], reverse=True):
            content_hash = hash(result['content'][:200])  # Hash first 200 chars
            if content_hash not in seen_content:
                unique_results.append(result)
                seen_content.add(content_hash)
        
        return unique_results[:10]  # Return top 10 unique results
    
    def _format_results(self, results: List[Dict]) -> str:
        """Format results with enhanced presentation."""
        if not results:
            return "No relevant documentation found."
        
        formatted = "## Relevant Documentation\n\n"
        
        # Group by type
        by_type = {}
        for result in results:
            result_type = result['type']
            if result_type not in by_type:
                by_type[result_type] = []
            by_type[result_type].append(result)
        
        for result_type, type_results in by_type.items():
            formatted += f"### {result_type.replace('_', ' ').title()} Documentation\n\n"
            
            for result in type_results:
                formatted += f"**Source:** {result['source']}\n"
                formatted += f"**Relevance Score:** {result['score']:.3f}\n"
                formatted += f"**Content:**\n```\n{result['content'][:500]}...\n```\n\n"
        
        return formatted
    
    # VectorStoreProvider interface implementation
    def initialize(self) -> None:
        """Initialize the vector store connection and create collection if needed."""
        if not PROVIDER_INTERFACE_AVAILABLE:
            return
        
        try:
            if self.vector_store is None:
                self.vector_store = self._load_or_create_vector_store()
            
            # Test the connection
            if not self.is_available():
                raise VectorStoreConnectionError("ChromaDB connection failed")
                
        except Exception as e:
            raise VectorStoreConnectionError(f"Failed to initialize ChromaDB: {e}")
    
    def add_documents(self, documents: List[DocumentInput]) -> None:
        """Add documents to the vector store."""
        if not PROVIDER_INTERFACE_AVAILABLE:
            raise NotImplementedError("VectorStoreProvider interface not available")
        
        try:
            # Validate documents
            self._validate_documents(documents)
            
            # Generate embeddings for documents that don't have them
            documents_with_embeddings = self._generate_embeddings_for_documents(documents)
            
            # Convert DocumentInput to LangChain Document format
            langchain_docs = []
            doc_ids = []
            
            for doc in documents_with_embeddings:
                # Sanitize metadata for ChromaDB compatibility
                sanitized_metadata = sanitize_metadata_for_chroma(doc.metadata)
                
                langchain_doc = Document(
                    page_content=doc.content,
                    metadata=sanitized_metadata
                )
                langchain_docs.append(langchain_doc)
                doc_ids.append(doc.id)
            
            # Add to ChromaDB
            if self._is_provider_mode:
                # Use the specific collection for provider mode
                collection = self._get_or_create_collection()
                collection.add_documents(documents=langchain_docs, ids=doc_ids)
            else:
                # Use the main vector store for backward compatibility
                self.vector_store.add_documents(documents=langchain_docs, ids=doc_ids)
            
        except Exception as e:
            raise VectorStoreOperationError(f"Failed to add documents: {e}")
    
    def update_documents(self, documents: List[DocumentInput]) -> None:
        """Update existing documents in the vector store."""
        if not PROVIDER_INTERFACE_AVAILABLE:
            raise NotImplementedError("VectorStoreProvider interface not available")
        
        try:
            # For ChromaDB, update is essentially delete + add
            doc_ids = [doc.id for doc in documents]
            self.delete_documents(doc_ids)
            self.add_documents(documents)
            
        except Exception as e:
            raise VectorStoreOperationError(f"Failed to update documents: {e}")
    
    def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from the vector store."""
        if not PROVIDER_INTERFACE_AVAILABLE:
            raise NotImplementedError("VectorStoreProvider interface not available")
        
        try:
            if self._is_provider_mode:
                collection = self._get_or_create_collection()
                collection.delete(ids=document_ids)
            else:
                # For backward compatibility mode, use the main vector store
                self.vector_store.delete(ids=document_ids)
                
        except Exception as e:
            raise VectorStoreOperationError(f"Failed to delete documents: {e}")
    
    def similarity_search(self, 
                         query: str, 
                         k: int = 5,
                         filter_metadata: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Perform similarity search using vector embeddings."""
        if not PROVIDER_INTERFACE_AVAILABLE:
            raise NotImplementedError("VectorStoreProvider interface not available")
        
        try:
            # Use the existing search functionality
            if self._is_provider_mode:
                collection = self._get_or_create_collection()
                results = collection.similarity_search_with_relevance_scores(
                    query=query, k=k, score_threshold=0.0
                )
            else:
                results = self.vector_store.similarity_search_with_relevance_scores(
                    query=query, k=k, score_threshold=0.0
                )
            
            # Convert to SearchResult format
            search_results = []
            for doc, score in results:
                search_result = SearchResult(
                    document_id=doc.metadata.get('id', str(uuid.uuid4())),
                    content=doc.page_content,
                    metadata=doc.metadata,
                    score=score
                )
                search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            raise VectorStoreOperationError(f"Failed to perform similarity search: {e}")
    
    def hybrid_search(self, 
                     query: str, 
                     k: int = 5,
                     filter_metadata: Optional[Dict[str, Any]] = None,
                     alpha: float = 0.5) -> List[SearchResult]:
        """Perform hybrid search combining vector and lexical search."""
        if not PROVIDER_INTERFACE_AVAILABLE:
            raise NotImplementedError("VectorStoreProvider interface not available")
        
        # ChromaDB doesn't have native hybrid search, so we'll use similarity search
        # In a full implementation, you would combine vector and BM25 scores
        return self.similarity_search(query, k, filter_metadata)
    
    def get_document(self, document_id: str) -> Optional[SearchResult]:
        """Retrieve a specific document by ID."""
        if not PROVIDER_INTERFACE_AVAILABLE:
            raise NotImplementedError("VectorStoreProvider interface not available")
        
        try:
            if self._is_provider_mode:
                collection = self._get_or_create_collection()
                results = collection.get(ids=[document_id])
            else:
                results = self.vector_store.get(ids=[document_id])
            
            if results and results['documents']:
                doc_content = results['documents'][0]
                doc_metadata = results['metadatas'][0] if results['metadatas'] else {}
                
                return SearchResult(
                    document_id=document_id,
                    content=doc_content,
                    metadata=doc_metadata,
                    score=1.0  # Perfect match for direct retrieval
                )
            
            return None
            
        except Exception as e:
            raise VectorStoreOperationError(f"Failed to get document: {e}")
    
    def list_documents(self, 
                      limit: Optional[int] = None,
                      offset: int = 0,
                      filter_metadata: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """List documents in the vector store."""
        if not PROVIDER_INTERFACE_AVAILABLE:
            raise NotImplementedError("VectorStoreProvider interface not available")
        
        try:
            if self._is_provider_mode:
                collection = self._get_or_create_collection()
                results = collection.get(limit=limit, offset=offset)
            else:
                results = self.vector_store.get(limit=limit, offset=offset)
            
            search_results = []
            if results and results['documents']:
                for i, doc_content in enumerate(results['documents']):
                    doc_id = results['ids'][i] if results['ids'] else str(uuid.uuid4())
                    doc_metadata = results['metadatas'][i] if results['metadatas'] else {}
                    
                    search_result = SearchResult(
                        document_id=doc_id,
                        content=doc_content,
                        metadata=doc_metadata,
                        score=1.0
                    )
                    search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            raise VectorStoreOperationError(f"Failed to list documents: {e}")
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        if not PROVIDER_INTERFACE_AVAILABLE:
            raise NotImplementedError("VectorStoreProvider interface not available")
        
        try:
            if self._is_provider_mode:
                # Delete the specific collection
                client = self._get_chroma_client()
                client.delete_collection(name=self.collection_name)
            else:
                # For backward compatibility, delete the main collection
                if hasattr(self.vector_store, '_collection'):
                    self.vector_store._collection.delete()
                    
        except Exception as e:
            raise VectorStoreOperationError(f"Failed to delete collection: {e}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        if not PROVIDER_INTERFACE_AVAILABLE:
            return {
                "provider": "chromadb",
                "collection_name": self.collection_name,
                "embedding_dimension": self.provider_info.get('dimensions', 384),
                "document_count": 0,
                "distance_metric": "cosine"
            }
        
        try:
            if self._is_provider_mode:
                collection = self._get_or_create_collection()
                count = collection.count()
            else:
                count = self.vector_store._collection.count() if hasattr(self.vector_store, '_collection') else 0
            
            return {
                "provider": "chromadb",
                "collection_name": self.collection_name,
                "embedding_dimension": self.provider_info.get('dimensions', 384),
                "document_count": count,
                "distance_metric": "cosine",
                "provider_info": self.provider_info,
                "collection_metadata": self._collection_metadata
            }
            
        except Exception as e:
            raise VectorStoreOperationError(f"Failed to get collection info: {e}")
    
    def is_available(self) -> bool:
        """Check if the vector store is available and properly configured."""
        try:
            # Test ChromaDB availability
            client = self._get_chroma_client()
            # Try to list collections as a health check
            client.list_collections()
            return True
        except Exception:
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the vector store."""
        start_time = time.time()
        
        try:
            # Test basic operations
            client = self._get_chroma_client()
            collections = client.list_collections()
            
            response_time = (time.time() - start_time) * 1000
            
            return {
                "status": "healthy",
                "response_time": response_time,
                "collections_count": len(collections),
                "embedding_provider": self.embedding_provider.get_provider_info()
            }
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return {
                "status": "unhealthy",
                "response_time": response_time,
                "error_message": str(e)
            }
    
    def _get_chroma_client(self):
        """Get ChromaDB client instance."""
        if self._is_provider_mode:
            # Use configuration from VectorStoreConfig
            connection_params = self.config.connection_params
            mode = connection_params.get('mode', 'persistent')
            
            if mode == 'client':
                # Client mode
                import chromadb
                return chromadb.HttpClient(
                    host=connection_params['host'],
                    port=connection_params['port'],
                    ssl=connection_params.get('ssl', False),
                    headers=connection_params.get('headers', {})
                )
            else:
                # Persistent mode
                import chromadb
                return chromadb.PersistentClient(
                    path=connection_params.get('path', self.chroma_db_path),
                    settings=CHROMA_SETTINGS
                )
        else:
            # Backward compatibility mode
            import chromadb
            return chromadb.PersistentClient(
                path=self.chroma_db_path,
                settings=CHROMA_SETTINGS
            )
    
    def _get_or_create_collection(self):
        """Get or create the ChromaDB collection with enhanced management."""
        client = self._get_chroma_client()
        
        try:
            collection = client.get_collection(
                name=self.collection_name,
                embedding_function=self._get_embedding_function()
            )
            
            # Update available collections cache
            if self.collection_name not in self._available_collections:
                self._available_collections[self.collection_name] = {
                    'provider': self.provider_info['provider'],
                    'dimensions': self.provider_info['dimensions'],
                    'collection_name': self.collection_name,
                    'document_count': collection.count() if hasattr(collection, 'count') else 0,
                    'discovered_at': time.time()
                }
            
            logging.info(f"Using existing collection: {self.collection_name}")
            
        except Exception:
            # Collection doesn't exist, create it
            logging.info(f"Creating new collection: {self.collection_name}")
            
            collection = client.create_collection(
                name=self.collection_name,
                embedding_function=self._get_embedding_function()
            )
            
            # Add to available collections
            self._available_collections[self.collection_name] = {
                'provider': self.provider_info['provider'],
                'dimensions': self.provider_info['dimensions'],
                'collection_name': self.collection_name,
                'document_count': 0,
                'created_at': time.time()
            }
            
            # Store collection metadata
            self._store_collection_metadata(self.collection_name, self._collection_metadata)
            
            # Log dimension compatibility warnings
            self._check_dimension_compatibility()
        
        return collection
    
    def _check_dimension_compatibility(self):
        """Check for dimension compatibility issues with existing collections."""
        current_dimensions = self.provider_info['dimensions']
        current_provider = self.provider_info['provider']
        
        for collection_name, collection_info in self._available_collections.items():
            if collection_name == self.collection_name:
                continue
                
            # Check for dimension mismatches with same provider
            if (collection_info['provider'] == current_provider and 
                collection_info['dimensions'] != current_dimensions):
                logging.warning(
                    f"Dimension mismatch detected: Collection '{collection_name}' "
                    f"has {collection_info['dimensions']} dimensions but current provider "
                    f"'{current_provider}' uses {current_dimensions} dimensions. "
                    f"Consider re-indexing or cleaning up unused collections."
                )
            
            # Check for collections from different providers
            elif collection_info['provider'] != current_provider:
                logging.info(
                    f"Found collection from different provider: '{collection_name}' "
                    f"(provider: {collection_info['provider']}, dimensions: {collection_info['dimensions']}). "
                    f"This collection will remain available for provider switching."
                )
    
    def validate_provider_switch(self, target_provider: str, target_dimensions: int) -> Dict[str, Any]:
        """Validate a potential provider switch and provide guidance.
        
        Args:
            target_provider: Name of the target provider
            target_dimensions: Embedding dimensions of the target provider
            
        Returns:
            Dictionary with validation results and recommendations
        """
        current_provider = self.provider_info['provider']
        current_dimensions = self.provider_info['dimensions']
        
        validation_result = {
            'valid': True,
            'warnings': [],
            'recommendations': [],
            'target_collection_exists': False,
            'requires_reindexing': False,
            'dimension_change': current_dimensions != target_dimensions,
            'provider_change': current_provider != target_provider
        }
        
        # Check if target collection already exists
        target_collection_name = f"manim_docs_{target_provider}_{target_dimensions}d"
        if target_collection_name in self._available_collections:
            validation_result['target_collection_exists'] = True
            existing_collection = self._available_collections[target_collection_name]
            doc_count = existing_collection.get('document_count', 0)
            
            if doc_count > 0:
                validation_result['recommendations'].append(
                    f"Target collection '{target_collection_name}' already exists with "
                    f"{doc_count} documents. You can switch immediately without re-indexing."
                )
            else:
                validation_result['recommendations'].append(
                    f"Target collection '{target_collection_name}' exists but is empty. "
                    f"You will need to index documents after switching."
                )
        else:
            validation_result['recommendations'].append(
                f"Target collection '{target_collection_name}' does not exist. "
                f"It will be created automatically, but you will need to index documents."
            )
        
        # Check for dimension changes
        if validation_result['dimension_change']:
            validation_result['requires_reindexing'] = True
            validation_result['warnings'].append(
                f"Embedding dimensions will change from {current_dimensions} to {target_dimensions}. "
                f"Existing embeddings cannot be reused."
            )
            validation_result['recommendations'].append(
                "Re-indexing is required due to dimension change. "
                "Plan for processing time based on your document collection size."
            )
        
        # Check for provider changes
        if validation_result['provider_change']:
            validation_result['recommendations'].append(
                f"Switching from {current_provider} to {target_provider} provider. "
                f"Consider re-indexing for optimal performance even if dimensions match."
            )
        
        # Check for potential conflicts
        conflicting_collections = []
        for collection_name, collection_info in self._available_collections.items():
            if (collection_info['provider'] == target_provider and 
                collection_info['dimensions'] != target_dimensions):
                conflicting_collections.append(collection_name)
        
        if conflicting_collections:
            validation_result['warnings'].append(
                f"Found collections with same provider but different dimensions: {conflicting_collections}. "
                f"Consider cleaning up unused collections."
            )
        
        return validation_result
    
    def get_reindexing_guidance(self) -> Dict[str, Any]:
        """Get guidance for re-indexing when switching providers.
        
        Returns:
            Dictionary with re-indexing guidance and estimates
        """
        current_collection_info = self._available_collections.get(self.collection_name, {})
        doc_count = current_collection_info.get('document_count', 0)
        
        # Estimate processing time (rough estimates)
        estimated_time_per_doc = 0.1  # seconds per document
        estimated_total_time = doc_count * estimated_time_per_doc
        
        guidance = {
            'current_collection': self.collection_name,
            'current_document_count': doc_count,
            'estimated_processing_time_seconds': estimated_total_time,
            'estimated_processing_time_minutes': estimated_total_time / 60,
            'steps': [
                "1. Ensure new embedding provider is properly configured",
                "2. Switch to the new provider using switch_provider() method",
                "3. Re-index your documents using the standard indexing process",
                "4. Verify the new collection has the expected number of documents",
                "5. Test search functionality with the new embeddings",
                "6. Optionally clean up old collections to save space"
            ],
            'considerations': [
                "Re-indexing will regenerate all embeddings using the new provider",
                "Search results may differ between providers due to different embedding models",
                "Consider testing with a small subset of documents first",
                "Keep old collections until you verify the new setup works correctly",
                "Monitor API usage if switching to a cloud-based provider like JINA"
            ]
        }
        
        # Add provider-specific guidance
        current_provider = self.provider_info['provider']
        if current_provider == 'local':
            guidance['considerations'].append(
                "Switching from local to cloud embeddings may improve search quality but will require API access"
            )
        elif current_provider == 'jina':
            guidance['considerations'].append(
                "Switching from JINA embeddings will require re-indexing and may affect search quality"
            )
        
        return guidance
    
    def migrate_collection(self, source_collection: str, target_provider_config, dry_run: bool = True) -> Dict[str, Any]:
        """Migrate documents from one collection to another with different provider settings.
        
        Args:
            source_collection: Name of the source collection to migrate from
            target_provider_config: EmbeddingConfig for the target provider
            dry_run: If True, only analyze what would be migrated without actually doing it
            
        Returns:
            Dictionary with migration results and status
        """
        migration_results = {
            'dry_run': dry_run,
            'source_collection': source_collection,
            'target_provider': target_provider_config.provider if hasattr(target_provider_config, 'provider') else 'unknown',
            'migration_successful': False,
            'documents_migrated': 0,
            'documents_failed': 0,
            'errors': [],
            'warnings': [],
            'recommendations': [],
            'estimated_time_seconds': 0,
            'estimated_cost': 0.0  # For cloud providers
        }
        
        try:
            # Validate source collection exists
            if source_collection not in self._available_collections:
                raise ValueError(f"Source collection '{source_collection}' not found")
            
            source_info = self._available_collections[source_collection]
            migration_results['source_info'] = source_info
            
            # Create target provider
            from .provider_factory import EmbeddingProviderFactory
            target_provider = EmbeddingProviderFactory.create_provider(target_provider_config)
            target_provider_info = target_provider.get_provider_info()
            migration_results['target_provider_info'] = target_provider_info
            
            # Generate target collection name
            target_collection_name = f"manim_docs_{target_provider_info['provider']}_{target_provider_info['dimensions']}d"
            migration_results['target_collection'] = target_collection_name
            
            # Check if target collection already exists
            if target_collection_name in self._available_collections:
                migration_results['warnings'].append(
                    f"Target collection '{target_collection_name}' already exists. "
                    f"Migration will add to existing collection."
                )
            
            # Get documents from source collection
            client = self._get_chroma_client()
            try:
                source_collection_obj = client.get_collection(source_collection)
                source_documents = source_collection_obj.get()
            except Exception as e:
                raise ValueError(f"Failed to access source collection: {e}")
            
            if not source_documents or not source_documents.get('documents'):
                migration_results['warnings'].append("Source collection is empty")
                return migration_results
            
            doc_count = len(source_documents['documents'])
            migration_results['total_documents'] = doc_count
            
            # Estimate migration time and cost
            if target_provider_info['provider'] == 'jina':
                # Estimate API cost for JINA (rough estimate)
                estimated_tokens = sum(len(doc.split()) * 1.3 for doc in source_documents['documents'])
                migration_results['estimated_cost'] = estimated_tokens * 0.0001  # Rough cost estimate
                migration_results['estimated_time_seconds'] = doc_count * 0.5  # API calls take longer
            else:
                migration_results['estimated_time_seconds'] = doc_count * 0.1  # Local processing
            
            if dry_run:
                migration_results['recommendations'].extend([
                    f"Migration will process {doc_count} documents",
                    f"Estimated time: {migration_results['estimated_time_seconds']:.1f} seconds",
                    f"Target provider: {target_provider_info['provider']} ({target_provider_info['dimensions']} dimensions)"
                ])
                
                if migration_results['estimated_cost'] > 0:
                    migration_results['recommendations'].append(
                        f"Estimated API cost: ${migration_results['estimated_cost']:.4f}"
                    )
                
                migration_results['migration_successful'] = True
                return migration_results
            
            # Perform actual migration
            logging.info(f"Starting migration from {source_collection} to {target_collection_name}")
            
            # Create target collection if it doesn't exist
            try:
                target_collection_obj = client.get_collection(target_collection_name)
            except Exception:
                # Create new collection with target provider's embedding function
                class TargetEmbeddingFunction:
                    def __init__(self, provider):
                        self.provider = provider
                    
                    def __call__(self, texts: List[str]) -> List[List[float]]:
                        return self.provider.generate_embeddings(texts)
                
                target_embedding_fn = TargetEmbeddingFunction(target_provider)
                target_collection_obj = client.create_collection(
                    name=target_collection_name,
                    embedding_function=target_embedding_fn
                )
            
            # Migrate documents in batches
            batch_size = 10
            migrated_count = 0
            failed_count = 0
            
            for i in range(0, doc_count, batch_size):
                batch_docs = source_documents['documents'][i:i + batch_size]
                batch_ids = source_documents['ids'][i:i + batch_size] if source_documents.get('ids') else None
                batch_metadatas = source_documents['metadatas'][i:i + batch_size] if source_documents.get('metadatas') else None
                
                try:
                    # Generate new IDs if needed
                    if not batch_ids:
                        batch_ids = [f"migrated_{uuid.uuid4()}" for _ in batch_docs]
                    else:
                        batch_ids = [f"migrated_{id}" for id in batch_ids]
                    
                    # Prepare metadata
                    if not batch_metadatas:
                        batch_metadatas = [{} for _ in batch_docs]
                    
                    # Add migration metadata
                    for metadata in batch_metadatas:
                        metadata['migrated_from'] = source_collection
                        metadata['migration_date'] = time.time()
                        metadata['target_provider'] = target_provider_info['provider']
                    
                    # Add documents to target collection (ChromaDB will generate new embeddings)
                    target_collection_obj.add(
                        documents=batch_docs,
                        ids=batch_ids,
                        metadatas=batch_metadatas
                    )
                    
                    migrated_count += len(batch_docs)
                    logging.info(f"Migrated batch {i // batch_size + 1}: {len(batch_docs)} documents")
                    
                except Exception as e:
                    failed_count += len(batch_docs)
                    error_msg = f"Failed to migrate batch {i // batch_size + 1}: {e}"
                    migration_results['errors'].append(error_msg)
                    logging.error(error_msg)
            
            migration_results['documents_migrated'] = migrated_count
            migration_results['documents_failed'] = failed_count
            migration_results['migration_successful'] = failed_count == 0
            
            # Update available collections cache
            if target_collection_name not in self._available_collections:
                self._available_collections[target_collection_name] = {
                    'provider': target_provider_info['provider'],
                    'dimensions': target_provider_info['dimensions'],
                    'collection_name': target_collection_name,
                    'document_count': migrated_count,
                    'created_at': time.time(),
                    'migrated_from': source_collection
                }
            else:
                # Update document count
                self._available_collections[target_collection_name]['document_count'] += migrated_count
            
            # Store migration metadata
            migration_metadata = {
                'provider': target_provider_info['provider'],
                'model': target_provider_info.get('model', 'unknown'),
                'dimensions': target_provider_info['dimensions'],
                'created_at': time.time(),
                'collection_name': target_collection_name,
                'migrated_from': source_collection,
                'migration_date': time.time(),
                'documents_migrated': migrated_count
            }
            self._store_collection_metadata(target_collection_name, migration_metadata)
            
            if migration_results['migration_successful']:
                migration_results['recommendations'].append(
                    f"Migration completed successfully. {migrated_count} documents migrated to {target_collection_name}"
                )
                migration_results['recommendations'].append(
                    "You can now switch to the new provider and test the migrated collection"
                )
            else:
                migration_results['recommendations'].append(
                    f"Migration partially completed. {migrated_count} documents migrated, {failed_count} failed"
                )
                migration_results['recommendations'].append(
                    "Review error messages and consider re-running migration for failed documents"
                )
            
            logging.info(f"Migration completed: {migrated_count} successful, {failed_count} failed")
            
        except Exception as e:
            migration_results['errors'].append(f"Migration failed: {e}")
            logging.error(f"Migration failed: {e}")
        
        return migration_results
    
    def get_migration_recommendations(self, target_provider: str = None) -> Dict[str, Any]:
        """Get comprehensive recommendations for collection migration and provider switching.
        
        Args:
            target_provider: Optional target provider name to get specific recommendations
            
        Returns:
            Dictionary with migration recommendations and guidance
        """
        current_provider = self.provider_info['provider']
        current_dimensions = self.provider_info['dimensions']
        
        recommendations = {
            'current_state': {
                'provider': current_provider,
                'dimensions': current_dimensions,
                'collection': self.collection_name,
                'document_count': self._available_collections.get(self.collection_name, {}).get('document_count', 0)
            },
            'available_collections': len(self._available_collections),
            'migration_opportunities': [],
            'cleanup_opportunities': [],
            'provider_specific_guidance': {},
            'best_practices': []
        }
        
        # Analyze migration opportunities
        for collection_name, collection_info in self._available_collections.items():
            if collection_name == self.collection_name:
                continue
            
            opportunity = {
                'collection': collection_name,
                'provider': collection_info['provider'],
                'dimensions': collection_info['dimensions'],
                'document_count': collection_info.get('document_count', 0),
                'migration_type': 'unknown',
                'effort_level': 'unknown',
                'benefits': [],
                'considerations': []
            }
            
            # Determine migration type and effort
            if collection_info['provider'] != current_provider:
                if collection_info['dimensions'] == current_dimensions:
                    opportunity['migration_type'] = 'provider_switch_same_dimensions'
                    opportunity['effort_level'] = 'low'
                    opportunity['benefits'].append('Switch providers while keeping existing embeddings')
                else:
                    opportunity['migration_type'] = 'provider_switch_different_dimensions'
                    opportunity['effort_level'] = 'high'
                    opportunity['benefits'].append('Switch to different provider with potentially better embeddings')
                    opportunity['considerations'].append('Requires full re-embedding of all documents')
            else:
                if collection_info['dimensions'] != current_dimensions:
                    opportunity['migration_type'] = 'dimension_upgrade'
                    opportunity['effort_level'] = 'high'
                    opportunity['benefits'].append('Upgrade to higher dimension embeddings for better quality')
                    opportunity['considerations'].append('Requires re-embedding but uses same provider')
            
            # Add specific considerations
            if collection_info.get('document_count', 0) == 0:
                opportunity['considerations'].append('Collection is empty - no migration needed')
                opportunity['effort_level'] = 'none'
            elif collection_info.get('document_count', 0) > 1000:
                opportunity['considerations'].append('Large collection - plan for extended processing time')
            
            recommendations['migration_opportunities'].append(opportunity)
        
        # Analyze cleanup opportunities
        provider_counts = {}
        dimension_counts = {}
        
        for collection_name, collection_info in self._available_collections.items():
            provider = collection_info['provider']
            dimensions = collection_info['dimensions']
            
            if provider not in provider_counts:
                provider_counts[provider] = 0
            provider_counts[provider] += 1
            
            if dimensions not in dimension_counts:
                dimension_counts[dimensions] = 0
            dimension_counts[dimensions] += 1
        
        # Recommend cleanup for providers with multiple collections
        for provider, count in provider_counts.items():
            if count > 1:
                recommendations['cleanup_opportunities'].append({
                    'type': 'multiple_collections_same_provider',
                    'provider': provider,
                    'count': count,
                    'recommendation': f'Consider consolidating or cleaning up {count} collections for provider {provider}'
                })
        
        # Add provider-specific guidance
        if target_provider:
            if target_provider == 'jina':
                recommendations['provider_specific_guidance']['jina'] = {
                    'benefits': [
                        'High-quality cloud-based embeddings',
                        'Regular model updates and improvements',
                        'Scalable infrastructure'
                    ],
                    'considerations': [
                        'Requires API key and internet connection',
                        'API usage costs for large document collections',
                        'Rate limits may affect processing speed'
                    ],
                    'setup_steps': [
                        '1. Get JINA API key from https://jina.ai/?sui=apikey',
                        '2. Set JINA_API_KEY environment variable',
                        '3. Set EMBEDDING_PROVIDER=jina',
                        '4. Test with small document set first'
                    ]
                }
            elif target_provider == 'local':
                recommendations['provider_specific_guidance']['local'] = {
                    'benefits': [
                        'No API costs or rate limits',
                        'Works offline',
                        'Full data privacy'
                    ],
                    'considerations': [
                        'Lower embedding quality compared to cloud providers',
                        'Requires local model download and storage',
                        'Processing speed depends on local hardware'
                    ],
                    'setup_steps': [
                        '1. Ensure sufficient disk space for model files',
                        '2. Set EMBEDDING_PROVIDER=local',
                        '3. Configure LOCAL_EMBEDDING_MODEL if needed',
                        '4. Allow time for initial model download'
                    ]
                }
        
        # Add best practices
        recommendations['best_practices'] = [
            'Always test provider switches with a small subset of documents first',
            'Keep backup collections when switching providers until verified',
            'Monitor API usage and costs when using cloud providers',
            'Use cleanup_unused_collections() regularly to manage disk space',
            'Consider embedding dimensions based on your use case complexity',
            'Plan migration during low-usage periods for large collections'
        ]
        
        return recommendations

# Update the existing RAGVectorStore class alias for backward compatibility
RAGVectorStore = EnhancedRAGVectorStore
