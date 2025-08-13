"""
Context7 Documentation Retrieval Functions

This module provides specialized functions for retrieving Manim Community documentation
through the Context7 MCP server, including library ID resolution, documentation retrieval
with topic filtering, and code snippet extraction.

Enhanced with caching and optimization features for improved performance.
"""

import asyncio
import logging
import re
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum

import mcp
from mcp.types import Tool, CallToolRequest, CallToolResult

from .client import MCPClient, MCPConnectionError, MCPServerNotFoundError
from .cache import get_documentation_cache, DocumentationCache

logger = logging.getLogger(__name__)


class DocumentationError(Exception):
    """Exception raised when documentation retrieval fails."""
    pass


class LibraryNotFoundError(DocumentationError):
    """Exception raised when library ID cannot be resolved."""
    pass


@dataclass
class CodeSnippet:
    """Represents a code snippet from documentation."""
    code: str
    language: str
    description: Optional[str] = None
    source_url: Optional[str] = None
    line_numbers: Optional[Tuple[int, int]] = None


@dataclass
class DocumentationSection:
    """Represents a section of documentation."""
    title: str
    content: str
    code_snippets: List[CodeSnippet]
    source_url: Optional[str] = None


@dataclass
class DocumentationResponse:
    """Response from documentation retrieval."""
    library_id: str
    topic: Optional[str]
    sections: List[DocumentationSection]
    total_tokens: int
    retrieved_at: str


class Context7DocsRetriever:
    """
    Context7 documentation retrieval client for Manim Community documentation.
    
    This class provides high-level functions for resolving library IDs,
    retrieving documentation with topic filtering, and extracting code snippets.
    
    Enhanced with caching and optimization features:
    - Documentation caching with TTL management
    - Memory-efficient storage
    - Cache invalidation strategies
    - Performance monitoring
    """
    
    # Manim Community library identifier for Context7
    MANIM_LIBRARY_NAME = "manim"
    MANIM_LIBRARY_ID = "/manimcommunity/manim"
    
    def __init__(
        self, 
        mcp_client: MCPClient,
        enable_caching: bool = True,
        cache_ttl_seconds: float = 1800  # 30 minutes default
    ):
        """
        Initialize Context7 documentation retriever.
        
        Args:
            mcp_client: Initialized MCPClient instance
            enable_caching: Whether to enable documentation caching
            cache_ttl_seconds: Default TTL for cached documentation
        """
        self.mcp_client = mcp_client
        self.enable_caching = enable_caching
        self.cache_ttl_seconds = cache_ttl_seconds
        
        # Legacy library cache for backward compatibility
        self._library_cache: Dict[str, str] = {}
        
        # Get global documentation cache
        self._doc_cache = get_documentation_cache() if enable_caching else None
        
        # Performance tracking
        self._retrieval_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_retrievals': 0,
            'total_time': 0.0
        }
        
        logger.info(f"Context7DocsRetriever initialized with caching={'enabled' if enable_caching else 'disabled'}")
        
    async def resolve_manim_library_id(self) -> str:
        """
        Resolve Manim Community library ID for Context7.
        
        Enhanced with caching for improved performance.
        
        Returns:
            Context7-compatible library ID for Manim Community
            
        Raises:
            LibraryNotFoundError: If library ID cannot be resolved
            MCPConnectionError: If Context7 server is not available
        """
        start_time = time.time()
        
        try:
            # Check documentation cache first if enabled
            if self._doc_cache:
                cached_id = self._doc_cache.get(f"library_id:{self.MANIM_LIBRARY_NAME}")
                if cached_id:
                    self._retrieval_stats['cache_hits'] += 1
                    logger.debug(f"Using cached library ID for {self.MANIM_LIBRARY_NAME}: {cached_id}")
                    return cached_id
            
            # Check legacy cache
            if self.MANIM_LIBRARY_NAME in self._library_cache:
                library_id = self._library_cache[self.MANIM_LIBRARY_NAME]
                logger.debug(f"Using legacy cached library ID for {self.MANIM_LIBRARY_NAME}: {library_id}")
                return library_id
            
            # Get Context7 session
            session = await self.mcp_client.get_context7_session()
            
            # Call resolve_library_id tool
            resolve_request = CallToolRequest(
                name="mcp_context7_resolve_library_id",
                arguments={"libraryName": self.MANIM_LIBRARY_NAME}
            )
            
            logger.info(f"Resolving library ID for: {self.MANIM_LIBRARY_NAME}")
            result = await session.call_tool(resolve_request)
            
            if not result.isError and result.content:
                # Parse the result to extract library ID
                library_id = self._parse_library_id_response(result.content)
                
                # Cache the result in both caches
                self._library_cache[self.MANIM_LIBRARY_NAME] = library_id
                
                if self._doc_cache:
                    # Cache with longer TTL since library IDs rarely change
                    self._doc_cache.put(
                        f"library_id:{self.MANIM_LIBRARY_NAME}", 
                        library_id, 
                        ttl_seconds=86400  # 24 hours
                    )
                
                self._retrieval_stats['cache_misses'] += 1
                self._retrieval_stats['total_retrievals'] += 1
                self._retrieval_stats['total_time'] += time.time() - start_time
                
                logger.info(f"Successfully resolved library ID: {library_id}")
                return library_id
            else:
                error_msg = f"Failed to resolve library ID for {self.MANIM_LIBRARY_NAME}"
                if result.isError:
                    error_msg += f": {result.content}"
                raise LibraryNotFoundError(error_msg)
                
        except MCPServerNotFoundError:
            raise MCPConnectionError("Context7 server not available for library ID resolution")
        except Exception as e:
            logger.error(f"Error resolving library ID for {self.MANIM_LIBRARY_NAME}: {e}")
            raise LibraryNotFoundError(f"Failed to resolve library ID: {e}")
    
    def _parse_library_id_response(self, content: List[Any]) -> str:
        """
        Parse library ID from Context7 response.
        
        Args:
            content: Response content from Context7
            
        Returns:
            Extracted library ID
            
        Raises:
            LibraryNotFoundError: If library ID cannot be extracted
        """
        try:
            # Handle different response formats
            if isinstance(content, list) and len(content) > 0:
                first_item = content[0]
                
                # Check if it's a text response
                if hasattr(first_item, 'text'):
                    text_content = first_item.text
                    
                    # Look for library ID patterns in the text
                    # Context7 typically returns library IDs in format /org/project
                    library_id_pattern = r'/[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+'
                    matches = re.findall(library_id_pattern, text_content)
                    
                    if matches:
                        # Prefer exact match for Manim Community
                        for match in matches:
                            if 'manim' in match.lower():
                                return match
                        # Return first match if no exact match
                        return matches[0]
                    
                    # If no pattern match, check if the text itself is a library ID
                    if text_content.startswith('/') and '/' in text_content[1:]:
                        return text_content.strip()
                
                # Handle direct string response
                elif isinstance(first_item, str):
                    if first_item.startswith('/') and '/' in first_item[1:]:
                        return first_item.strip()
            
            # Fallback to known Manim Community library ID
            logger.warning("Could not parse library ID from response, using known Manim ID")
            return self.MANIM_LIBRARY_ID
            
        except Exception as e:
            logger.error(f"Error parsing library ID response: {e}")
            # Fallback to known Manim Community library ID
            return self.MANIM_LIBRARY_ID
    
    async def retrieve_documentation(
        self,
        topic: Optional[str] = None,
        max_tokens: int = 10000,
        library_id: Optional[str] = None
    ) -> DocumentationResponse:
        """
        Retrieve Manim Community documentation with optional topic filtering.
        
        Enhanced with intelligent caching and performance optimization.
        
        Args:
            topic: Specific topic to focus documentation on (e.g., 'animations', 'scenes')
            max_tokens: Maximum number of tokens to retrieve
            library_id: Specific library ID to use (auto-resolved if None)
            
        Returns:
            DocumentationResponse with retrieved documentation
            
        Raises:
            DocumentationError: If documentation retrieval fails
            MCPConnectionError: If Context7 server is not available
        """
        start_time = time.time()
        
        try:
            # Resolve library ID if not provided
            if library_id is None:
                library_id = await self.resolve_manim_library_id()
            
            # Check cache first if enabled
            if self._doc_cache:
                cached_doc = self._doc_cache.get_documentation(library_id, topic)
                if cached_doc:
                    self._retrieval_stats['cache_hits'] += 1
                    logger.debug(f"Using cached documentation for {library_id}, topic: {topic}")
                    return cached_doc
            
            # Get Context7 session
            session = await self.mcp_client.get_context7_session()
            
            # Prepare documentation retrieval request
            get_docs_args = {
                "context7CompatibleLibraryID": library_id,
                "tokens": max_tokens
            }
            
            if topic:
                get_docs_args["topic"] = topic
            
            get_docs_request = CallToolRequest(
                name="mcp_context7_get_library_docs",
                arguments=get_docs_args
            )
            
            logger.info(f"Retrieving documentation for library: {library_id}, topic: {topic}")
            result = await session.call_tool(get_docs_request)
            
            if not result.isError and result.content:
                # Parse the documentation response
                doc_response = self._parse_documentation_response(
                    result.content, library_id, topic, max_tokens
                )
                
                # Cache the result if caching is enabled
                if self._doc_cache:
                    self._doc_cache.put_documentation(
                        library_id, 
                        topic, 
                        doc_response, 
                        ttl_seconds=self.cache_ttl_seconds
                    )
                
                # Update stats
                self._retrieval_stats['cache_misses'] += 1
                self._retrieval_stats['total_retrievals'] += 1
                self._retrieval_stats['total_time'] += time.time() - start_time
                
                logger.info(f"Successfully retrieved {len(doc_response.sections)} documentation sections")
                return doc_response
            else:
                error_msg = f"Failed to retrieve documentation for {library_id}"
                if result.isError:
                    error_msg += f": {result.content}"
                raise DocumentationError(error_msg)
                
        except MCPServerNotFoundError:
            raise MCPConnectionError("Context7 server not available for documentation retrieval")
        except Exception as e:
            logger.error(f"Error retrieving documentation: {e}")
            raise DocumentationError(f"Failed to retrieve documentation: {e}")
    
    def _parse_documentation_response(
        self,
        content: List[Any],
        library_id: str,
        topic: Optional[str],
        max_tokens: int
    ) -> DocumentationResponse:
        """
        Parse documentation response from Context7.
        
        Args:
            content: Response content from Context7
            library_id: Library ID used for retrieval
            topic: Topic filter used
            max_tokens: Token limit used
            
        Returns:
            Parsed DocumentationResponse
        """
        sections = []
        
        try:
            for item in content:
                if hasattr(item, 'text'):
                    text_content = item.text
                    
                    # Extract sections from the documentation text
                    doc_sections = self._extract_documentation_sections(text_content)
                    sections.extend(doc_sections)
        
        except Exception as e:
            logger.error(f"Error parsing documentation response: {e}")
            # Create a single section with raw content as fallback
            if content and hasattr(content[0], 'text'):
                sections = [DocumentationSection(
                    title="Documentation",
                    content=content[0].text,
                    code_snippets=[]
                )]
        
        return DocumentationResponse(
            library_id=library_id,
            topic=topic,
            sections=sections,
            total_tokens=max_tokens,  # Approximate, actual count would need token counting
            retrieved_at=asyncio.get_event_loop().time().__str__()
        )
    
    def _extract_documentation_sections(self, text_content: str) -> List[DocumentationSection]:
        """
        Extract structured sections from documentation text.
        
        Args:
            text_content: Raw documentation text
            
        Returns:
            List of DocumentationSection objects
        """
        sections = []
        
        try:
            # Split content by common section markers
            section_patterns = [
                r'^#{1,3}\s+(.+)$',  # Markdown headers
                r'^([A-Z][A-Za-z\s]+):$',  # Title: format
                r'^([A-Z][A-Za-z\s]+)\n=+$',  # Underlined titles
            ]
            
            # Simple approach: split by double newlines and treat as sections
            raw_sections = text_content.split('\n\n')
            
            for i, raw_section in enumerate(raw_sections):
                if not raw_section.strip():
                    continue
                
                # Extract title from first line or use generic title
                lines = raw_section.strip().split('\n')
                title = f"Section {i+1}"
                content = raw_section.strip()
                
                # Try to identify title from first line
                first_line = lines[0].strip()
                if first_line and (first_line.startswith('#') or first_line.endswith(':')):
                    title = first_line.lstrip('#').rstrip(':').strip()
                    content = '\n'.join(lines[1:]).strip() if len(lines) > 1 else ""
                
                # Extract code snippets from this section
                code_snippets = self.extract_code_snippets(raw_section)
                
                sections.append(DocumentationSection(
                    title=title,
                    content=content,
                    code_snippets=code_snippets
                ))
        
        except Exception as e:
            logger.error(f"Error extracting documentation sections: {e}")
            # Fallback: create single section
            sections = [DocumentationSection(
                title="Documentation",
                content=text_content,
                code_snippets=self.extract_code_snippets(text_content)
            )]
        
        return sections
    
    def extract_code_snippets(self, text: str) -> List[CodeSnippet]:
        """
        Extract and format code snippets from documentation text.
        
        Args:
            text: Documentation text containing code snippets
            
        Returns:
            List of extracted and formatted CodeSnippet objects
        """
        snippets = []
        
        try:
            # Pattern for fenced code blocks (```language)
            fenced_pattern = r'```(\w+)?\n(.*?)\n```'
            fenced_matches = re.findall(fenced_pattern, text, re.DOTALL)
            
            for language, code in fenced_matches:
                language = language or 'python'  # Default to Python for Manim
                snippets.append(CodeSnippet(
                    code=code.strip(),
                    language=language,
                    description=None
                ))
            
            # Pattern for indented code blocks (4+ spaces)
            indented_pattern = r'\n((?:    .+\n?)+)'
            indented_matches = re.findall(indented_pattern, text)
            
            for code_block in indented_matches:
                # Clean up indentation
                lines = code_block.split('\n')
                cleaned_lines = [line[4:] if line.startswith('    ') else line for line in lines]
                cleaned_code = '\n'.join(cleaned_lines).strip()
                
                if cleaned_code and not any(snippet.code == cleaned_code for snippet in snippets):
                    snippets.append(CodeSnippet(
                        code=cleaned_code,
                        language='python',  # Assume Python for Manim
                        description=None
                    ))
            
            # Pattern for inline code with context
            inline_pattern = r'`([^`]+)`'
            inline_matches = re.findall(inline_pattern, text)
            
            for inline_code in inline_matches:
                # Only include substantial inline code (more than just variable names)
                if len(inline_code) > 10 and ('(' in inline_code or '.' in inline_code):
                    if not any(snippet.code == inline_code for snippet in snippets):
                        snippets.append(CodeSnippet(
                            code=inline_code,
                            language='python',
                            description="Inline code example"
                        ))
        
        except Exception as e:
            logger.error(f"Error extracting code snippets: {e}")
        
        return snippets
    
    def format_code_snippets(self, snippets: List[CodeSnippet], format_type: str = "markdown") -> str:
        """
        Format code snippets for display or further processing.
        
        Args:
            snippets: List of CodeSnippet objects to format
            format_type: Format type ('markdown', 'plain', 'html')
            
        Returns:
            Formatted string containing all code snippets
        """
        if not snippets:
            return ""
        
        formatted_parts = []
        
        try:
            for i, snippet in enumerate(snippets, 1):
                if format_type == "markdown":
                    formatted_part = f"### Code Example {i}\n\n"
                    if snippet.description:
                        formatted_part += f"{snippet.description}\n\n"
                    formatted_part += f"```{snippet.language}\n{snippet.code}\n```\n"
                
                elif format_type == "plain":
                    formatted_part = f"Code Example {i}:\n"
                    if snippet.description:
                        formatted_part += f"{snippet.description}\n"
                    formatted_part += f"{snippet.code}\n"
                    formatted_part += "-" * 40 + "\n"
                
                elif format_type == "html":
                    formatted_part = f"<div class='code-example'>\n"
                    formatted_part += f"<h4>Code Example {i}</h4>\n"
                    if snippet.description:
                        formatted_part += f"<p>{snippet.description}</p>\n"
                    formatted_part += f"<pre><code class='language-{snippet.language}'>{snippet.code}</code></pre>\n"
                    formatted_part += "</div>\n"
                
                else:
                    # Default to markdown
                    formatted_part = f"```{snippet.language}\n{snippet.code}\n```\n"
                
                formatted_parts.append(formatted_part)
        
        except Exception as e:
            logger.error(f"Error formatting code snippets: {e}")
            # Fallback: simple concatenation
            formatted_parts = [snippet.code for snippet in snippets]
        
        return "\n".join(formatted_parts)
    
    async def get_manim_documentation(
        self,
        topic: Optional[str] = None,
        include_code_examples: bool = True,
        max_tokens: int = 10000
    ) -> Dict[str, Any]:
        """
        High-level function to get Manim Community documentation with code examples.
        
        Args:
            topic: Specific Manim topic (e.g., 'animations', 'scenes', 'mobjects')
            include_code_examples: Whether to extract and format code examples
            max_tokens: Maximum tokens to retrieve
            
        Returns:
            Dictionary containing documentation and formatted code examples
            
        Raises:
            DocumentationError: If documentation retrieval fails
        """
        try:
            # Retrieve documentation
            doc_response = await self.retrieve_documentation(
                topic=topic,
                max_tokens=max_tokens
            )
            
            # Prepare response
            result = {
                "library_id": doc_response.library_id,
                "topic": doc_response.topic,
                "sections": [],
                "code_examples": [],
                "formatted_code": "",
                "total_sections": len(doc_response.sections),
                "retrieved_at": doc_response.retrieved_at
            }
            
            all_code_snippets = []
            
            for section in doc_response.sections:
                section_data = {
                    "title": section.title,
                    "content": section.content,
                    "code_snippets_count": len(section.code_snippets)
                }
                
                if include_code_examples:
                    section_data["code_snippets"] = [
                        {
                            "code": snippet.code,
                            "language": snippet.language,
                            "description": snippet.description
                        }
                        for snippet in section.code_snippets
                    ]
                    all_code_snippets.extend(section.code_snippets)
                
                result["sections"].append(section_data)
            
            if include_code_examples and all_code_snippets:
                result["code_examples"] = [
                    {
                        "code": snippet.code,
                        "language": snippet.language,
                        "description": snippet.description
                    }
                    for snippet in all_code_snippets
                ]
                result["formatted_code"] = self.format_code_snippets(all_code_snippets)
            
            logger.info(f"Retrieved Manim documentation: {len(doc_response.sections)} sections, "
                       f"{len(all_code_snippets)} code examples")
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting Manim documentation: {e}")
            raise DocumentationError(f"Failed to get Manim documentation: {e}")


# Convenience functions for common operations

async def resolve_manim_library_id(mcp_client: MCPClient) -> str:
    """
    Convenience function to resolve Manim Community library ID.
    
    Args:
        mcp_client: Initialized MCPClient instance
        
    Returns:
        Context7-compatible library ID for Manim Community
    """
    retriever = Context7DocsRetriever(mcp_client)
    return await retriever.resolve_manim_library_id()


async def get_manim_docs(
    mcp_client: MCPClient,
    topic: Optional[str] = None,
    max_tokens: int = 10000
) -> Dict[str, Any]:
    """
    Convenience function to retrieve Manim Community documentation.
    
    Args:
        mcp_client: Initialized MCPClient instance
        topic: Specific topic to focus on
        max_tokens: Maximum tokens to retrieve
        
    Returns:
        Dictionary containing documentation and code examples
    """
    retriever = Context7DocsRetriever(mcp_client)
    return await retriever.get_manim_documentation(
        topic=topic,
        max_tokens=max_tokens
    )


async def extract_manim_code_examples(
    mcp_client: MCPClient,
    topic: Optional[str] = None
) -> List[CodeSnippet]:
    """
    Convenience function to extract code examples from Manim documentation.
    
    Args:
        mcp_client: Initialized MCPClient instance
        topic: Specific topic to focus on
        
    Returns:
        List of CodeSnippet objects
    """
    retriever = Context7DocsRetriever(mcp_client)
    doc_response = await retriever.retrieve_documentation(topic=topic)
    
    all_snippets = []
    for section in doc_response.sections:
        all_snippets.extend(section.code_snippets)
    
    return all_snippets