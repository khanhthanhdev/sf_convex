"""
MCPAgent Implementation for Manim-specific AI Assistance

This module provides MCPAgent functionality that integrates with Context7 MCP server
to create intelligent agents capable of providing accurate Manim guidance based on
current documentation retrieved through the Model Context Protocol.
"""

import asyncio
import logging
import os
import time
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum

# LangChain imports for agent creation
try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    ChatOpenAI = None

try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    ChatAnthropic = None

try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    ChatGroq = None

try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain.tools import Tool
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_core.runnables import RunnableConfig
    LANGCHAIN_CORE_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_CORE_AVAILABLE = False
    raise ImportError(f"LangChain core dependencies not installed: {e}. Install with: pip install langchain langchain-openai")

from .client import MCPClient, MCPConnectionError, MCPServerNotFoundError
from .context7_docs import Context7DocsRetriever, DocumentationError, DocumentationResponse

logger = logging.getLogger(__name__)


class AgentError(Exception):
    """Exception raised when agent operations fail."""
    pass


class LLMProviderError(AgentError):
    """Exception raised when LLM provider configuration fails."""
    pass


@dataclass
class AgentConfig:
    """Configuration for MCPAgent."""
    max_steps: int = 10
    timeout_seconds: int = 300
    system_prompt: Optional[str] = None
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    enable_tool_restrictions: bool = True
    enable_sandboxed_execution: bool = False
    documentation_max_tokens: int = 10000
    auto_retrieve_docs: bool = True


@dataclass
class DocumentationStrategy:
    """Strategy for documentation retrieval based on query analysis."""
    should_prefetch: bool
    topics: List[str]
    priority: str  # 'high', 'medium', 'low'
    reasoning: str
    estimated_tokens: int


@dataclass
class AgentResponse:
    """Response from agent query processing."""
    answer: str
    code_examples: List[str]
    documentation_used: List[str]
    confidence: float
    execution_time: float
    steps_taken: int
    sources: List[str]


class MCPAgent:
    """
    MCP-enabled AI agent with Manim expertise and Context7 integration.
    
    This agent can:
    - Query Manim Community documentation through Context7
    - Provide accurate code examples based on current documentation
    - Generate responses using LLM integration with retrieved documentation
    - Handle multi-step interactions with configurable limits
    """
    
    # Default system prompt for Manim expertise
    DEFAULT_MANIM_SYSTEM_PROMPT = """You are a Manim expert assistant with access to current Manim Community documentation through Context7.

Your capabilities:
- Access up-to-date Manim Community documentation and code examples
- Provide accurate guidance on Manim animations, scenes, and mobjects
- Generate working Manim code based on current API documentation
- Help with Manim installation, configuration, and troubleshooting
- Explain Manim concepts with practical examples

Guidelines:
- Always use the most current documentation available through Context7
- Provide working code examples when possible
- Explain complex concepts step by step
- Reference official documentation sources
- Be precise about Manim version compatibility
- Suggest best practices for Manim development

When users ask about Manim functionality:
1. First retrieve relevant documentation if needed
2. Provide accurate information based on current docs
3. Include practical code examples
4. Explain the concepts clearly
5. Suggest related topics or advanced usage

Remember: Your knowledge comes from real-time documentation retrieval, so you can provide the most current and accurate Manim guidance."""
    
    def __init__(
        self,
        llm: Any,
        mcp_client: MCPClient,
        config: Optional[AgentConfig] = None
    ):
        """
        Initialize MCPAgent with LLM integration and MCP client.
        
        Args:
            llm: LangChain LLM instance (ChatOpenAI, ChatAnthropic, etc.)
            mcp_client: Initialized MCPClient instance
            config: Agent configuration options
            
        Raises:
            AgentError: If initialization fails
            LLMProviderError: If LLM configuration is invalid
        """
        self.llm = llm
        self.mcp_client = mcp_client
        self.config = config or AgentConfig()
        
        # Initialize Context7 documentation retriever
        self.docs_retriever = Context7DocsRetriever(mcp_client)
        
        # Agent execution components
        self.agent_executor: Optional[AgentExecutor] = None
        self.tools: List[Tool] = []
        
        # Performance tracking
        self._query_count = 0
        self._total_execution_time = 0.0
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.info("MCPAgent initialized with Manim-specific configuration")
    
    @classmethod
    def create_with_openai(
        cls,
        mcp_client: MCPClient,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        config: Optional[AgentConfig] = None
    ) -> "MCPAgent":
        """
        Create MCPAgent with OpenAI LLM.
        
        Args:
            mcp_client: Initialized MCPClient instance
            model: OpenAI model name
            api_key: OpenAI API key (uses environment variable if None)
            config: Agent configuration
            
        Returns:
            Configured MCPAgent instance
            
        Raises:
            LLMProviderError: If OpenAI configuration fails
        """
        if not OPENAI_AVAILABLE:
            raise LLMProviderError("OpenAI not available. Install with: pip install langchain-openai")
        
        try:
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise LLMProviderError("OpenAI API key not provided")
            
            llm = ChatOpenAI(
                model=model,
                api_key=api_key,
                temperature=config.temperature if config else 0.1,
                max_tokens=config.max_tokens if config else None
            )
            
            return cls(llm, mcp_client, config)
            
        except Exception as e:
            raise LLMProviderError(f"Failed to create OpenAI LLM: {e}")
    
    @classmethod
    def create_with_anthropic(
        cls,
        mcp_client: MCPClient,
        model: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        config: Optional[AgentConfig] = None
    ) -> "MCPAgent":
        """
        Create MCPAgent with Anthropic LLM.
        
        Args:
            mcp_client: Initialized MCPClient instance
            model: Anthropic model name
            api_key: Anthropic API key (uses environment variable if None)
            config: Agent configuration
            
        Returns:
            Configured MCPAgent instance
            
        Raises:
            LLMProviderError: If Anthropic configuration fails
        """
        if not ANTHROPIC_AVAILABLE:
            raise LLMProviderError("Anthropic not available. Install with: pip install langchain-anthropic")
        
        try:
            api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise LLMProviderError("Anthropic API key not provided")
            
            llm = ChatAnthropic(
                model=model,
                api_key=api_key,
                temperature=config.temperature if config else 0.1,
                max_tokens=config.max_tokens if config else None
            )
            
            return cls(llm, mcp_client, config)
            
        except Exception as e:
            raise LLMProviderError(f"Failed to create Anthropic LLM: {e}")
    
    @classmethod
    def create_with_groq(
        cls,
        mcp_client: MCPClient,
        model: str = "llama-3.1-70b-versatile",
        api_key: Optional[str] = None,
        config: Optional[AgentConfig] = None
    ) -> "MCPAgent":
        """
        Create MCPAgent with Groq LLM.
        
        Args:
            mcp_client: Initialized MCPClient instance
            model: Groq model name
            api_key: Groq API key (uses environment variable if None)
            config: Agent configuration
            
        Returns:
            Configured MCPAgent instance
            
        Raises:
            LLMProviderError: If Groq configuration fails
        """
        if not GROQ_AVAILABLE:
            raise LLMProviderError("Groq not available. Install with: pip install langchain-groq")
        
        try:
            api_key = api_key or os.getenv("GROQ_API_KEY")
            if not api_key:
                raise LLMProviderError("Groq API key not provided")
            
            llm = ChatGroq(
                model=model,
                api_key=api_key,
                temperature=config.temperature if config else 0.1,
                max_tokens=config.max_tokens if config else None
            )
            
            return cls(llm, mcp_client, config)
            
        except Exception as e:
            raise LLMProviderError(f"Failed to create Groq LLM: {e}")
    
    async def initialize(self) -> None:
        """
        Initialize the agent by setting up tools and agent executor.
        
        Raises:
            AgentError: If initialization fails
        """
        try:
            # Create tools for the agent
            await self._create_tools()
            
            # Create agent executor
            await self._create_agent_executor()
            
            # Validate agent setup
            await self._validate_agent()
            
            logger.info("MCPAgent initialization completed successfully")
            
        except Exception as e:
            logger.error(f"MCPAgent initialization failed: {e}")
            raise AgentError(f"Failed to initialize agent: {e}")
    
    async def _create_tools(self) -> None:
        """Create tools for the agent to use."""
        self.tools = [
            Tool(
                name="retrieve_manim_documentation",
                description="Retrieve current Manim Community documentation for a specific topic. "
                           "Use this when you need up-to-date information about Manim functionality, "
                           "API changes, or code examples. Specify a topic like 'animations', 'scenes', "
                           "'mobjects', 'text', 'geometry', etc.",
                func=self._retrieve_documentation_tool
            ),
            Tool(
                name="get_manim_code_examples",
                description="Get code examples from Manim Community documentation for a specific topic. "
                           "Use this when users ask for practical examples or want to see how to implement "
                           "specific Manim functionality.",
                func=self._get_code_examples_tool
            ),
            Tool(
                name="resolve_manim_library_info",
                description="Get information about the Manim Community library including version info "
                           "and available documentation sections. Use this for general library information.",
                func=self._resolve_library_info_tool
            )
        ]
        
        logger.info(f"Created {len(self.tools)} tools for MCPAgent")
    
    async def _create_agent_executor(self) -> None:
        """Create the agent executor with tools and prompt."""
        try:
            # Create system prompt
            system_prompt = self.config.system_prompt or self.DEFAULT_MANIM_SYSTEM_PROMPT
            
            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}")
            ])
            
            # Create agent
            agent = create_tool_calling_agent(self.llm, self.tools, prompt)
            
            # Create agent executor with configuration
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                max_iterations=self.config.max_steps,
                verbose=True,
                handle_parsing_errors=True,
                return_intermediate_steps=True
            )
            
            logger.info("Agent executor created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create agent executor: {e}")
            raise AgentError(f"Agent executor creation failed: {e}")
    
    async def _validate_agent(self) -> None:
        """Validate agent setup and connections."""
        try:
            # Test MCP client connection
            if not self.mcp_client.is_server_connected('context7'):
                logger.warning("Context7 server not connected - attempting reconnection")
                success = await self.mcp_client.reconnect_server('context7')
                if not success:
                    raise AgentError("Cannot connect to Context7 server")
            
            # Test documentation retrieval
            try:
                await self.docs_retriever.resolve_manim_library_id()
                logger.info("Documentation retrieval validation passed")
            except Exception as e:
                logger.warning(f"Documentation retrieval test failed: {e}")
                # Don't fail initialization, but log the issue
            
            # Test LLM
            try:
                test_response = await self.llm.ainvoke([HumanMessage(content="Hello")])
                logger.info("LLM validation passed")
            except Exception as e:
                raise AgentError(f"LLM validation failed: {e}")
            
            logger.info("Agent validation completed successfully")
            
        except Exception as e:
            logger.error(f"Agent validation failed: {e}")
            raise AgentError(f"Agent validation failed: {e}")
    
    async def _retrieve_documentation_tool(self, topic: str) -> str:
        """Tool function to retrieve Manim documentation."""
        try:
            docs = await self.docs_retriever.get_manim_documentation(
                topic=topic,
                max_tokens=self.config.documentation_max_tokens
            )
            
            # Format documentation for the agent
            formatted_docs = f"Documentation for topic '{topic}':\n\n"
            
            for section in docs["sections"]:
                formatted_docs += f"## {section['title']}\n\n"
                formatted_docs += f"{section['content']}\n\n"
                
                if section.get("code_snippets"):
                    formatted_docs += "### Code Examples:\n\n"
                    for snippet in section["code_snippets"]:
                        formatted_docs += f"```{snippet['language']}\n{snippet['code']}\n```\n\n"
            
            return formatted_docs
            
        except Exception as e:
            logger.error(f"Documentation retrieval tool error: {e}")
            return f"Error retrieving documentation for '{topic}': {e}"
    
    async def _get_code_examples_tool(self, topic: str) -> str:
        """Tool function to get code examples from Manim documentation."""
        try:
            docs = await self.docs_retriever.get_manim_documentation(
                topic=topic,
                include_code_examples=True,
                max_tokens=self.config.documentation_max_tokens
            )
            
            if not docs.get("code_examples"):
                return f"No code examples found for topic '{topic}'"
            
            # Format code examples
            formatted_examples = f"Code examples for '{topic}':\n\n"
            
            for i, example in enumerate(docs["code_examples"], 1):
                formatted_examples += f"### Example {i}\n\n"
                if example.get("description"):
                    formatted_examples += f"{example['description']}\n\n"
                formatted_examples += f"```{example['language']}\n{example['code']}\n```\n\n"
            
            return formatted_examples
            
        except Exception as e:
            logger.error(f"Code examples tool error: {e}")
            return f"Error retrieving code examples for '{topic}': {e}"
    
    async def _resolve_library_info_tool(self, query: str = "") -> str:
        """Tool function to get Manim library information."""
        try:
            library_id = await self.docs_retriever.resolve_manim_library_id()
            
            # Get general documentation to extract library info
            docs = await self.docs_retriever.get_manim_documentation(
                topic=None,  # General documentation
                max_tokens=5000
            )
            
            info = f"Manim Community Library Information:\n\n"
            info += f"Library ID: {library_id}\n"
            info += f"Total sections available: {docs['total_sections']}\n"
            info += f"Retrieved at: {docs['retrieved_at']}\n\n"
            
            if docs["sections"]:
                info += "Available documentation sections:\n"
                for section in docs["sections"][:5]:  # Show first 5 sections
                    info += f"- {section['title']}\n"
            
            return info
            
        except Exception as e:
            logger.error(f"Library info tool error: {e}")
            return f"Error retrieving library information: {e}"
    
    async def query(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Process user query and generate response with Manim expertise.
        
        This enhanced query processing pipeline:
        1. Analyzes the query to determine documentation needs
        2. Pre-fetches relevant documentation when beneficial
        3. Integrates documentation into the agent workflow
        4. Formats results with extracted code examples
        
        Args:
            user_input: User's question or request
            context: Optional additional context
            
        Returns:
            AgentResponse with answer and metadata
            
        Raises:
            AgentError: If query processing fails
        """
        start_time = time.time()
        self._query_count += 1
        
        try:
            if not self.agent_executor:
                raise AgentError("Agent not initialized - call initialize() first")
            
            logger.info(f"Processing query: {user_input[:100]}...")
            
            # Step 1: Analyze query to determine documentation needs
            doc_strategy = await self._analyze_query_for_documentation_needs(user_input)
            logger.info(f"Documentation strategy: {doc_strategy}")
            
            # Step 2: Pre-fetch documentation if beneficial
            prefetched_docs = None
            if doc_strategy.should_prefetch and self.config.auto_retrieve_docs:
                prefetched_docs = await self._prefetch_documentation(doc_strategy)
                logger.info(f"Pre-fetched documentation for topics: {doc_strategy.topics}")
            
            # Step 3: Prepare enhanced input with documentation context
            agent_input = await self._prepare_enhanced_agent_input(
                user_input, context, prefetched_docs, doc_strategy
            )
            
            # Step 4: Execute agent with timeout
            try:
                result = await asyncio.wait_for(
                    self.agent_executor.ainvoke(agent_input),
                    timeout=self.config.timeout_seconds
                )
            except asyncio.TimeoutError:
                raise AgentError(f"Query timed out after {self.config.timeout_seconds} seconds")
            
            # Step 5: Enhanced response processing and formatting
            response = await self._process_and_format_response(
                result, user_input, start_time, doc_strategy, prefetched_docs
            )
            
            logger.info(f"Query processed successfully in {response.execution_time:.2f}s with {response.steps_taken} steps")
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Query processing failed after {execution_time:.2f}s: {e}")
            raise AgentError(f"Query processing failed: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get agent performance statistics.
        
        Returns:
            Dictionary containing performance metrics
        """
        stats = {
            'query_count': self._query_count,
            'total_execution_time': self._total_execution_time,
            'average_execution_time': self._total_execution_time / max(self._query_count, 1),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_hit_rate': self._cache_hits / max(self._cache_hits + self._cache_misses, 1)
        }
        
        # Add MCP client cache stats if available
        if hasattr(self.mcp_client, 'get_cache_stats'):
            stats['mcp_client_cache'] = self.mcp_client.get_cache_stats()
        
        # Add documentation retriever stats if available
        if hasattr(self.docs_retriever, '_retrieval_stats'):
            stats['documentation_retrieval'] = self.docs_retriever._retrieval_stats.copy()
        
        return stats
    
    def clear_performance_stats(self) -> None:
        """Clear performance statistics."""
        self._query_count = 0
        self._total_execution_time = 0.0
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Cleared agent performance statistics")
    
    def clear_cache(self) -> None:
        """Clear all caches used by the agent."""
        if hasattr(self.mcp_client, 'clear_cache'):
            self.mcp_client.clear_cache()
        
        if hasattr(self.docs_retriever, '_doc_cache') and self.docs_retriever._doc_cache:
            self.docs_retriever._doc_cache.clear()
        
        logger.info("Cleared all agent caches")
    
    def _extract_code_blocks(self, text: str) -> List[str]:
        """Extract code blocks from text."""
        import re
        
        # Pattern for fenced code blocks - handle optional language and flexible newlines
        pattern = r'```(?:\w+)?\s*\n?(.*?)\n?```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        return [match.strip() for match in matches if match.strip()]
    
    async def _analyze_query_for_documentation_needs(self, user_input: str) -> DocumentationStrategy:
        """
        Analyze user query to determine documentation retrieval strategy.
        
        This method implements intelligent query analysis to determine:
        - Whether to pre-fetch documentation
        - Which topics to focus on
        - Priority level for documentation retrieval
        
        Args:
            user_input: User's query text
            
        Returns:
            DocumentationStrategy with retrieval recommendations
        """
        user_input_lower = user_input.lower()
        
        # Define Manim-specific keywords and their associated topics
        manim_topic_keywords = {
            'animations': ['animate', 'animation', 'transform', 'morph', 'transition', 'move', 'rotate', 'scale'],
            'scenes': ['scene', 'construct', 'play', 'wait', 'render', 'camera', 'background'],
            'mobjects': ['mobject', 'vmobject', 'text', 'circle', 'square', 'line', 'polygon', 'shape'],
            'geometry': ['circle', 'square', 'rectangle', 'polygon', 'line', 'arc', 'curve', 'path'],
            'text': ['text', 'latex', 'mathtex', 'tex', 'formula', 'equation', 'label'],
            'colors': ['color', 'fill', 'stroke', 'opacity', 'gradient', 'rgb', 'hex'],
            'positioning': ['position', 'move_to', 'shift', 'next_to', 'align', 'arrange', 'center'],
            'plotting': ['plot', 'graph', 'axes', 'function', 'chart', 'data', 'coordinate'],
            'config': ['config', 'setup', 'install', 'import', 'quality', 'resolution', 'format']
        }
        
        # Analyze query for topic relevance
        detected_topics = []
        keyword_matches = 0
        
        for topic, keywords in manim_topic_keywords.items():
            topic_score = sum(1 for keyword in keywords if keyword in user_input_lower)
            if topic_score > 0:
                detected_topics.append((topic, topic_score))
                keyword_matches += topic_score
        
        # Sort topics by relevance score
        detected_topics.sort(key=lambda x: x[1], reverse=True)
        top_topics = [topic for topic, score in detected_topics[:3]]  # Top 3 most relevant topics
        
        # Determine if we should pre-fetch documentation
        should_prefetch = False
        priority = 'low'
        reasoning = "General query - no specific documentation needs detected"
        estimated_tokens = 5000
        
        # High priority: Specific technical questions or code requests
        code_indicators = ['how to', 'example', 'code', 'implement', 'create', 'make', 'build', 'write']
        technical_indicators = ['error', 'problem', 'issue', 'not working', 'help', 'fix']
        
        has_code_request = any(indicator in user_input_lower for indicator in code_indicators)
        has_technical_issue = any(indicator in user_input_lower for indicator in technical_indicators)
        
        if keyword_matches >= 3 or has_code_request:
            should_prefetch = True
            priority = 'high'
            reasoning = "High relevance - specific Manim functionality requested"
            estimated_tokens = 10000
        elif keyword_matches >= 1 or has_technical_issue:
            should_prefetch = True
            priority = 'medium'
            reasoning = "Medium relevance - Manim-related query detected"
            estimated_tokens = 7000
        elif any(manim_word in user_input_lower for manim_word in ['manim', 'animation', 'scene']):
            should_prefetch = True
            priority = 'medium'
            reasoning = "Manim context detected - may benefit from documentation"
            estimated_tokens = 5000
        
        # If no specific topics detected but Manim context exists, use general topics
        if should_prefetch and not top_topics:
            top_topics = ['scenes', 'animations']  # Default to most common topics
        
        return DocumentationStrategy(
            should_prefetch=should_prefetch,
            topics=top_topics,
            priority=priority,
            reasoning=reasoning,
            estimated_tokens=estimated_tokens
        )
    
    async def _prefetch_documentation(self, strategy: DocumentationStrategy) -> Dict[str, Any]:
        """
        Pre-fetch documentation based on the determined strategy.
        
        Args:
            strategy: DocumentationStrategy with retrieval parameters
            
        Returns:
            Dictionary containing pre-fetched documentation
        """
        prefetched_docs = {
            'topics': {},
            'total_sections': 0,
            'total_code_examples': 0,
            'retrieval_time': 0
        }
        
        start_time = time.time()
        
        try:
            # Fetch documentation for each identified topic
            for topic in strategy.topics:
                try:
                    topic_docs = await self.docs_retriever.get_manim_documentation(
                        topic=topic,
                        max_tokens=strategy.estimated_tokens // len(strategy.topics)
                    )
                    
                    prefetched_docs['topics'][topic] = topic_docs
                    prefetched_docs['total_sections'] += topic_docs.get('total_sections', 0)
                    prefetched_docs['total_code_examples'] += len(topic_docs.get('code_examples', []))
                    
                    logger.debug(f"Pre-fetched documentation for topic '{topic}': "
                               f"{topic_docs.get('total_sections', 0)} sections")
                    
                except Exception as e:
                    logger.warning(f"Failed to pre-fetch documentation for topic '{topic}': {e}")
                    # Continue with other topics
                    continue
            
            prefetched_docs['retrieval_time'] = time.time() - start_time
            
            logger.info(f"Pre-fetched documentation: {prefetched_docs['total_sections']} sections, "
                       f"{prefetched_docs['total_code_examples']} code examples in "
                       f"{prefetched_docs['retrieval_time']:.2f}s")
            
        except Exception as e:
            logger.error(f"Error during documentation pre-fetching: {e}")
            # Return empty structure on error
            prefetched_docs['error'] = str(e)
        
        return prefetched_docs
    
    async def _prepare_enhanced_agent_input(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]],
        prefetched_docs: Optional[Dict[str, Any]],
        strategy: DocumentationStrategy
    ) -> Dict[str, Any]:
        """
        Prepare enhanced agent input with documentation context.
        
        Args:
            user_input: Original user query
            context: Optional additional context
            prefetched_docs: Pre-fetched documentation if available
            strategy: Documentation strategy used
            
        Returns:
            Enhanced input dictionary for agent execution
        """
        agent_input = {"input": user_input}
        
        # Add original context if provided
        if context:
            agent_input["context"] = context
        
        # Add documentation context if available
        if prefetched_docs and prefetched_docs.get('topics'):
            doc_context = self._format_documentation_context(prefetched_docs, strategy)
            agent_input["documentation_context"] = doc_context
            
            # Enhance the input prompt with documentation awareness
            enhanced_prompt = f"""Query: {user_input}

Available Documentation Context:
{doc_context}

Please use the provided documentation context to give accurate, up-to-date information. 
If the documentation contains relevant code examples, include them in your response.
"""
            agent_input["input"] = enhanced_prompt
        
        # Add strategy information for agent awareness
        agent_input["doc_strategy"] = {
            "topics": strategy.topics,
            "priority": strategy.priority,
            "reasoning": strategy.reasoning
        }
        
        return agent_input
    
    def _format_documentation_context(
        self,
        prefetched_docs: Dict[str, Any],
        strategy: DocumentationStrategy
    ) -> str:
        """
        Format pre-fetched documentation for agent context.
        
        Args:
            prefetched_docs: Pre-fetched documentation data
            strategy: Documentation strategy used
            
        Returns:
            Formatted documentation context string
        """
        context_parts = []
        
        context_parts.append(f"Documentation retrieved for topics: {', '.join(strategy.topics)}")
        context_parts.append(f"Priority: {strategy.priority}")
        context_parts.append("")
        
        for topic, topic_docs in prefetched_docs.get('topics', {}).items():
            context_parts.append(f"=== {topic.upper()} DOCUMENTATION ===")
            
            # Add key sections
            for section in topic_docs.get('sections', [])[:3]:  # Limit to top 3 sections
                context_parts.append(f"## {section['title']}")
                # Truncate content to avoid overwhelming the context
                content = section['content'][:500]
                if len(section['content']) > 500:
                    content += "... [truncated]"
                context_parts.append(content)
                context_parts.append("")
            
            # Add code examples
            code_examples = topic_docs.get('code_examples', [])
            if code_examples:
                context_parts.append("### Code Examples:")
                for i, example in enumerate(code_examples[:2], 1):  # Limit to 2 examples per topic
                    context_parts.append(f"Example {i}:")
                    context_parts.append(f"```{example['language']}")
                    context_parts.append(example['code'])
                    context_parts.append("```")
                    context_parts.append("")
            
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    async def _process_and_format_response(
        self,
        result: Dict[str, Any],
        user_input: str,
        start_time: float,
        strategy: DocumentationStrategy,
        prefetched_docs: Optional[Dict[str, Any]]
    ) -> AgentResponse:
        """
        Process and format the agent response with enhanced metadata.
        
        Args:
            result: Raw result from agent executor
            user_input: Original user query
            start_time: Query start time
            strategy: Documentation strategy used
            prefetched_docs: Pre-fetched documentation if available
            
        Returns:
            Enhanced AgentResponse with formatted results
        """
        # Extract basic response components
        answer = result.get("output", "No response generated")
        intermediate_steps = result.get("intermediate_steps", [])
        
        # Enhanced code example extraction
        code_examples = []
        documentation_used = []
        sources = []
        
        # Extract from intermediate steps
        for step in intermediate_steps:
            if len(step) >= 2:
                tool_input = step[0]
                tool_output = step[1]
                
                # Extract code examples from tool outputs
                if "```" in str(tool_output):
                    code_blocks = self._extract_code_blocks(str(tool_output))
                    code_examples.extend(code_blocks)
                
                # Track documentation usage
                if hasattr(tool_input, 'tool') and 'documentation' in tool_input.tool:
                    topic = getattr(tool_input, 'tool_input', {}).get('topic', 'general')
                    if topic not in documentation_used:
                        documentation_used.append(topic)
                
                # Extract sources
                if "Library ID:" in str(tool_output) or "Manim" in str(tool_output):
                    if "Manim Community Documentation" not in sources:
                        sources.append("Manim Community Documentation")
        
        # Extract code examples from the final answer as well
        answer_code_blocks = self._extract_code_blocks(answer)
        code_examples.extend(answer_code_blocks)
        
        # Remove duplicates while preserving order
        code_examples = list(dict.fromkeys(code_examples))
        
        # Add documentation topics from strategy if documentation was used
        if prefetched_docs and prefetched_docs.get('topics'):
            for topic in strategy.topics:
                if topic not in documentation_used:
                    documentation_used.append(topic)
        
        # Ensure we have sources
        if not sources:
            sources = ["Manim Community Documentation"]
        
        execution_time = time.time() - start_time
        self._total_execution_time += execution_time
        
        # Enhanced confidence calculation
        confidence = self._calculate_enhanced_confidence(
            answer, len(intermediate_steps), len(code_examples), 
            strategy, prefetched_docs, documentation_used
        )
        
        return AgentResponse(
            answer=answer,
            code_examples=code_examples,
            documentation_used=documentation_used,
            confidence=confidence,
            execution_time=execution_time,
            steps_taken=len(intermediate_steps),
            sources=sources
        )
    
    def _calculate_enhanced_confidence(
        self,
        answer: str,
        steps_taken: int,
        code_examples_count: int,
        strategy: DocumentationStrategy,
        prefetched_docs: Optional[Dict[str, Any]],
        documentation_used: List[str]
    ) -> float:
        """
        Calculate enhanced confidence score based on documentation integration.
        
        Args:
            answer: Generated answer text
            steps_taken: Number of agent steps
            code_examples_count: Number of code examples found
            strategy: Documentation strategy used
            prefetched_docs: Pre-fetched documentation data
            documentation_used: List of documentation topics used
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 0.4  # Base confidence
        
        # Boost confidence based on documentation strategy success
        if strategy.should_prefetch and prefetched_docs:
            if prefetched_docs.get('total_sections', 0) > 0:
                confidence += 0.2  # Successfully retrieved documentation
            if prefetched_docs.get('total_code_examples', 0) > 0:
                confidence += 0.1  # Retrieved code examples
        
        # Increase confidence based on documentation usage
        if documentation_used:
            confidence += min(0.2, len(documentation_used) * 0.05)
        
        # Increase confidence based on agent steps (shows thorough processing)
        if steps_taken > 0:
            confidence += min(0.15, steps_taken * 0.03)
        
        # Increase confidence if code examples are provided
        if code_examples_count > 0:
            confidence += min(0.15, code_examples_count * 0.03)
        
        # Increase confidence based on answer quality indicators
        if len(answer) > 200:
            confidence += 0.05
        if len(answer) > 500:
            confidence += 0.05
        
        # Check for specific Manim-related keywords in answer
        manim_keywords = ['manim', 'scene', 'mobject', 'animation', 'render', 'construct', 'play']
        keyword_count = sum(1 for keyword in manim_keywords if keyword.lower() in answer.lower())
        confidence += min(0.1, keyword_count * 0.02)
        
        # Boost confidence if strategy priority was high and we delivered
        if strategy.priority == 'high' and (code_examples_count > 0 or len(documentation_used) > 0):
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _calculate_confidence(self, answer: str, steps_taken: int, code_examples_count: int) -> float:
        """Calculate confidence score for the response (legacy method)."""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on documentation usage
        if steps_taken > 0:
            confidence += min(0.3, steps_taken * 0.1)
        
        # Increase confidence if code examples are provided
        if code_examples_count > 0:
            confidence += min(0.2, code_examples_count * 0.05)
        
        # Increase confidence based on answer length and detail
        if len(answer) > 200:
            confidence += 0.1
        
        # Check for specific Manim-related keywords
        manim_keywords = ['manim', 'scene', 'mobject', 'animation', 'render']
        keyword_count = sum(1 for keyword in manim_keywords if keyword.lower() in answer.lower())
        confidence += min(0.2, keyword_count * 0.04)
        
        return min(1.0, confidence)
    
    async def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics."""
        return {
            "query_count": self._query_count,
            "total_execution_time": self._total_execution_time,
            "average_execution_time": self._total_execution_time / max(1, self._query_count),
            "mcp_connection_status": self.mcp_client.get_connection_status(),
            "config": {
                "max_steps": self.config.max_steps,
                "timeout_seconds": self.config.timeout_seconds,
                "documentation_max_tokens": self.config.documentation_max_tokens,
                "auto_retrieve_docs": self.config.auto_retrieve_docs
            }
        }
    
    async def close(self) -> None:
        """Close agent and cleanup resources."""
        logger.info("Closing MCPAgent...")
        
        # Agent executor doesn't need explicit cleanup
        self.agent_executor = None
        self.tools.clear()
        
        logger.info("MCPAgent closed successfully")


# Convenience functions for common agent operations

async def create_manim_agent(
    mcp_client: MCPClient,
    model: Optional[str] = None,
    config: Optional[AgentConfig] = None
) -> MCPAgent:
    """
    Create and initialize a Manim-specific MCPAgent with OpenAI.
    
    Args:
        mcp_client: Initialized MCPClient instance
        model: OpenAI model name (defaults to "gpt-4o")
        config: Agent configuration
        
    Returns:
        Initialized MCPAgent instance
        
    Raises:
        AgentError: If agent creation fails
    """
    try:
        agent = MCPAgent.create_with_openai(mcp_client, model or "gpt-4o", config=config)
        await agent.initialize()
        return agent
        
    except Exception as e:
        logger.error(f"Failed to create Manim agent: {e}")
        raise AgentError(f"Agent creation failed: {e}")


async def quick_manim_query(
    query: str,
    mcp_client: MCPClient,
    model: Optional[str] = None
) -> str:
    """
    Quick function to query Manim documentation and get a response.
    
    Args:
        query: User's question about Manim
        mcp_client: Initialized MCPClient instance
        model: OpenAI model name
        
    Returns:
        Agent's response as string
        
    Raises:
        AgentError: If query fails
    """
    agent = await create_manim_agent(mcp_client, model)
    try:
        response = await agent.query(query)
        return response.answer
    finally:
        await agent.close()