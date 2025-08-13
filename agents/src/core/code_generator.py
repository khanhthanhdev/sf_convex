import os
import re
import json
from typing import Union, List, Dict, Optional
from PIL import Image
import glob
import asyncio

from src.utils.utils import extract_json
from mllm_tools.utils import _prepare_text_inputs, _extract_code, _prepare_text_image_inputs
from mllm_tools.gemini import GeminiWrapper
from mllm_tools.vertex_ai import VertexAIWrapper
from task_generator import (
    get_prompt_code_generation,
    get_prompt_fix_error,
    get_prompt_visual_fix_error,
    get_banned_reasonings,
    get_prompt_rag_query_generation_fix_error,
    get_prompt_context_learning_code,
    get_prompt_rag_query_generation_code
)
from task_generator.prompts_raw import (
    _code_font_size,
    _code_disable,
    _code_limit,
    _prompt_manim_cheatsheet
)
from src.rag.vector_store import RAGVectorStore # Import RAGVectorStore

# MCP imports for Context7 integration
try:
    from ..mcp_client.client import MCPClient
    from ..mcp_client.context7_docs import Context7DocsRetriever
    MCP_AVAILABLE = True
except ImportError:
    try:
        # Alternative import path
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        from mcp_client.client import MCPClient
        from mcp_client.context7_docs import Context7DocsRetriever
        MCP_AVAILABLE = True
    except ImportError:
        MCP_AVAILABLE = False

class CodeGenerator:
    """A class for generating and managing Manim code."""

    def __init__(self, scene_model, helper_model, output_dir="output", print_response=False, use_rag=False, use_context_learning=False, context_learning_path="data/context_learning", chroma_db_path="rag/chroma_db", manim_docs_path="rag/manim_docs", embedding_model="azure/text-embedding-3-large", use_visual_fix_code=False, use_langfuse=True, session_id=None, mcp_client: Optional['MCPClient'] = None):
        """Initialize the CodeGenerator.

        Args:
            scene_model: The model used for scene generation
            helper_model: The model used for helper tasks
            output_dir (str, optional): Directory for output files. Defaults to "output".
            print_response (bool, optional): Whether to print model responses. Defaults to False.
            use_rag (bool, optional): Whether to use RAG. Defaults to False.
            use_context_learning (bool, optional): Whether to use context learning. Defaults to False.
            context_learning_path (str, optional): Path to context learning examples. Defaults to "data/context_learning".
            chroma_db_path (str, optional): Path to ChromaDB. Defaults to "rag/chroma_db".
            manim_docs_path (str, optional): Path to Manim docs. Defaults to "rag/manim_docs".
            embedding_model (str, optional): Name of embedding model. Defaults to "azure/text-embedding-3-large".
            use_visual_fix_code (bool, optional): Whether to use visual code fixing. Defaults to False.
            use_langfuse (bool, optional): Whether to use Langfuse logging. Defaults to True.
            session_id (str, optional): Session identifier. Defaults to None.
            mcp_client (MCPClient, optional): MCP client for Context7 integration. Defaults to None.
        """
        self.scene_model = scene_model
        self.helper_model = helper_model
        self.output_dir = output_dir
        self.print_response = print_response
        self.use_rag = use_rag
        self.use_context_learning = use_context_learning
        self.context_learning_path = context_learning_path
        self.context_examples = self._load_context_examples() if use_context_learning else None
        self.manim_docs_path = manim_docs_path

        self.use_visual_fix_code = use_visual_fix_code
        self.banned_reasonings = get_banned_reasonings()
        self.session_id = session_id # Use session_id passed from VideoGenerator

        # MCP Client integration for Context7 documentation
        self.mcp_client = mcp_client
        self.context7_retriever = None
        
        # Initialize Context7 retriever if MCP client is provided
        if self.mcp_client and MCP_AVAILABLE:
            self.context7_retriever = Context7DocsRetriever(self.mcp_client)
            print("Context7 documentation retriever initialized")
        elif self.mcp_client and not MCP_AVAILABLE:
            print("Warning: MCP client provided but MCP dependencies not available")

        # Initialize traditional RAG if enabled and no MCP client provided
        if use_rag and not self.mcp_client:
            self.vector_store = RAGVectorStore(
                chroma_db_path=chroma_db_path,
                manim_docs_path=manim_docs_path,
                embedding_model=embedding_model,
                session_id=self.session_id,
                use_langfuse=use_langfuse
            )
        else:
            self.vector_store = None

    def _load_context_examples(self) -> str:
        """Load all context learning examples from the specified directory.

        Returns:
            str: Formatted context learning examples, or None if no examples found.
        """
        examples = []
        for example_file in glob.glob(f"{self.context_learning_path}/**/*.py", recursive=True):
            with open(example_file, 'r') as f:
                examples.append(f"# Example from {os.path.basename(example_file)}\n{f.read()}\n")

        # Format examples using get_prompt_context_learning_code instead of _prompt_context_learning
        if examples:
            formatted_examples = get_prompt_context_learning_code(
                examples="\n".join(examples)
            )
            return formatted_examples
        return None

    def _generate_rag_queries_code(self, implementation: str, scene_trace_id: str = None, topic: str = None, scene_number: int = None, session_id: str = None, relevant_plugins: List[str] = []) -> Union[List[str], str]:
        """Generate documentation queries and retrieve Context7 documentation or traditional RAG.

        Args:
            implementation (str): The implementation plan text
            scene_trace_id (str, optional): Trace ID for the scene. Defaults to None.
            topic (str, optional): Topic of the scene. Defaults to None.
            scene_number (int, optional): Scene number. Defaults to None.
            session_id (str, optional): Session identifier. Defaults to None.
            relevant_plugins (List[str], optional): List of relevant plugins. Defaults to empty list.

        Returns:
            Union[List[str], str]: List of RAG queries for traditional RAG or formatted documentation string for Context7
        """
        # If Context7 retriever is available, use it instead of traditional RAG
        if self.context7_retriever:
            return self._retrieve_context7_documentation(
                implementation=implementation,
                scene_trace_id=scene_trace_id,
                topic=topic,
                scene_number=scene_number,
                session_id=session_id,
                relevant_plugins=relevant_plugins
            )

        # Fallback to traditional RAG query generation
        # Create a cache key for this scene
        cache_key = f"{topic}_scene{scene_number}"

        # Check if we already have a cache file for this scene
        cache_dir = os.path.join(self.output_dir, re.sub(r'[^a-z0-9_]+', '_', topic.lower()), f"scene{scene_number}", "rag_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "rag_queries_code.json")

        # If cache file exists, load and return cached queries
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cached_queries = json.load(f)
                print(f"Using cached RAG queries for {cache_key}")
                return cached_queries

        # Generate new queries if not cached
        if relevant_plugins:
            prompt = get_prompt_rag_query_generation_code(implementation, ", ".join(relevant_plugins))
        else:
            prompt = get_prompt_rag_query_generation_code(implementation, "No plugins are relevant.")

        queries = self.helper_model(
            _prepare_text_inputs(prompt),
            metadata={"generation_name": "rag_query_generation", "trace_id": scene_trace_id, "tags": [topic, f"scene{scene_number}"], "session_id": session_id}
        )

        print(f"RAG queries: {queries}")
        # retreive json triple backticks
        
        try: # add try-except block to handle potential json decode errors
            queries = re.search(r'```json(.*)```', queries, re.DOTALL).group(1)
            queries = json.loads(queries)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError when parsing RAG queries for storyboard: {e}")
            print(f"Response text was: {queries}")
            return [] # Return empty list in case of parsing error

        # Cache the queries
        with open(cache_file, 'w') as f:
            json.dump(queries, f)

        return queries

    def _retrieve_context7_documentation(self, implementation: str, scene_trace_id: str = None, topic: str = None, scene_number: int = None, session_id: str = None, relevant_plugins: List[str] = []) -> str:
        """Retrieve Manim documentation using Context7 MCP server.

        Args:
            implementation (str): The implementation plan text
            scene_trace_id (str, optional): Trace ID for the scene. Defaults to None.
            topic (str, optional): Topic of the scene. Defaults to None.
            scene_number (int, optional): Scene number. Defaults to None.
            session_id (str, optional): Session identifier. Defaults to None.
            relevant_plugins (List[str], optional): List of relevant plugins. Defaults to empty list.

        Returns:
            str: Formatted documentation string for use in prompts
        """
        try:
            # Create a cache key for this scene
            cache_key = f"{topic}_scene{scene_number}_context7"
            cache_dir = os.path.join(self.output_dir, re.sub(r'[^a-z0-9_]+', '_', topic.lower()), f"scene{scene_number}", "context7_cache")
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, "context7_docs.json")

            # Check cache first
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cached_docs = json.load(f)
                    print(f"Using cached Context7 documentation for {cache_key}")
                    return cached_docs.get('formatted_docs', '')

            # Extract topic from implementation for Context7 query
            context7_topic = self._extract_manim_topic_from_implementation(implementation)
            
            print(f"Retrieving Context7 documentation for topic: {context7_topic}")
            
            # Run async documentation retrieval in sync context
            loop = None
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            if loop.is_running():
                # If we're already in an async context, create a new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.context7_retriever.get_manim_documentation(
                            topic=context7_topic,
                            include_code_examples=True,
                            max_tokens=8000
                        )
                    )
                    doc_response = future.result(timeout=30)
            else:
                # Run directly in the event loop
                doc_response = loop.run_until_complete(
                    self.context7_retriever.get_manim_documentation(
                        topic=context7_topic,
                        include_code_examples=True,
                        max_tokens=8000
                    )
                )

            # Format the documentation for use in prompts
            formatted_docs = self._format_context7_docs_for_prompt(doc_response)
            
            # Cache the result
            cache_data = {
                'topic': context7_topic,
                'formatted_docs': formatted_docs,
                'retrieved_at': doc_response.get('retrieved_at', ''),
                'total_sections': doc_response.get('total_sections', 0)
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            print(f"Retrieved and cached Context7 documentation: {doc_response.get('total_sections', 0)} sections")
            return formatted_docs

        except Exception as e:
            print(f"Error retrieving Context7 documentation: {e}")
            # Fallback to empty documentation
            return "# Manim Documentation\n\nNo documentation available from Context7 server."

    def _extract_manim_topic_from_implementation(self, implementation: str) -> Optional[str]:
        """Extract relevant Manim topic from implementation text.

        Args:
            implementation (str): Implementation plan text

        Returns:
            Optional[str]: Extracted topic or None for general documentation
        """
        # Common Manim topics to look for
        manim_topics = {
            'animation': ['animate', 'animation', 'transform', 'morph', 'transition'],
            'mobject': ['mobject', 'vmobject', 'text', 'shape', 'geometry'],
            'scene': ['scene', 'construct', 'play', 'wait'],
            'camera': ['camera', 'zoom', 'frame', 'view'],
            'coordinate': ['coordinate', 'axes', 'graph', 'plot'],
            'color': ['color', 'fill', 'stroke', 'opacity'],
            'movement': ['move', 'shift', 'rotate', 'scale'],
            'creation': ['create', 'write', 'draw', 'show', 'fade']
        }
        
        implementation_lower = implementation.lower()
        
        # Find the most relevant topic based on keyword frequency
        topic_scores = {}
        for topic, keywords in manim_topics.items():
            score = sum(implementation_lower.count(keyword) for keyword in keywords)
            if score > 0:
                topic_scores[topic] = score
        
        if topic_scores:
            # Return the topic with the highest score
            best_topic = max(topic_scores, key=topic_scores.get)
            print(f"Detected Manim topic: {best_topic} (score: {topic_scores[best_topic]})")
            return best_topic
        
        # Return None for general documentation if no specific topic detected
        return None

    def _format_context7_docs_for_prompt(self, doc_response: Dict) -> str:
        """Format Context7 documentation response for use in code generation prompts.

        Args:
            doc_response (Dict): Documentation response from Context7

        Returns:
            str: Formatted documentation string
        """
        if not doc_response or not doc_response.get('sections'):
            return "# Manim Documentation\n\nNo documentation sections available."
        
        formatted_parts = []
        formatted_parts.append("# Manim Community Documentation")
        formatted_parts.append(f"Retrieved from Context7 - Library: {doc_response.get('library_id', 'Unknown')}")
        
        if doc_response.get('topic'):
            formatted_parts.append(f"Topic: {doc_response['topic']}")
        
        formatted_parts.append("")
        
        # Add documentation sections
        for i, section in enumerate(doc_response.get('sections', []), 1):
            formatted_parts.append(f"## Section {i}: {section.get('title', 'Untitled')}")
            
            if section.get('content'):
                formatted_parts.append(section['content'])
            
            # Add code snippets from this section
            if section.get('code_snippets'):
                formatted_parts.append("\n### Code Examples:")
                for j, snippet in enumerate(section['code_snippets'], 1):
                    formatted_parts.append(f"\n#### Example {j}:")
                    if snippet.get('description'):
                        formatted_parts.append(snippet['description'])
                    formatted_parts.append(f"```{snippet.get('language', 'python')}")
                    formatted_parts.append(snippet.get('code', ''))
                    formatted_parts.append("```")
            
            formatted_parts.append("")
        
        # Add formatted code examples at the end
        if doc_response.get('formatted_code'):
            formatted_parts.append("## Additional Code Examples")
            formatted_parts.append(doc_response['formatted_code'])
        
        return "\n".join(formatted_parts)

    def _generate_rag_queries_error_fix(self, error: str, code: str, scene_trace_id: str = None, topic: str = None, scene_number: int = None, session_id: str = None, relevant_plugins: List[str] = []) -> List[str]:
        """Generate RAG queries for fixing code errors.

        Args:
            error (str): The error message to fix
            code (str): The code containing the error
            scene_trace_id (str, optional): Trace ID for the scene. Defaults to None.
            topic (str, optional): Topic of the scene. Defaults to None.
            scene_number (int, optional): Scene number. Defaults to None.
            session_id (str, optional): Session identifier. Defaults to None.
            relevant_plugins (List[str], optional): List of relevant plugins. Defaults to empty list.

        Returns:
            List[str]: List of generated RAG queries for error fixing
        """
        # Create a cache key for this scene and error
        cache_key = f"{topic}_scene{scene_number}_error_fix"

        # Check if we already have a cache file for error fix queries
        cache_dir = os.path.join(self.output_dir, re.sub(r'[^a-z0-9_]+', '_', topic.lower()), f"scene{scene_number}", "rag_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "rag_queries_error_fix.json")

        # If cache file exists, load and return cached queries
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cached_queries = json.load(f)
                print(f"Using cached RAG queries for error fix in {cache_key}")
                return cached_queries

        # Generate new queries for error fix if not cached
        prompt = get_prompt_rag_query_generation_fix_error(
            error=error,
            code=code,
            relevant_plugins=", ".join(relevant_plugins) if relevant_plugins else "No plugins are relevant."
        )

        queries = self.helper_model(
            _prepare_text_inputs(prompt),
            metadata={"generation_name": "rag-query-generation-fix-error", "trace_id": scene_trace_id, "tags": [topic, f"scene{scene_number}"], "session_id": session_id}
        )

        # remove json triple backticks
        queries = queries.replace("```json", "").replace("```", "")
        try: # add try-except block to handle potential json decode errors
            queries = json.loads(queries)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError when parsing RAG queries for error fix: {e}")
            print(f"Response text was: {queries}")
            return [] # Return empty list in case of parsing error

        # Cache the queries
        with open(cache_file, 'w') as f:
            json.dump(queries, f)

        return queries

    def _extract_code_with_retries(self, response_text: str, pattern: str, generation_name: str = None, trace_id: str = None, session_id: str = None, max_retries: int = 10) -> str:
        """Extract code from response text with retry logic.

        Args:
            response_text (str): The text containing code to extract
            pattern (str): Regex pattern for extracting code
            generation_name (str, optional): Name of generation step. Defaults to None.
            trace_id (str, optional): Trace identifier. Defaults to None.
            session_id (str, optional): Session identifier. Defaults to None.
            max_retries (int, optional): Maximum number of retries. Defaults to 10.

        Returns:
            str: The extracted code

        Raises:
            ValueError: If code extraction fails after max retries
        """
        retry_prompt = """
        Please extract the Python code in the correct format using the pattern: {pattern}. 
        You MUST NOT include any other text or comments. 
        You MUST return the exact same code as in the previous response, NO CONTENT EDITING is allowed.
        Previous response: 
        {response_text}
        """

        for attempt in range(max_retries):
            code_match = re.search(pattern, response_text, re.DOTALL)
            if code_match:
                return code_match.group(1)
            
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1}: Failed to extract code pattern. Retrying...")
                # Regenerate response with a more explicit prompt
                response_text = self.scene_model(
                    _prepare_text_inputs(retry_prompt.format(pattern=pattern, response_text=response_text)),
                    metadata={
                        "generation_name": f"{generation_name}_format_retry_{attempt + 1}",
                        "trace_id": trace_id,
                        "session_id": session_id
                    }
                )
        
        raise ValueError(f"Failed to extract code pattern after {max_retries} attempts. Pattern: {pattern}")

    def generate_manim_code(self,
                            topic: str,
                            description: str,                            
                            scene_outline: str,
                            scene_implementation: str,
                            scene_number: int,
                            additional_context: Union[str, List[str]] = None,
                            scene_trace_id: str = None,
                            session_id: str = None,
                            rag_queries_cache: Dict = None) -> str:
        """Generate Manim code from video plan.

        Args:
            topic (str): Topic of the scene
            description (str): Description of the scene
            scene_outline (str): Outline of the scene
            scene_implementation (str): Implementation details
            scene_number (int): Scene number
            additional_context (Union[str, List[str]], optional): Additional context. Defaults to None.
            scene_trace_id (str, optional): Trace identifier. Defaults to None.
            session_id (str, optional): Session identifier. Defaults to None.
            rag_queries_cache (Dict, optional): Cache for RAG queries. Defaults to None.

        Returns:
            Tuple[str, str]: Generated code and response text
        """
        if self.use_context_learning:
            # Add context examples to additional_context
            if additional_context is None:
                additional_context = []
            elif isinstance(additional_context, str):
                additional_context = [additional_context]
            
            # Now using the properly formatted code examples
            if self.context_examples:
                additional_context.append(self.context_examples)

        if self.use_rag or self.context7_retriever:
            # Use Context7 documentation if available, otherwise fall back to traditional RAG
            if self.context7_retriever:
                # Retrieve Context7 documentation directly
                retrieved_docs = self._generate_rag_queries_code(
                    implementation=scene_implementation,
                    scene_trace_id=scene_trace_id,
                    topic=topic,
                    scene_number=scene_number,
                    session_id=session_id
                )
                
                # Add Context7 documentation to context
                if additional_context is None:
                    additional_context = []
                additional_context.append(retrieved_docs)
                print("Using Context7 documentation for code generation")
                
            elif self.vector_store:
                # Traditional RAG approach
                rag_queries = self._generate_rag_queries_code(
                    implementation=scene_implementation,
                    scene_trace_id=scene_trace_id,
                    topic=topic,
                    scene_number=scene_number,
                    session_id=session_id
                )

                retrieved_docs = self.vector_store.find_relevant_docs(
                    queries=rag_queries,
                    k=2, # number of documents to retrieve
                    trace_id=scene_trace_id,
                    topic=topic,
                    scene_number=scene_number
                )
                # Format the retrieved documents into a string
                if additional_context is None:
                    additional_context = []
                additional_context.append(retrieved_docs)
                print("Using traditional RAG for code generation")

        # Format code generation prompt with plan and retrieved context
        prompt = get_prompt_code_generation(
            scene_outline=scene_outline,
            scene_implementation=scene_implementation,
            topic=topic,
            description=description,
            scene_number=scene_number,
            additional_context=additional_context
        )

        # Generate code using model
        response_text = self.scene_model(
            _prepare_text_inputs(prompt),
            metadata={"generation_name": "code_generation", "trace_id": scene_trace_id, "tags": [topic, f"scene{scene_number}"], "session_id": session_id}
        )

        # Extract code with retries
        code = self._extract_code_with_retries(
            response_text,
            r"```python(.*)```",
            generation_name="code_generation",
            trace_id=scene_trace_id,
            session_id=session_id
        )
        return code, response_text

    def fix_code_errors(self, implementation_plan: str, code: str, error: str, scene_trace_id: str, topic: str, scene_number: int, session_id: str, rag_queries_cache: Dict = None) -> str:
        """Fix errors in generated Manim code.

        Args:
            implementation_plan (str): Original implementation plan
            code (str): Code containing errors
            error (str): Error message to fix
            scene_trace_id (str): Trace identifier
            topic (str): Topic of the scene
            scene_number (int): Scene number
            session_id (str): Session identifier
            rag_queries_cache (Dict, optional): Cache for RAG queries. Defaults to None.

        Returns:
            Tuple[str, str]: Fixed code and response text
        """
        # Format error fix prompt
        prompt = get_prompt_fix_error(implementation_plan=implementation_plan, manim_code=code, error=error)

        if self.use_rag or self.context7_retriever:
            # Use Context7 documentation if available, otherwise fall back to traditional RAG
            if self.context7_retriever:
                # For error fixing, retrieve general Manim documentation
                try:
                    retrieved_docs = self._retrieve_context7_documentation(
                        implementation=f"Error fixing: {error}\nCode: {code}",
                        scene_trace_id=scene_trace_id,
                        topic=topic,
                        scene_number=scene_number,
                        session_id=session_id
                    )
                    prompt = get_prompt_fix_error(implementation_plan=implementation_plan, manim_code=code, error=error, additional_context=retrieved_docs)
                    print("Using Context7 documentation for error fixing")
                except Exception as e:
                    print(f"Error retrieving Context7 docs for error fixing: {e}")
                    prompt = get_prompt_fix_error(implementation_plan=implementation_plan, manim_code=code, error=error)
                    
            elif self.vector_store:
                # Traditional RAG approach for error fixing
                rag_queries = self._generate_rag_queries_error_fix(
                    error=error,
                    code=code,
                    scene_trace_id=scene_trace_id,
                    topic=topic,
                    scene_number=scene_number,
                    session_id=session_id
                )
                retrieved_docs = self.vector_store.find_relevant_docs(
                    queries=rag_queries,
                    k=2, # number of documents to retrieve for error fixing
                    trace_id=scene_trace_id,
                    topic=topic,
                    scene_number=scene_number
                )
                # Format the retrieved documents into a string
                prompt = get_prompt_fix_error(implementation_plan=implementation_plan, manim_code=code, error=error, additional_context=retrieved_docs)
                print("Using traditional RAG for error fixing")

        # Get fixed code from model
        response_text = self.scene_model(
            _prepare_text_inputs(prompt),
            metadata={"generation_name": "code_fix_error", "trace_id": scene_trace_id, "tags": [topic, f"scene{scene_number}"], "session_id": session_id}
        )

        # Extract fixed code with retries
        fixed_code = self._extract_code_with_retries(
            response_text,
            r"```python(.*)```",
            generation_name="code_fix_error",
            trace_id=scene_trace_id,
            session_id=session_id
        )
        return fixed_code, response_text

    def visual_self_reflection(self, code: str, media_path: Union[str, Image.Image], scene_trace_id: str, topic: str, scene_number: int, session_id: str) -> str:
        """Use snapshot image or mp4 video to fix code.

        Args:
            code (str): Code to fix
            media_path (Union[str, Image.Image]): Path to media file or PIL Image
            scene_trace_id (str): Trace identifier
            topic (str): Topic of the scene
            scene_number (int): Scene number
            session_id (str): Session identifier

        Returns:
            Tuple[str, str]: Fixed code and response text
        """
        
        # Determine if we're dealing with video or image
        is_video = isinstance(media_path, str) and media_path.endswith('.mp4')
        
        # Load prompt template
        with open('task_generator/prompts_raw/prompt_visual_self_reflection.txt', 'r') as f:
            prompt_template = f.read()
        
        # Format prompt
        prompt = prompt_template.format(code=code)
        
        # Prepare input based on media type
        if is_video and isinstance(self.scene_model, (GeminiWrapper, VertexAIWrapper)):
            # For video with Gemini models
            messages = [
                {"type": "text", "content": prompt},
                {"type": "video", "content": media_path}
            ]
        else:
            # For images or non-Gemini models
            if isinstance(media_path, str):
                media = Image.open(media_path)
            else:
                media = media_path
            messages = [
                {"type": "text", "content": prompt},
                {"type": "image", "content": media}
            ]
        
        # Get model response
        response_text = self.scene_model(
            messages,
            metadata={
                "generation_name": "visual_self_reflection",
                "trace_id": scene_trace_id,
                "tags": [topic, f"scene{scene_number}"],
                "session_id": session_id
            }
        )
        
        # Extract code with retries
        fixed_code = self._extract_code_with_retries(
            response_text,
            r"```python(.*)```",
            generation_name="visual_self_reflection",
            trace_id=scene_trace_id,
            session_id=session_id
        )
        return fixed_code, response_text