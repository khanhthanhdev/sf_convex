import os
import re
import json
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

from mllm_tools.utils import _prepare_text_inputs
from task_generator import (
    get_prompt_rag_query_generation_fix_error,
    get_prompt_detect_plugins,
    get_prompt_rag_query_generation_technical,
    get_prompt_rag_query_generation_vision_storyboard,
    get_prompt_rag_query_generation_narration,
    get_prompt_rag_query_generation_code
)

# Import centralized configuration management
from src.config.manager import ConfigurationManager

# Import new provider system
try:
    from .provider_factory import create_embedding_provider, EmbeddingProviderFactory
    from .vector_store_factory import create_vector_store, VectorStoreFactory
    from .vector_store_providers import VectorStoreConfig
    from .embedding_providers import EmbeddingConfig
    NEW_PROVIDER_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"New provider system not available: {e}")
    NEW_PROVIDER_SYSTEM_AVAILABLE = False

# Fallback to legacy vector store
from src.rag.vector_store import EnhancedRAGVectorStore as RAGVectorStore

# Import enhanced components
try:
    from .query_generation import (
        IntelligentQueryGenerator, InputContext, TaskType, EnhancedQuery
    )
    from .context_aware_retrieval import (
        ContextAwareRetriever, RetrievalContext, TaskType as RetrievalTaskType
    )
    from .semantic_query_expansion import SemanticQueryExpansionEngine
    from .performance_cache import (
        PerformanceOptimizedCache, CacheConfig, CacheType
    )
    from .error_handling import RobustErrorHandler, ErrorHandlingConfig
    from .quality_monitor import QualityMonitor
    from .plugin_detection import ContextAwarePluginDetector
    from .feedback_collector import FeedbackCollector
    from .rag_quality_evaluator import RAGQualityEvaluator
    ENHANCED_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Enhanced components not fully available: {e}")
    ENHANCED_COMPONENTS_AVAILABLE = False

@dataclass
class RAGConfig:
    """Configuration for enhanced RAG features."""
    use_enhanced_components: bool = True
    enable_caching: bool = True
    enable_quality_monitoring: bool = True
    enable_error_handling: bool = True
    cache_ttl: int = 3600
    max_cache_size: int = 1000
    performance_threshold: float = 2.0  # seconds
    quality_threshold: float = 0.7

class RAGIntegration:
    """Class for integrating RAG (Retrieval Augmented Generation) functionality.

    This class handles RAG integration including plugin detection, query generation,
    and document retrieval.

    Args:
        helper_model: Model used for generating queries and processing text
        output_dir (str): Directory for output files
        chroma_db_path (str): Path to ChromaDB
        manim_docs_path (str): Path to Manim documentation
        embedding_model (str): Name of embedding model to use
        use_langfuse (bool, optional): Whether to use Langfuse logging. Defaults to True
        session_id (str, optional): Session identifier. Defaults to None
    """

    def __init__(self, helper_model, output_dir, chroma_db_path=None, manim_docs_path=None, embedding_model=None, 
                 use_langfuse=True, session_id=None, config: Optional[RAGConfig] = None,
                 embedding_provider=None, vector_store_provider=None, config_manager=None):
        """Initialize RAG integration with centralized configuration management.
        
        Args:
            helper_model: Model used for generating queries and processing text
            output_dir (str): Directory for output files
            chroma_db_path (str, optional): Path to ChromaDB (legacy parameter, overridden by config)
            manim_docs_path (str, optional): Path to Manim documentation (legacy parameter, overridden by config)
            embedding_model (str, optional): Name of embedding model to use (legacy parameter, overridden by config)
            use_langfuse (bool, optional): Whether to use Langfuse logging. Defaults to True
            session_id (str, optional): Session identifier. Defaults to None
            config (RAGConfig, optional): RAG configuration. Defaults to None
            embedding_provider (EmbeddingProvider, optional): Embedding provider instance. Defaults to None
            vector_store_provider (VectorStoreProvider, optional): Vector store provider instance. Defaults to None
            config_manager (ConfigurationManager, optional): Configuration manager instance. Defaults to None
        """
        self.helper_model = helper_model
        self.output_dir = output_dir
        self.session_id = session_id
        self.relevant_plugins = None
        
        # Initialize configuration manager
        self.config_manager = config_manager or ConfigurationManager()
        
        # Get RAG configuration from centralized config
        rag_config = self.config_manager.get_rag_config()
        if rag_config:
            self.config = config or RAGConfig(
                use_enhanced_components=rag_config.enable_semantic_search,
                enable_caching=rag_config.enable_caching,
                enable_quality_monitoring=rag_config.enable_quality_monitoring,
                enable_error_handling=True,
                cache_ttl=rag_config.cache_ttl,
                max_cache_size=rag_config.max_cache_size,
                performance_threshold=2.0,
                quality_threshold=rag_config.quality_threshold
            )
            
            # Override legacy parameters with configuration values
            if not manim_docs_path:
                manim_docs_path = "data/rag/manim_docs"  # Default path
            if not chroma_db_path:
                chroma_db_path = rag_config.vector_store_config.connection_params.get('path', 'data/rag/chroma_db')
            if not embedding_model:
                embedding_model = rag_config.embedding_config.model_name
        else:
            self.config = config or RAGConfig()
            # Use legacy parameters as fallbacks
            manim_docs_path = manim_docs_path or "data/rag/manim_docs"
            chroma_db_path = chroma_db_path or "data/rag/chroma_db"
            embedding_model = embedding_model or "hf:ibm-granite/granite-embedding-30m-english"
        
        self.manim_docs_path = manim_docs_path
        
        # Initialize embedding and vector store providers
        self.embedding_provider = None
        self.vector_store_provider = None
        
        # Initialize vector store (new provider system or legacy)
        self._initialize_vector_store(
            chroma_db_path=chroma_db_path,
            embedding_model=embedding_model,
            use_langfuse=use_langfuse,
            embedding_provider=embedding_provider,
            vector_store_provider=vector_store_provider
        )

        # Initialize enhanced components if available and enabled
        self._initialize_enhanced_components()

    def _initialize_vector_store(self, chroma_db_path: str, embedding_model: str, 
                                use_langfuse: bool, embedding_provider=None, 
                                vector_store_provider=None) -> None:
        """Initialize vector store using centralized configuration and new provider system.
        
        Args:
            chroma_db_path (str): Path to ChromaDB (legacy parameter, overridden by config)
            embedding_model (str): Name of embedding model (legacy parameter, overridden by config) 
            use_langfuse (bool): Whether to use Langfuse logging
            embedding_provider: Optional embedding provider instance
            vector_store_provider: Optional vector store provider instance
        """
        # Try to use the new provider system with centralized configuration
        if NEW_PROVIDER_SYSTEM_AVAILABLE:
            try:
                # Get RAG configuration from centralized config
                rag_config = self.config_manager.get_rag_config()
                
                if rag_config and rag_config.enabled:
                    # Use centralized configuration for providers
                    if embedding_provider is None:
                        # Create embedding provider from centralized config
                        embedding_config = EmbeddingConfig(
                            provider=rag_config.embedding_config.provider,
                            model_name=rag_config.embedding_config.model_name,
                            api_key=rag_config.embedding_config.api_key,
                            dimensions=rag_config.embedding_config.dimensions,
                            batch_size=rag_config.embedding_config.batch_size,
                            timeout=rag_config.embedding_config.timeout
                        )
                        self.embedding_provider = create_embedding_provider(embedding_config)
                        print(f"Created embedding provider from config: {self.embedding_provider.config.provider}")
                    else:
                        self.embedding_provider = embedding_provider
                        print(f"Using provided embedding provider: {self.embedding_provider.config.provider}")
                    
                    if vector_store_provider is None:
                        # Create vector store provider from centralized config
                        provider_name = rag_config.vector_store_config.provider
                        if provider_name == "chroma":
                            provider_name = "chromadb"
                        
                        # Create connection params based on provider
                        connection_params = rag_config.vector_store_config.connection_params.copy()
                        if provider_name == "chromadb" and 'path' not in connection_params:
                            connection_params['path'] = chroma_db_path
                        
                        vector_store_config = VectorStoreConfig(
                            provider=provider_name,
                            collection_name=rag_config.vector_store_config.collection_name,
                            connection_params=connection_params,
                            embedding_dimension=rag_config.embedding_config.dimensions,
                            distance_metric=rag_config.vector_store_config.distance_metric
                        )
                        self.vector_store_provider = create_vector_store(
                            config=vector_store_config,
                            embedding_provider=self.embedding_provider
                        )
                        print(f"Created vector store provider from config: {self.vector_store_provider.config.provider}")
                    else:
                        self.vector_store_provider = vector_store_provider
                        print(f"Using provided vector store provider: {self.vector_store_provider.config.provider}")
                    
                    # Set the vector store to the new provider (for backward compatibility)
                    self.vector_store = self.vector_store_provider
                    
                    print("Successfully initialized new provider system with centralized configuration")
                    return
                else:
                    print("RAG is disabled in configuration, falling back to legacy initialization")
                
            except Exception as e:
                print(f"New provider system initialization with config failed: {e}")
                print("Falling back to environment-based initialization")
                
                # Try environment-based initialization as fallback
                try:
                    if embedding_provider is None:
                        self.embedding_provider = create_embedding_provider()
                        print(f"Created embedding provider from environment: {self.embedding_provider.config.provider}")
                    else:
                        self.embedding_provider = embedding_provider
                        print(f"Using provided embedding provider: {self.embedding_provider.config.provider}")
                    
                    if vector_store_provider is None:
                        self.vector_store_provider = create_vector_store(
                            embedding_provider=self.embedding_provider
                        )
                        print(f"Created vector store provider from environment: {self.vector_store_provider.config.provider}")
                    else:
                        self.vector_store_provider = vector_store_provider
                        print(f"Using provided vector store provider: {self.vector_store_provider.config.provider}")
                    
                    # Set the vector store to the new provider (for backward compatibility)
                    self.vector_store = self.vector_store_provider
                    
                    print("Successfully initialized new provider system from environment")
                    return
                    
                except Exception as env_error:
                    print(f"Environment-based provider system initialization failed: {env_error}")
                    print("Falling back to legacy vector store initialization")
        
        # Fallback to legacy vector store initialization
        print("Using legacy vector store initialization")
        self.vector_store = RAGVectorStore(
            chroma_db_path=chroma_db_path,
            manim_docs_path=self.manim_docs_path,
            embedding_model=embedding_model,
            session_id=self.session_id,
            use_langfuse=use_langfuse,
            helper_model=self.helper_model
        )

    def _initialize_enhanced_components(self):
        """Initialize enhanced RAG components if available and enabled."""
        if not ENHANCED_COMPONENTS_AVAILABLE or not self.config.use_enhanced_components:
            print("Enhanced components not available or disabled, using legacy functionality")
            self.enhanced_query_generator = None
            self.context_aware_retriever = None
            self.semantic_expansion_engine = None
            self.performance_cache = None
            self.error_handler = None
            self.quality_monitor = None
            self.enhanced_plugin_detector = None
            self.feedback_collector = None
            self.quality_evaluator = None
            return

        try:
            # Initialize performance cache
            if self.config.enable_caching:
                cache_config = CacheConfig(
                    ttl=self.config.cache_ttl,
                    max_size=self.config.max_cache_size
                )
                self.performance_cache = PerformanceOptimizedCache(cache_config)
            else:
                self.performance_cache = None

            # Initialize error handler
            if self.config.enable_error_handling:
                error_config = ErrorHandlingConfig()
                self.error_handler = RobustErrorHandler(error_config)
            else:
                self.error_handler = None

            # Initialize intelligent query generator
            self.enhanced_query_generator = IntelligentQueryGenerator(
                llm_model=self.helper_model,
                session_id=self.session_id
            )

            # Initialize context-aware retriever
            self.context_aware_retriever = ContextAwareRetriever(
                vector_store=self.vector_store,
                cache=self.performance_cache
            )

            # Initialize semantic expansion engine
            self.semantic_expansion_engine = SemanticQueryExpansionEngine(
                llm_model=self.helper_model
            )

            # Initialize enhanced plugin detector
            self.enhanced_plugin_detector = ContextAwarePluginDetector(
                manim_docs_path=self.manim_docs_path
            )

            # Initialize quality monitoring components
            if self.config.enable_quality_monitoring:
                self.quality_monitor = QualityMonitor(
                    threshold=self.config.quality_threshold
                )
                self.feedback_collector = FeedbackCollector()
                self.quality_evaluator = RAGQualityEvaluator()
            else:
                self.quality_monitor = None
                self.feedback_collector = None
                self.quality_evaluator = None

            print("Enhanced RAG components initialized successfully")

        except Exception as e:
            print(f"Error initializing enhanced components: {e}")
            # Fall back to None for all components
            self.enhanced_query_generator = None
            self.context_aware_retriever = None
            self.semantic_expansion_engine = None
            self.performance_cache = None
            self.error_handler = None
            self.quality_monitor = None
            self.enhanced_plugin_detector = None
            self.feedback_collector = None
            self.quality_evaluator = None

    def set_relevant_plugins(self, plugins: List[str]) -> None:
        """Set the relevant plugins for the current video.

        Args:
            plugins (List[str]): List of plugin names to set as relevant
        """
        self.relevant_plugins = plugins

    def detect_relevant_plugins(self, topic: str, description: str) -> List[str]:
        """Detect which plugins might be relevant based on topic and description.

        Args:
            topic (str): Topic of the video
            description (str): Description of the video content

        Returns:
            List[str]: List of detected relevant plugin names
        """
        # Load plugin descriptions
        plugins = self._load_plugin_descriptions()
        if not plugins:
            return []

        # Get formatted prompt using the task_generator function
        prompt = get_prompt_detect_plugins(
            topic=topic,
            description=description,
            plugin_descriptions=json.dumps([{'name': p['name'], 'description': p['description']} for p in plugins], indent=2)
        )

        try:
            response = self.helper_model(
                _prepare_text_inputs(prompt),
                metadata={"generation_name": "detect-relevant-plugins", "tags": [topic, "plugin-detection"], "session_id": self.session_id}
            )            # Clean the response to ensure it only contains the JSON array
            json_match = re.search(r'```json(.*)```', response, re.DOTALL)
            if not json_match:
                print(f"No JSON block found in plugin detection response: {response[:200]}...")
                return []
            response = json_match.group(1)
            try:
                relevant_plugins = json.loads(response)
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError when parsing relevant plugins: {e}")
                print(f"Response text was: {response}")
                return []

            print(f"LLM detected relevant plugins: {relevant_plugins}")
            return relevant_plugins
        except Exception as e:
            print(f"Error detecting plugins with LLM: {e}")
            return []

    def _load_plugin_descriptions(self) -> list:
        """Load plugin descriptions from JSON file.

        Returns:
            list: List of plugin descriptions, empty list if loading fails
        """
        try:
            plugin_config_path = os.path.join(
                self.manim_docs_path,
                "plugin_docs",
                "plugins.json"
            )
            if os.path.exists(plugin_config_path):
                with open(plugin_config_path, "r") as f:
                    return json.load(f)
            else:
                print(f"Plugin descriptions file not found at {plugin_config_path}")
                return []
        except Exception as e:
            print(f"Error loading plugin descriptions: {e}")
            return []

    def _generate_rag_queries_storyboard(self, scene_plan: str, scene_trace_id: str = None, topic: str = None, scene_number: int = None, session_id: str = None, relevant_plugins: List[str] = []) -> List[str]:
        """Generate RAG queries from the scene plan to help create storyboard.

        Args:
            scene_plan (str): Scene plan text to generate queries from
            scene_trace_id (str, optional): Trace identifier for the scene. Defaults to None
            topic (str, optional): Topic name. Defaults to None
            scene_number (int, optional): Scene number. Defaults to None
            session_id (str, optional): Session identifier. Defaults to None
            relevant_plugins (List[str], optional): List of relevant plugins. Defaults to empty list

        Returns:
            List[str]: List of generated RAG queries
        """
        # Use enhanced query generation if available
        if self.enhanced_query_generator:
            try:
                input_context = InputContext(
                    task_type=TaskType.STORYBOARD_CREATION,
                    content=scene_plan,
                    relevant_plugins=relevant_plugins,
                    topic=topic,
                    scene_number=scene_number
                )
                enhanced_queries = self.enhanced_query_generator.generate_queries(input_context)
                queries = [q.query_text for q in enhanced_queries]
                
                # Cache the enhanced queries
                cache_dir = os.path.join(self.output_dir, re.sub(r'[^a-z0-9_]+', '_', topic.lower()), f"scene{scene_number}", "rag_cache")
                os.makedirs(cache_dir, exist_ok=True)
                cache_file = os.path.join(cache_dir, "rag_queries_storyboard.json")
                with open(cache_file, 'w') as f:
                    json.dump(queries, f)
                
                return queries
            except Exception as e:
                print(f"Enhanced query generation failed, falling back to legacy: {e}")
        
        # Legacy implementation
        cache_key = f"{topic}_scene{scene_number}_storyboard_rag"
        cache_dir = os.path.join(self.output_dir, re.sub(r'[^a-z0-9_]+', '_', topic.lower()), f"scene{scene_number}", "rag_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "rag_queries_storyboard.json")

        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)

        # Format relevant plugins as a string
        plugins_str = ", ".join(relevant_plugins) if relevant_plugins else "No plugins are relevant."
        
        # Generate the prompt with only the required arguments
        prompt = get_prompt_rag_query_generation_vision_storyboard(
            scene_plan=scene_plan,
            relevant_plugins=plugins_str
        )
        queries = self.helper_model(
            _prepare_text_inputs(prompt),
            metadata={"generation_name": "rag_query_generation_storyboard", "trace_id": scene_trace_id, "tags": [topic, f"scene{scene_number}"], "session_id": session_id}
        )
        
        # retreive json triple backticks
        
        try: # add try-except block to handle potential json decode errors
            json_match = re.search(r'```json(.*)```', queries, re.DOTALL)
            if not json_match:
                print(f"No JSON block found in storyboard RAG queries response: {queries[:200]}...")
                return []
            queries = json_match.group(1)
            queries = json.loads(queries)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError when parsing RAG queries for storyboard: {e}")
            print(f"Response text was: {queries}")
            return [] # Return empty list in case of parsing error

        # Cache the queries
        with open(cache_file, 'w') as f:
            json.dump(queries, f)

        return queries

    def _generate_rag_queries_technical(self, storyboard: str, scene_trace_id: str = None, topic: str = None, scene_number: int = None, session_id: str = None, relevant_plugins: List[str] = []) -> List[str]:
        """Generate RAG queries from the storyboard to help create technical implementation.

        Args:
            storyboard (str): Storyboard text to generate queries from
            scene_trace_id (str, optional): Trace identifier for the scene. Defaults to None
            topic (str, optional): Topic name. Defaults to None
            scene_number (int, optional): Scene number. Defaults to None
            session_id (str, optional): Session identifier. Defaults to None
            relevant_plugins (List[str], optional): List of relevant plugins. Defaults to empty list

        Returns:
            List[str]: List of generated RAG queries
        """
        # Use enhanced query generation if available
        if self.enhanced_query_generator:
            try:
                input_context = InputContext(
                    task_type=TaskType.TECHNICAL_IMPLEMENTATION,
                    content=storyboard,
                    relevant_plugins=relevant_plugins,
                    topic=topic,
                    scene_number=scene_number
                )
                enhanced_queries = self.enhanced_query_generator.generate_queries(input_context)
                queries = [q.query_text for q in enhanced_queries]
                
                # Cache the enhanced queries
                cache_dir = os.path.join(self.output_dir, re.sub(r'[^a-z0-9_]+', '_', topic.lower()), f"scene{scene_number}", "rag_cache")
                os.makedirs(cache_dir, exist_ok=True)
                cache_file = os.path.join(cache_dir, "rag_queries_technical.json")
                with open(cache_file, 'w') as f:
                    json.dump(queries, f)
                
                return queries
            except Exception as e:
                print(f"Enhanced query generation failed, falling back to legacy: {e}")
        
        # Legacy implementation
        cache_key = f"{topic}_scene{scene_number}_technical_rag"
        cache_dir = os.path.join(self.output_dir, re.sub(r'[^a-z0-9_]+', '_', topic.lower()), f"scene{scene_number}", "rag_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "rag_queries_technical.json")

        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)        
        prompt = get_prompt_rag_query_generation_technical(
            storyboard=storyboard,
            relevant_plugins=", ".join(relevant_plugins) if relevant_plugins else "No plugins are relevant."
        )
        
        queries = self.helper_model(
            _prepare_text_inputs(prompt),
            metadata={"generation_name": "rag_query_generation_technical", "trace_id": scene_trace_id, "tags": [topic, f"scene{scene_number}"], "session_id": session_id}
        )

        try: # add try-except block to handle potential json decode errors
            json_match = re.search(r'```json(.*)```', queries, re.DOTALL)
            if not json_match:
                print(f"No JSON block found in technical RAG queries response: {queries[:200]}...")
                return []
            queries = json_match.group(1)
            queries = json.loads(queries)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError when parsing RAG queries for technical implementation: {e}")
            print(f"Response text was: {queries}")
            return [] # Return empty list in case of parsing error

        # Cache the queries
        with open(cache_file, 'w') as f:
            json.dump(queries, f)

        return queries

    def _generate_rag_queries_narration(self, storyboard: str, scene_trace_id: str = None, topic: str = None, scene_number: int = None, session_id: str = None, relevant_plugins: List[str] = []) -> List[str]:
        """Generate RAG queries from the storyboard to help create narration plan.

        Args:
            storyboard (str): Storyboard text to generate queries from
            scene_trace_id (str, optional): Trace identifier for the scene. Defaults to None
            topic (str, optional): Topic name. Defaults to None
            scene_number (int, optional): Scene number. Defaults to None
            session_id (str, optional): Session identifier. Defaults to None
            relevant_plugins (List[str], optional): List of relevant plugins. Defaults to empty list

        Returns:
            List[str]: List of generated RAG queries
        """
        # Use enhanced query generation if available
        if self.enhanced_query_generator:
            try:
                input_context = InputContext(
                    task_type=TaskType.NARRATION,
                    content=storyboard,
                    relevant_plugins=relevant_plugins,
                    topic=topic,
                    scene_number=scene_number
                )
                enhanced_queries = self.enhanced_query_generator.generate_queries(input_context)
                queries = [q.query_text for q in enhanced_queries]
                
                # Cache the enhanced queries
                cache_dir = os.path.join(self.output_dir, re.sub(r'[^a-z0-9_]+', '_', topic.lower()), f"scene{scene_number}", "rag_cache")
                os.makedirs(cache_dir, exist_ok=True)
                cache_file = os.path.join(cache_dir, "rag_queries_narration.json")
                with open(cache_file, 'w') as f:
                    json.dump(queries, f)
                
                return queries
            except Exception as e:
                print(f"Enhanced query generation failed, falling back to legacy: {e}")
        
        # Legacy implementation
        cache_key = f"{topic}_scene{scene_number}_narration_rag"
        cache_dir = os.path.join(self.output_dir, re.sub(r'[^a-z0-9_]+', '_', topic.lower()), f"scene{scene_number}", "rag_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "rag_queries_narration.json")

        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
                
        prompt = get_prompt_rag_query_generation_narration(
            storyboard=storyboard,
            relevant_plugins=", ".join(relevant_plugins) if relevant_plugins else "No plugins are relevant."
        )
        
        queries = self.helper_model(
            _prepare_text_inputs(prompt),
            metadata={"generation_name": "rag_query_generation_narration", "trace_id": scene_trace_id, "tags": [topic, f"scene{scene_number}"], "session_id": session_id}
        )

        try: # add try-except block to handle potential json decode errors
            json_match = re.search(r'```json(.*)```', queries, re.DOTALL)
            if not json_match:
                print(f"No JSON block found in narration RAG queries response: {queries[:200]}...")
                return []
            queries = json_match.group(1)
            queries = json.loads(queries)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError when parsing narration RAG queries: {e}")
            print(f"Response text was: {queries}")
            return [] # Return empty list in case of parsing error

        # Cache the queries
        with open(cache_file, 'w') as f:
            json.dump(queries, f)

        return queries

    def get_relevant_docs(self, rag_queries: List[Dict], scene_trace_id: str, topic: str, scene_number: int) -> List[str]:
        """Get relevant documentation using the vector store.

        Args:
            rag_queries (List[Dict]): List of RAG queries to search for
            scene_trace_id (str): Trace identifier for the scene
            topic (str): Topic name
            scene_number (int): Scene number

        Returns:
            List[str]: List of relevant documentation snippets
        """
        # Use enhanced context-aware retrieval if available
        if self.context_aware_retriever:
            try:
                # Convert queries to enhanced format
                enhanced_queries = []
                for query in rag_queries:
                    if isinstance(query, dict):
                        query_text = query.get('query', str(query))
                    else:
                        query_text = str(query)
                    
                    enhanced_query = EnhancedQuery(
                        query_text=query_text,
                        intent=QueryIntent.API_REFERENCE,  # Default intent
                        complexity_level=ComplexityLevel.MODERATE,
                        relevant_plugins=self.relevant_plugins or []
                    )
                    enhanced_queries.append(enhanced_query)
                
                # Create retrieval context
                retrieval_context = RetrievalContext(
                    task_type=RetrievalTaskType.IMPLEMENTATION,
                    query_intent=QueryIntent.API_REFERENCE,
                    complexity_level=ComplexityLevel.MODERATE,
                    relevant_plugins=self.relevant_plugins or []
                )
                
                # Retrieve with context awareness
                ranked_results = self.context_aware_retriever.retrieve_documents(
                    enhanced_queries, retrieval_context
                )
                
                # Extract content from ranked results
                docs = [result.chunk.content for result in ranked_results]
                
                # Log quality metrics if available
                if self.quality_monitor:
                    self.quality_monitor.log_retrieval_event(
                        queries=rag_queries,
                        results=ranked_results,
                        context=retrieval_context
                    )
                
                return docs
                
            except Exception as e:
                print(f"Enhanced retrieval failed, falling back to legacy: {e}")
                if self.error_handler:
                    self.error_handler.handle_retrieval_error(e)
        
        # Legacy implementation
        return self.vector_store.find_relevant_docs(
            queries=rag_queries,
            k=2,
            trace_id=scene_trace_id,
            topic=topic,
            scene_number=scene_number
        )
    
    def _generate_rag_queries_code(self, implementation_plan: str, scene_trace_id: str = None, topic: str = None, scene_number: int = None, relevant_plugins: List[str] = None) -> List[str]:
        """Generate RAG queries from implementation plan.

        Args:
            implementation_plan (str): Implementation plan text to generate queries from
            scene_trace_id (str, optional): Trace identifier for the scene. Defaults to None
            topic (str, optional): Topic name. Defaults to None
            scene_number (int, optional): Scene number. Defaults to None
            relevant_plugins (List[str], optional): List of relevant plugins. Defaults to None

        Returns:
            List[str]: List of generated RAG queries
        """
        # Use enhanced query generation if available
        if self.enhanced_query_generator:
            try:
                input_context = InputContext(
                    task_type=TaskType.IMPLEMENTATION_PLAN,
                    content=implementation_plan,
                    relevant_plugins=relevant_plugins or [],
                    topic=topic,
                    scene_number=scene_number
                )
                enhanced_queries = self.enhanced_query_generator.generate_queries(input_context)
                queries = [q.query_text for q in enhanced_queries]
                
                # Cache the enhanced queries
                cache_dir = os.path.join(self.output_dir, re.sub(r'[^a-z0-9_]+', '_', topic.lower()), f"scene{scene_number}", "rag_cache")
                os.makedirs(cache_dir, exist_ok=True)
                cache_file = os.path.join(cache_dir, "rag_queries_code.json")
                with open(cache_file, 'w') as f:
                    json.dump(queries, f)
                
                return queries
            except Exception as e:
                print(f"Enhanced query generation failed, falling back to legacy: {e}")
        
        # Legacy implementation
        cache_key = f"{topic}_scene{scene_number}"
        cache_dir = os.path.join(self.output_dir, re.sub(r'[^a-z0-9_]+', '_', topic.lower()), f"scene{scene_number}", "rag_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "rag_queries_code.json")

        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)

        prompt = get_prompt_rag_query_generation_code(
            implementation_plan=implementation_plan,
            relevant_plugins=", ".join(relevant_plugins) if relevant_plugins else "No plugins are relevant."
        )
        
        try:
            response = self.helper_model(
                _prepare_text_inputs(prompt),
                metadata={"generation_name": "rag_query_generation_code", "trace_id": scene_trace_id, "tags": [topic, f"scene{scene_number}"], "session_id": self.session_id}
            )
            
            # Clean and parse response
            json_match = re.search(r'```json(.*)```', response, re.DOTALL)
            if not json_match:
                print(f"No JSON block found in code RAG queries response: {response[:200]}...")
                return []
            response = json_match.group(1)
            queries = json.loads(response)

            # Cache the queries
            with open(cache_file, 'w') as f:
                json.dump(queries, f)

            return queries
        except Exception as e:
            print(f"Error generating RAG queries: {e}")
            return []

    def _generate_rag_queries_error_fix(self, error: str, code: str, scene_trace_id: str = None, topic: str = None, scene_number: int = None, session_id: str = None) -> List[str]:
        """Generate RAG queries for fixing code errors.

        Args:
            error (str): Error message to generate queries from
            code (str): Code containing the error
            scene_trace_id (str, optional): Trace identifier for the scene. Defaults to None
            topic (str, optional): Topic name. Defaults to None
            scene_number (int, optional): Scene number. Defaults to None
            session_id (str, optional): Session identifier. Defaults to None

        Returns:
            List[str]: List of generated RAG queries
        """
        # Use enhanced query generation if available
        if self.enhanced_query_generator:
            try:
                input_context = InputContext(
                    task_type=TaskType.ERROR_FIXING,
                    content=f"Error: {error}\nCode: {code}",
                    relevant_plugins=self.relevant_plugins or [],
                    topic=topic,
                    scene_number=scene_number,
                    error_message=error
                )
                enhanced_queries = self.enhanced_query_generator.generate_queries(input_context)
                queries = [q.query_text for q in enhanced_queries]
                
                # Cache the enhanced queries
                cache_dir = os.path.join(self.output_dir, re.sub(r'[^a-z0-9_]+', '_', topic.lower()), f"scene{scene_number}", "rag_cache")
                os.makedirs(cache_dir, exist_ok=True)
                cache_file = os.path.join(cache_dir, "rag_queries_error_fix.json")
                with open(cache_file, 'w') as f:
                    json.dump(queries, f)
                
                return queries
            except Exception as e:
                print(f"Enhanced query generation failed, falling back to legacy: {e}")
        
        # Legacy implementation
        if self.relevant_plugins is None:
            print("Warning: No plugins have been detected yet")
            plugins_str = "No plugins are relevant."
        else:
            plugins_str = ", ".join(self.relevant_plugins) if self.relevant_plugins else "No plugins are relevant."

        cache_key = f"{topic}_scene{scene_number}_error_fix"
        cache_dir = os.path.join(self.output_dir, re.sub(r'[^a-z0-9_]+', '_', topic.lower()), f"scene{scene_number}", "rag_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "rag_queries_error_fix.json")

        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cached_queries = json.load(f)
                print(f"Using cached RAG queries for error fix in {cache_key}")
                return cached_queries

        prompt = get_prompt_rag_query_generation_fix_error(
            error=error, 
            code=code, 
            relevant_plugins=plugins_str
        )

        queries = self.helper_model(
            _prepare_text_inputs(prompt),
            metadata={"generation_name": "rag-query-generation-fix-error", "trace_id": scene_trace_id, "tags": [topic, f"scene{scene_number}"], "session_id": session_id}
        )

        try:  
            # retrieve json triple backticks
            json_match = re.search(r'```json(.*)```', queries, re.DOTALL)
            if not json_match:
                print(f"No JSON block found in error fix RAG queries response: {queries[:200]}...")
                return []
            queries = json_match.group(1)
            queries = json.loads(queries)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError when parsing RAG queries for error fix: {e}")
            print(f"Response text was: {queries}")
            return []

        # Cache the queries
        with open(cache_file, 'w') as f:
            json.dump(queries, f)

        return queries

    def get_enhanced_retrieval_results(self, queries: List[str], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get enhanced retrieval results with full metadata and scoring.
        
        Args:
            queries (List[str]): List of query strings
            context (Dict[str, Any]): Context information for retrieval
            
        Returns:
            List[Dict[str, Any]]: Enhanced results with metadata and scores
        """
        if not self.context_aware_retriever:
            # Fallback to basic retrieval
            basic_results = self.vector_store.find_relevant_docs(
                queries=queries, k=5
            )
            return [{"content": doc, "score": 1.0, "metadata": {}} for doc in basic_results]
        
        try:
            # Convert to enhanced queries
            enhanced_queries = []
            for query in queries:
                enhanced_query = EnhancedQuery(
                    query_text=query,
                    intent=QueryIntent.API_REFERENCE,
                    complexity_level=ComplexityLevel.MODERATE,
                    relevant_plugins=self.relevant_plugins or []
                )
                enhanced_queries.append(enhanced_query)
            
            # Create retrieval context
            retrieval_context = RetrievalContext(
                task_type=RetrievalTaskType.IMPLEMENTATION,
                query_intent=QueryIntent.API_REFERENCE,
                complexity_level=ComplexityLevel.MODERATE,
                relevant_plugins=self.relevant_plugins or [],
                **context
            )
            
            # Get enhanced results
            ranked_results = self.context_aware_retriever.retrieve_documents(
                enhanced_queries, retrieval_context
            )
            
            # Convert to dictionary format
            enhanced_results = []
            for result in ranked_results:
                enhanced_results.append({
                    "content": result.chunk.content,
                    "similarity_score": result.similarity_score,
                    "context_score": result.context_score,
                    "final_score": result.final_score,
                    "result_type": result.result_type.value if hasattr(result.result_type, 'value') else str(result.result_type),
                    "metadata": result.metadata.__dict__ if hasattr(result.metadata, '__dict__') else {}
                })
            
            return enhanced_results
            
        except Exception as e:
            print(f"Enhanced retrieval failed: {e}")
            if self.error_handler:
                self.error_handler.handle_retrieval_error(e)
            return []

    def evaluate_retrieval_quality(self, queries: List[str], expected_results: List[str] = None) -> Dict[str, float]:
        """Evaluate the quality of retrieval results.
        
        Args:
            queries (List[str]): List of queries to evaluate
            expected_results (List[str], optional): Expected results for evaluation
            
        Returns:
            Dict[str, float]: Quality metrics
        """
        if not self.quality_evaluator:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        try:
            # Get retrieval results
            results = self.get_enhanced_retrieval_results(queries, {})
            
            # Evaluate quality
            metrics = self.quality_evaluator.evaluate_retrieval_quality(
                queries=queries,
                results=results,
                expected_results=expected_results or []
            )
            
            return {
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1_score,
                "ndcg": getattr(metrics, 'ndcg', 0.0)
            }
            
        except Exception as e:
            print(f"Quality evaluation failed: {e}")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def collect_feedback(self, query: str, results: List[str], feedback: Dict[str, Any]) -> None:
        """Collect user feedback on retrieval results.
        
        Args:
            query (str): The original query
            results (List[str]): Retrieved results
            feedback (Dict[str, Any]): User feedback data
        """
        if self.feedback_collector:
            try:
                self.feedback_collector.collect_feedback(
                    query=query,
                    results=results,
                    feedback=feedback
                )
            except Exception as e:
                print(f"Feedback collection failed: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics.
        
        Returns:
            Dict[str, Any]: Performance metrics
        """
        metrics = {
            "cache_hit_rate": 0.0,
            "average_response_time": 0.0,
            "total_queries": 0,
            "error_rate": 0.0
        }
        
        if self.performance_cache:
            try:
                cache_stats = self.performance_cache.get_cache_stats()
                metrics.update(cache_stats)
            except Exception as e:
                print(f"Failed to get cache metrics: {e}")
        
        if self.quality_monitor:
            try:
                quality_stats = self.quality_monitor.get_performance_stats()
                metrics.update(quality_stats)
            except Exception as e:
                print(f"Failed to get quality metrics: {e}")
        
        return metrics

    def update_configuration(self, new_config: RAGConfig) -> None:
        """Update RAG configuration and reinitialize components if needed.
        
        Args:
            new_config (RAGConfig): New configuration
        """
        old_config = self.config
        self.config = new_config
        
        # Reinitialize if enhanced components setting changed
        if old_config.use_enhanced_components != new_config.use_enhanced_components:
            self._initialize_enhanced_components()
        
        # Update individual component configurations
        if self.performance_cache and old_config.cache_ttl != new_config.cache_ttl:
            try:
                self.performance_cache.update_ttl(new_config.cache_ttl)
            except Exception as e:
                print(f"Failed to update cache TTL: {e}")
        
        if self.quality_monitor and old_config.quality_threshold != new_config.quality_threshold:
            try:
                self.quality_monitor.update_threshold(new_config.quality_threshold)
            except Exception as e:
                print(f"Failed to update quality threshold: {e}")

    def clear_cache(self) -> None:
        """Clear all cached data."""
        if self.performance_cache:
            try:
                self.performance_cache.clear_all_caches()
                print("Cache cleared successfully")
            except Exception as e:
                print(f"Failed to clear cache: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status and health.
        
        Returns:
            Dict[str, Any]: System status information
        """
        status = {
            "enhanced_components_available": ENHANCED_COMPONENTS_AVAILABLE,
            "enhanced_components_enabled": self.config.use_enhanced_components,
            "components_status": {
                "query_generator": self.enhanced_query_generator is not None,
                "context_retriever": self.context_aware_retriever is not None,
                "semantic_expansion": self.semantic_expansion_engine is not None,
                "performance_cache": self.performance_cache is not None,
                "error_handler": self.error_handler is not None,
                "quality_monitor": self.quality_monitor is not None,
                "plugin_detector": self.enhanced_plugin_detector is not None,
                "feedback_collector": self.feedback_collector is not None,
                "quality_evaluator": self.quality_evaluator is not None
            },
            "configuration": {
                "caching_enabled": self.config.enable_caching,
                "quality_monitoring_enabled": self.config.enable_quality_monitoring,
                "error_handling_enabled": self.config.enable_error_handling,
                "cache_ttl": self.config.cache_ttl,
                "performance_threshold": self.config.performance_threshold,
                "quality_threshold": self.config.quality_threshold
            }
        }
        
        # Add performance metrics if available
        try:
            status["performance_metrics"] = self.get_performance_metrics()
        except Exception as e:
            status["performance_metrics"] = {"error": str(e)}
        
        return status
    
    # Provider Management Features
    
    def switch_embedding_provider(self, provider_name: str) -> bool:
        """Switch to a different embedding provider.
        
        Args:
            provider_name (str): Name of the provider to switch to ("jina", "local")
            
        Returns:
            bool: True if switch was successful, False otherwise
        """
        if not NEW_PROVIDER_SYSTEM_AVAILABLE:
            print("Provider switching requires the new provider system")
            return False
        
        try:
            # Create new embedding provider
            if provider_name == "jina":
                config = ConfigurationManager._load_jina_config()
            elif provider_name == "local":
                config = ConfigurationManager._load_local_config()
            else:
                print(f"Unknown provider: {provider_name}")
                return False
            
            new_embedding_provider = EmbeddingProviderFactory.create_provider(config)
            
            # Create new vector store with the new embedding provider
            new_vector_store = create_vector_store(
                embedding_provider=new_embedding_provider
            )
            
            # Update the providers
            old_embedding_provider = self.embedding_provider
            old_vector_store = self.vector_store_provider
            
            self.embedding_provider = new_embedding_provider
            self.vector_store_provider = new_vector_store
            self.vector_store = new_vector_store
            
            # Reinitialize enhanced components with new vector store
            if self.context_aware_retriever:
                self.context_aware_retriever = ContextAwareRetriever(
                    vector_store=self.vector_store,
                    cache=self.performance_cache
                )
            
            print(f"Successfully switched to {provider_name} embedding provider")
            print(f"Embedding dimensions: {self.embedding_provider.get_embedding_dimension()}")
            print(f"Vector store provider: {self.vector_store_provider.config.provider}")
            
            return True
            
        except Exception as e:
            print(f"Failed to switch embedding provider: {e}")
            return False
    
    def get_provider_status(self) -> Dict[str, Any]:
        """Get detailed status of all providers.
        
        Returns:
            Dict[str, Any]: Provider status information
        """
        status = {
            "new_provider_system_available": NEW_PROVIDER_SYSTEM_AVAILABLE,
            "embedding_provider": None,
            "vector_store_provider": None,
            "legacy_vector_store": None
        }
        
        # Embedding provider status
        if self.embedding_provider:
            try:
                provider_info = self.embedding_provider.get_provider_info()
                status["embedding_provider"] = {
                    "provider": provider_info.get("provider", "unknown"),
                    "model": provider_info.get("model", "unknown"),
                    "dimensions": provider_info.get("dimensions", 0),
                    "available": self.embedding_provider.is_available(),
                    "config": {
                        "provider": self.embedding_provider.config.provider,
                        "model_name": self.embedding_provider.config.model_name,
                        "dimensions": self.embedding_provider.config.dimensions,
                        "batch_size": self.embedding_provider.config.batch_size
                    }
                }
            except Exception as e:
                status["embedding_provider"] = {"error": str(e)}
        
        # Vector store provider status
        if self.vector_store_provider:
            try:
                if hasattr(self.vector_store_provider, 'health_check'):
                    health = self.vector_store_provider.health_check()
                    status["vector_store_provider"] = {
                        "provider": self.vector_store_provider.config.provider,
                        "collection_name": self.vector_store_provider.config.collection_name,
                        "embedding_dimension": self.vector_store_provider.config.embedding_dimension,
                        "distance_metric": self.vector_store_provider.config.distance_metric,
                        "health": health,
                        "available": self.vector_store_provider.is_available()
                    }
                else:
                    status["vector_store_provider"] = {
                        "provider": "legacy",
                        "available": True
                    }
            except Exception as e:
                status["vector_store_provider"] = {"error": str(e)}
        
        # Legacy vector store status
        if hasattr(self.vector_store, 'embedding_model'):
            status["legacy_vector_store"] = {
                "type": "EnhancedRAGVectorStore",
                "embedding_model": getattr(self.vector_store, 'embedding_model', 'unknown'),
                "chroma_db_path": getattr(self.vector_store, 'chroma_db_path', 'unknown')
            }
        
        return status
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate current configuration and provide recommendations.
        
        Returns:
            Dict[str, Any]: Validation results and recommendations
        """
        validation = {
            "overall_status": "unknown",
            "issues": [],
            "warnings": [],
            "recommendations": [],
            "provider_validation": {}
        }
        
        try:
            # Validate embedding provider
            if self.embedding_provider:
                if not self.embedding_provider.is_available():
                    validation["issues"].append("Embedding provider is not available")
                    validation["recommendations"].append("Check embedding provider configuration and dependencies")
                else:
                    validation["provider_validation"]["embedding"] = "healthy"
            else:
                validation["warnings"].append("Using legacy vector store without new embedding provider")
                validation["recommendations"].append("Consider upgrading to the new provider system")
            
            # Validate vector store provider
            if self.vector_store_provider:
                if hasattr(self.vector_store_provider, 'is_available'):
                    if not self.vector_store_provider.is_available():
                        validation["issues"].append("Vector store provider is not available")
                        validation["recommendations"].append("Check vector store configuration and connectivity")
                    else:
                        validation["provider_validation"]["vector_store"] = "healthy"
            
            # Check dimension compatibility
            if self.embedding_provider and self.vector_store_provider:
                embedding_dim = self.embedding_provider.get_embedding_dimension()
                vector_store_dim = self.vector_store_provider.config.embedding_dimension
                
                if embedding_dim != vector_store_dim:
                    validation["issues"].append(
                        f"Dimension mismatch: embedding provider {embedding_dim}d vs vector store {vector_store_dim}d"
                    )
                    validation["recommendations"].append(
                        "Ensure embedding and vector store dimensions match or re-index documents"
                    )
            
            # Determine overall status
            if validation["issues"]:
                validation["overall_status"] = "error"
            elif validation["warnings"]:
                validation["overall_status"] = "warning"
            else:
                validation["overall_status"] = "healthy"
            
            # Add environment variable validation if new system is available
            if NEW_PROVIDER_SYSTEM_AVAILABLE:
                try:
                    env_validation = ConfigurationManager.validate_environment_variables()
                    if env_validation:
                        validation["environment_issues"] = env_validation
                        for var, issue in env_validation.items():
                            validation["recommendations"].append(f"Fix {var}: {issue}")
                except Exception as e:
                    validation["warnings"].append(f"Environment validation failed: {e}")
            
        except Exception as e:
            validation["issues"].append(f"Configuration validation failed: {e}")
            validation["overall_status"] = "error"
        
        return validation
    
    def get_provider_comparison(self) -> Dict[str, Any]:
        """Get comparison of available providers.
        
        Returns:
            Dict[str, Any]: Provider comparison information
        """
        comparison = {
            "current_providers": {},
            "available_providers": {},
            "recommendations": {}
        }
        
        # Current provider information
        provider_status = self.get_provider_status()
        comparison["current_providers"] = provider_status
        
        # Available provider information
        if NEW_PROVIDER_SYSTEM_AVAILABLE:
            try:
                factory = EmbeddingProviderFactory()
                available_providers = factory.get_available_providers()
                
                for provider_info in available_providers:
                    provider_name = provider_info["name"]
                    comparison["available_providers"][provider_name] = {
                        "available": provider_info["available"],
                        "description": provider_info["description"],
                        "configured": provider_info.get("configured", False)
                    }
                    
                    if provider_info.get("info"):
                        comparison["available_providers"][provider_name].update(provider_info["info"])
                
                # Generate recommendations
                current_embedding = provider_status.get("embedding_provider", {}).get("provider")
                
                if current_embedding == "local":
                    comparison["recommendations"]["upgrade_to_jina"] = (
                        "Consider upgrading to JINA for better performance and latest models"
                    )
                elif current_embedding == "jina":
                    comparison["recommendations"]["local_fallback"] = (
                        "Local provider available as fallback for offline usage"
                    )
                
            except Exception as e:
                comparison["available_providers"]["error"] = str(e)
        
        return comparison
    
    def get_debugging_info(self) -> Dict[str, Any]:
        """Get comprehensive debugging information.
        
        Returns:
            Dict[str, Any]: Debugging information
        """
        debug_info = {
            "system_status": self.get_system_status(),
            "provider_status": self.get_provider_status(),
            "configuration_validation": self.validate_configuration(),
            "performance_metrics": self.get_performance_metrics(),
            "component_versions": {
                "new_provider_system": NEW_PROVIDER_SYSTEM_AVAILABLE,
                "enhanced_components": ENHANCED_COMPONENTS_AVAILABLE
            },
            "recent_errors": []
        }
        
        # Add error handler information if available
        if self.error_handler:
            try:
                debug_info["recent_errors"] = self.error_handler.get_recent_errors()
            except Exception as e:
                debug_info["recent_errors"] = [f"Error handler access failed: {e}"]
        
        # Add cache information if available
        if self.performance_cache:
            try:
                cache_stats = self.performance_cache.get_cache_stats()
                debug_info["cache_status"] = cache_stats
            except Exception as e:
                debug_info["cache_status"] = {"error": str(e)}
        
        return debug_info
