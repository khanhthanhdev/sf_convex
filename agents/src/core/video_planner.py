import os
import re
import json
import glob
from typing import List, Optional, Dict, Any
import uuid
import asyncio

from mllm_tools.utils import _prepare_text_inputs
from src.utils.utils import extract_xml
from task_generator import (
    get_prompt_scene_plan,
    get_prompt_scene_vision_storyboard,
    get_prompt_scene_technical_implementation,
    get_prompt_scene_animation_narration,
    get_prompt_context_learning_scene_plan,
    get_prompt_context_learning_vision_storyboard,
    get_prompt_context_learning_technical_implementation,
    get_prompt_context_learning_animation_narration,
    get_prompt_context_learning_code
)
from src.rag.rag_integration import RAGIntegration

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

class VideoPlanner:
    """A class for planning and generating video content.

    This class handles the planning and generation of video content including scene outlines,
    vision storyboards, technical implementations, and animation narrations.

    Args:
        planner_model: The model used for planning tasks
        helper_model: Optional helper model, defaults to planner_model if None
        output_dir (str): Directory for output files. Defaults to "output"
        print_response (bool): Whether to print model responses. Defaults to False
        use_context_learning (bool): Whether to use context learning. Defaults to False
        context_learning_path (str): Path to context learning examples. Defaults to "data/context_learning"
        use_rag (bool): Whether to use RAG. Defaults to False
        session_id (str): Session identifier. Defaults to None
        chroma_db_path (str): Path to ChromaDB. Defaults to "data/rag/chroma_db"
        manim_docs_path (str): Path to Manim docs. Defaults to "data/rag/manim_docs"
        embedding_model (str): Name of embedding model. Defaults to "text-embedding-ada-002"
        use_langfuse (bool): Whether to use Langfuse logging. Defaults to True
        mcp_client (MCPClient): MCP client for Context7 integration. Defaults to None
    """

    def __init__(self, planner_model, helper_model=None, output_dir="output", print_response=False, use_context_learning=False, context_learning_path="data/context_learning", use_rag=False, session_id=None, chroma_db_path="data/rag/chroma_db", manim_docs_path="data/rag/manim_docs", embedding_model="text-embedding-ada-002", use_langfuse=True, mcp_client=None):
        self.planner_model = planner_model
        self.helper_model = helper_model if helper_model is not None else planner_model
        self.output_dir = output_dir
        self.print_response = print_response
        self.use_context_learning = use_context_learning
        self.context_learning_path = context_learning_path
        
        # MCP Client integration for Context7 documentation
        self.mcp_client = mcp_client
        self.context7_retriever = None
        
        # Initialize Context7 retriever if MCP client is provided
        if self.mcp_client and MCP_AVAILABLE:
            self.context7_retriever = Context7DocsRetriever(self.mcp_client)
            print("Context7 documentation retriever initialized for VideoPlanner")
        elif self.mcp_client and not MCP_AVAILABLE:
            print("Warning: MCP client provided but MCP dependencies not available")
        
        # Initialize different types of context examples only if no MCP client and context learning is enabled
        if use_context_learning and not self.mcp_client:
            self.scene_plan_examples = self._load_context_examples('scene_plan')
            self.vision_storyboard_examples = self._load_context_examples('scene_vision_storyboard')
            self.technical_implementation_examples = self._load_context_examples('technical_implementation')
            self.animation_narration_examples = self._load_context_examples('scene_animation_narration')
            self.code_examples = self._load_context_examples('code')
        else:
            # Set to None when using Context7 or when context learning is disabled
            self.scene_plan_examples = None
            self.vision_storyboard_examples = None
            self.technical_implementation_examples = None
            self.animation_narration_examples = None
            self.code_examples = None
        self.use_rag = use_rag
        self.rag_integration = None
        # Initialize traditional RAG if enabled and no MCP client provided
        if use_rag and not self.mcp_client:
            self.rag_integration = RAGIntegration(
                helper_model=helper_model,
                output_dir=output_dir,
                chroma_db_path=chroma_db_path,
                manim_docs_path=manim_docs_path,
                embedding_model=embedding_model,
                use_langfuse=use_langfuse,
                session_id=session_id
            )
        self.relevant_plugins = []  # Initialize as an empty list

    def _load_context_examples(self, example_type: str) -> str:
        """Load context learning examples of a specific type from files.

        Args:
            example_type (str): Type of examples to load ('scene_plan', 'scene_vision_storyboard', etc.)

        Returns:
            str: Formatted string containing the loaded examples, or None if no examples found
        """
        examples = []
        
        # Define file patterns for different types
        file_patterns = {
            'scene_plan': '*_scene_plan.txt',
            'scene_vision_storyboard': '*_scene_vision_storyboard.txt',
            'technical_implementation': '*_technical_implementation.txt',
            'scene_animation_narration': '*_scene_animation_narration.txt',
            'code': '*.py'
        }
        
        pattern = file_patterns.get(example_type)
        if not pattern:
            return None

        # Search in subdirectories of context_learning_path
        for root, _, _ in os.walk(self.context_learning_path):
            for example_file in glob.glob(os.path.join(root, pattern)):
                with open(example_file, 'r') as f:
                    content = f.read()
                    if example_type == 'code':
                        examples.append(f"# Example from {os.path.basename(example_file)}\n{content}\n")
                    else:
                        examples.append(f"# Example from {os.path.basename(example_file)}\n{content}\n")

        # Format examples using appropriate template
        if examples:
            formatted_examples = self._format_examples(example_type, examples)
            return formatted_examples
        return None

    def _format_examples(self, example_type: str, examples: List[str]) -> str:
        """Format examples using the appropriate template based on their type.

        Args:
            example_type (str): Type of examples to format
            examples (List[str]): List of example strings to format

        Returns:
            str: Formatted examples string, or None if no template found
        """
        templates = {
            'scene_plan': get_prompt_context_learning_scene_plan,
            'scene_vision_storyboard': get_prompt_context_learning_vision_storyboard,
            'technical_implementation': get_prompt_context_learning_technical_implementation,
            'scene_animation_narration': get_prompt_context_learning_animation_narration,
            'code': get_prompt_context_learning_code
        }
        
        template = templates.get(example_type)
        if template:
            return template(examples="\n".join(examples))
        return None

    async def _retrieve_context7_documentation_for_planning(self, stage: str, topic: str, description: str, scene_content: str = None) -> str:
        """Retrieve Manim documentation using Context7 MCP server for planning stages.

        Args:
            stage (str): Planning stage ('scene_plan', 'storyboard', 'technical', 'narration')
            topic (str): Topic of the video
            description (str): Description of the video content
            scene_content (str, optional): Specific scene content for context

        Returns:
            str: Formatted documentation string for use in prompts
        """
        if not self.context7_retriever:
            return ""

        try:
            # Extract Manim-related topics from the content
            context7_topic = self._extract_manim_topic_for_planning(stage, topic, description, scene_content)
            
            print(f"Retrieving Context7 documentation for {stage} stage, topic: {context7_topic}")
            
            # Retrieve documentation
            doc_response = await self.context7_retriever.get_manim_documentation(
                topic=context7_topic,
                include_code_examples=True,
                max_tokens=6000  # Smaller token limit for planning stages
            )

            # Format the documentation for use in prompts
            formatted_docs = self._format_context7_docs_for_planning(doc_response, stage)
            
            print(f"Retrieved Context7 documentation for {stage}: {doc_response.get('total_sections', 0)} sections")
            return formatted_docs

        except Exception as e:
            print(f"Error retrieving Context7 documentation for {stage}: {e}")
            # Return empty documentation on error
            return ""

    def _extract_manim_topic_for_planning(self, stage: str, topic: str, description: str, scene_content: str = None) -> Optional[str]:
        """Extract relevant Manim topic from planning content.

        Args:
            stage (str): Planning stage
            topic (str): Video topic
            description (str): Video description
            scene_content (str, optional): Scene-specific content

        Returns:
            Optional[str]: Extracted Manim topic or None
        """
        # Combine all content for analysis
        content = f"{topic} {description}"
        if scene_content:
            content += f" {scene_content}"
        
        content_lower = content.lower()
        
        # Map common topics to Manim documentation topics
        topic_mapping = {
            'animation': 'animations',
            'scene': 'scenes',
            'mobject': 'mobjects',
            'graph': 'graphs',
            'plot': 'plotting',
            'text': 'text',
            'geometry': 'geometry',
            'transform': 'transformations',
            'camera': 'camera',
            '3d': '3d',
            'math': 'mathematical',
            'equation': 'equations',
            'function': 'functions'
        }
        
        # Look for Manim-specific keywords
        for keyword, manim_topic in topic_mapping.items():
            if keyword in content_lower:
                return manim_topic
        
        # Default topics based on planning stage
        stage_defaults = {
            'scene_plan': 'scenes',
            'storyboard': 'animations',
            'technical': 'mobjects',
            'narration': 'animations'
        }
        
        return stage_defaults.get(stage, 'animations')

    def _format_context7_docs_for_planning(self, doc_response: Dict[str, Any], stage: str) -> str:
        """Format Context7 documentation response for planning prompts.

        Args:
            doc_response: Documentation response from Context7
            stage: Planning stage for context-specific formatting

        Returns:
            str: Formatted documentation string
        """
        if not doc_response or not doc_response.get('sections'):
            return ""

        formatted_parts = []
        formatted_parts.append("# Manim Community Documentation\n")
        formatted_parts.append(f"Retrieved for {stage} planning stage\n")
        
        # Add sections with content
        for section in doc_response.get('sections', []):
            if section.get('content'):
                formatted_parts.append(f"## {section.get('title', 'Documentation Section')}\n")
                formatted_parts.append(f"{section.get('content')}\n")
        
        # Add code examples if available
        if doc_response.get('formatted_code'):
            formatted_parts.append("## Code Examples\n")
            formatted_parts.append(doc_response.get('formatted_code'))
        
        return "\n".join(formatted_parts)

    def generate_scene_outline(self,
                            topic: str,
                            description: str,
                            session_id: str) -> str:
        """Generate a scene outline based on the topic and description.

        Args:
            topic (str): The topic of the video
            description (str): Description of the video content
            session_id (str): Session identifier

        Returns:
            str: Generated scene outline
        """
        # Detect relevant plugins upfront if traditional RAG is enabled (not using Context7)
        if self.use_rag and self.rag_integration:
            self.relevant_plugins = self.rag_integration.detect_relevant_plugins(topic, description) or []
            self.rag_integration.set_relevant_plugins(self.relevant_plugins)
            print(f"Detected relevant plugins: {self.relevant_plugins}")

        prompt = get_prompt_scene_plan(topic, description)
        
        # Add Context7 documentation or context learning examples
        if self.context7_retriever:
            # Use Context7 documentation for scene planning
            try:
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
                            self._retrieve_context7_documentation_for_planning(
                                'scene_plan', topic, description
                            )
                        )
                        context7_docs = future.result(timeout=30)
                else:
                    # Run directly in the event loop
                    context7_docs = loop.run_until_complete(
                        self._retrieve_context7_documentation_for_planning(
                            'scene_plan', topic, description
                        )
                    )
                
                if context7_docs:
                    prompt += f"\n\nHere is current Manim Community documentation for reference:\n{context7_docs}"
            except Exception as e:
                print(f"Error retrieving Context7 documentation for scene planning: {e}")
        elif self.use_context_learning and self.scene_plan_examples:
            prompt += f"\n\nHere are some example scene plans for reference:\n{self.scene_plan_examples}"

        # Generate plan using planner model
        response_text = self.planner_model(
            _prepare_text_inputs(prompt),
            metadata={"generation_name": "scene_outline", "tags": [topic, "scene-outline"], "session_id": session_id}
        )
        # extract scene outline <SCENE_OUTLINE> ... </SCENE_OUTLINE>
        scene_outline_match = re.search(r'(<SCENE_OUTLINE>.*?</SCENE_OUTLINE>)', response_text, re.DOTALL)
        scene_outline = scene_outline_match.group(1) if scene_outline_match else response_text

        # replace all spaces and special characters with underscores for file path compatibility
        file_prefix = topic.lower()
        file_prefix = re.sub(r'[^a-z0-9_]+', '_', file_prefix)
        # save plan to file
        os.makedirs(os.path.join(self.output_dir, file_prefix), exist_ok=True) # Ensure directory exists
        with open(os.path.join(self.output_dir, file_prefix, f"{file_prefix}_scene_outline.txt"), "w") as f:
            f.write(scene_outline)
        print(f"Plan saved to {file_prefix}_scene_outline.txt")

        return scene_outline

    async def _generate_scene_implementation_single(self, topic: str, description: str, scene_outline_i: str, i: int, file_prefix: str, session_id: str, scene_trace_id: str) -> str:
        """Generate implementation plan for a single scene.

        Args:
            topic (str): The topic of the video
            description (str): Description of the video content
            scene_outline_i (str): Outline for this specific scene
            i (int): Scene number
            file_prefix (str): Prefix for output files
            session_id (str): Session identifier
            scene_trace_id (str): Unique trace ID for this scene

        Returns:
            str: Generated implementation plan for the scene
        """
        # Initialize empty implementation plan
        implementation_plan = ""
        scene_dir = os.path.join(self.output_dir, file_prefix, f"scene{i}")
        subplan_dir = os.path.join(scene_dir, "subplans")
        os.makedirs(scene_dir, exist_ok=True)
        os.makedirs(subplan_dir, exist_ok=True)

        # Save scene_trace_id to file
        trace_id_file = os.path.join(subplan_dir, "scene_trace_id.txt")
        with open(trace_id_file, 'w') as f:
            f.write(scene_trace_id)
        print(f"Scene trace ID saved to {trace_id_file}")

        # ===== Step 1: Generate Scene Vision and Storyboard =====
        # ===================================================
        prompt_vision_storyboard = get_prompt_scene_vision_storyboard(i, topic, description, scene_outline_i, self.relevant_plugins)

        # Add Context7 documentation or vision storyboard examples
        if self.context7_retriever:
            try:
                context7_docs = await self._retrieve_context7_documentation_for_planning(
                    'storyboard', topic, description, scene_outline_i
                )
                if context7_docs:
                    prompt_vision_storyboard += f"\n\nHere is current Manim Community documentation for storyboard planning:\n{context7_docs}"
            except Exception as e:
                print(f"Error retrieving Context7 documentation for storyboard: {e}")
        elif self.use_context_learning and self.vision_storyboard_examples:
            prompt_vision_storyboard += f"\n\nHere are some example storyboards:\n{self.vision_storyboard_examples}"
        
        # Use traditional RAG only if Context7 is not available
        if self.rag_integration and not self.context7_retriever:
            # Generate RAG queries
            rag_queries = self.rag_integration._generate_rag_queries_storyboard(
                scene_plan=scene_outline_i,
                scene_trace_id=scene_trace_id,
                topic=topic,
                scene_number=i,
                session_id=session_id,
                relevant_plugins=self.relevant_plugins
            )

            retrieved_docs = self.rag_integration.get_relevant_docs(
                rag_queries=rag_queries,
                scene_trace_id=scene_trace_id,
                topic=topic,
                scene_number=i
            )
            
            # Add documentation to prompt
            prompt_vision_storyboard += f"\n\n{retrieved_docs}"

        vision_storyboard_plan = self.planner_model(
            _prepare_text_inputs(prompt_vision_storyboard),
            metadata={"generation_name": "scene_vision_storyboard", "trace_id": scene_trace_id, "tags": [topic, f"scene{i}"], "session_id": session_id}
        )
        # extract vision storyboard plan <SCENE_VISION_STORYBOARD_PLAN> ... </SCENE_VISION_STORYBOARD_PLAN>
        vision_match = re.search(r'(<SCENE_VISION_STORYBOARD_PLAN>.*?</SCENE_VISION_STORYBOARD_PLAN>)', vision_storyboard_plan, re.DOTALL)
        vision_storyboard_plan = vision_match.group(1) if vision_match else vision_storyboard_plan
        implementation_plan += vision_storyboard_plan + "\n\n"
        file_path_vs = os.path.join(subplan_dir, f"{file_prefix}_scene{i}_vision_storyboard_plan.txt")
        with open(file_path_vs, "w") as f:
            f.write(vision_storyboard_plan)
        print(f"Scene {i} Vision and Storyboard Plan saved to {file_path_vs}")

        # ===== Step 2: Generate Technical Implementation Plan =====
        # =========================================================
        prompt_technical_implementation = get_prompt_scene_technical_implementation(i, topic, description, scene_outline_i, vision_storyboard_plan, self.relevant_plugins)

        # Add Context7 documentation or technical implementation examples
        if self.context7_retriever:
            try:
                context7_docs = await self._retrieve_context7_documentation_for_planning(
                    'technical', topic, description, f"{scene_outline_i}\n{vision_storyboard_plan}"
                )
                if context7_docs:
                    prompt_technical_implementation += f"\n\nHere is current Manim Community documentation for technical implementation:\n{context7_docs}"
            except Exception as e:
                print(f"Error retrieving Context7 documentation for technical implementation: {e}")
        elif self.use_context_learning and self.technical_implementation_examples:
            prompt_technical_implementation += f"\n\nHere are some example technical implementations:\n{self.technical_implementation_examples}"

        # Use traditional RAG only if Context7 is not available
        if self.rag_integration and not self.context7_retriever:
            # Generate RAG queries
            rag_queries = self.rag_integration._generate_rag_queries_technical(
                storyboard=vision_storyboard_plan,
                scene_trace_id=scene_trace_id,
                topic=topic,
                scene_number=i,
                session_id=session_id,
                relevant_plugins=self.relevant_plugins
            )

            retrieved_docs = self.rag_integration.get_relevant_docs(
                rag_queries=rag_queries,
                scene_trace_id=scene_trace_id,
                topic=topic,
                scene_number=i
            )

            # Add documentation to prompt
            prompt_technical_implementation += f"\n\n{retrieved_docs}"

        technical_implementation_plan = self.planner_model(
            _prepare_text_inputs(prompt_technical_implementation),
            metadata={"generation_name": "scene_technical_implementation", "trace_id": scene_trace_id, "tags": [topic, f"scene{i}"], "session_id": session_id}
        )
        # extract technical implementation plan <SCENE_TECHNICAL_IMPLEMENTATION_PLAN> ... </SCENE_TECHNICAL_IMPLEMENTATION_PLAN>
        technical_match = re.search(r'(<SCENE_TECHNICAL_IMPLEMENTATION_PLAN>.*?</SCENE_TECHNICAL_IMPLEMENTATION_PLAN>)', technical_implementation_plan, re.DOTALL)
        technical_implementation_plan = technical_match.group(1) if technical_match else technical_implementation_plan
        implementation_plan += technical_implementation_plan + "\n\n"
        file_path_ti = os.path.join(subplan_dir, f"{file_prefix}_scene{i}_technical_implementation_plan.txt")
        with open(file_path_ti, "w") as f:
            f.write(technical_implementation_plan)
        print(f"Scene {i} Technical Implementation Plan saved to {file_path_ti}")
       
        # ===== Step 3: Generate Animation and Narration Plan =====
        # =========================================================
        prompt_animation_narration = get_prompt_scene_animation_narration(i, topic, description, scene_outline_i, vision_storyboard_plan, technical_implementation_plan, self.relevant_plugins)
        
        # Add Context7 documentation or animation narration examples
        if self.context7_retriever:
            try:
                context7_docs = await self._retrieve_context7_documentation_for_planning(
                    'narration', topic, description, f"{scene_outline_i}\n{vision_storyboard_plan}\n{technical_implementation_plan}"
                )
                if context7_docs:
                    prompt_animation_narration += f"\n\nHere is current Manim Community documentation for animation and narration:\n{context7_docs}"
            except Exception as e:
                print(f"Error retrieving Context7 documentation for animation narration: {e}")
        elif self.use_context_learning and self.animation_narration_examples:
            prompt_animation_narration += f"\n\nHere are some example animation and narration plans:\n{self.animation_narration_examples}"
        
        # Use traditional RAG only if Context7 is not available
        if self.rag_integration and not self.context7_retriever:
            rag_queries = self.rag_integration._generate_rag_queries_narration(
                storyboard=vision_storyboard_plan,
                scene_trace_id=scene_trace_id,
                topic=topic,
                scene_number=i,
                session_id=session_id,
                relevant_plugins=self.relevant_plugins
            )
            retrieved_docs = self.rag_integration.get_relevant_docs(
                rag_queries=rag_queries,
                scene_trace_id=scene_trace_id,
                topic=topic,
                scene_number=i
            )
            prompt_animation_narration += f"\n\n{retrieved_docs}"

        animation_narration_plan = self.planner_model(
            _prepare_text_inputs(prompt_animation_narration),
            metadata={"generation_name": "scene_animation_narration", "trace_id": scene_trace_id, "tags": [topic, f"scene{i}"], "session_id": session_id}
        )
        # extract animation narration plan <SCENE_ANIMATION_NARRATION_PLAN> ... </SCENE_ANIMATION_NARRATION_PLAN>
        animation_match = re.search(r'(<SCENE_ANIMATION_NARRATION_PLAN>.*?</SCENE_ANIMATION_NARRATION_PLAN>)', animation_narration_plan, re.DOTALL)
        animation_narration_plan = animation_match.group(1) if animation_match else animation_narration_plan
        implementation_plan += animation_narration_plan + "\n\n"
        file_path_an = os.path.join(subplan_dir, f"{file_prefix}_scene{i}_animation_narration_plan.txt")
        with open(file_path_an, "w") as f:
            f.write(animation_narration_plan)
        print(f"Scene {i} Animation and Narration Plan saved to {file_path_an}")

        # ===== Step 4: Save Implementation Plan =====
        # ==========================================
        # save the overall implementation plan to file
        with open(os.path.join(self.output_dir, file_prefix, f"scene{i}", f"{file_prefix}_scene{i}_implementation_plan.txt"), "w") as f:
            f.write(f"# Scene {i} Implementation Plan\n\n")
            f.write(implementation_plan)
        print(f"Scene {i} Implementation Plan saved to {file_path_ti}")

        return implementation_plan

    async def generate_scene_implementation(self,
                                      topic: str,
                                      description: str,
                                      plan: str,
                                      session_id: str) -> List[str]:
        """Generate detailed implementation plans for all scenes.

        Args:
            topic (str): The topic of the video
            description (str): Description of the video content
            plan (str): Overall scene plan
            session_id (str): Session identifier

        Returns:
            List[str]: List of implementation plans for each scene
        """
        # extract scene outline <SCENE_OUTLINE> ... </SCENE_OUTLINE>
        scene_outline = re.search(r'(<SCENE_OUTLINE>.*?</SCENE_OUTLINE>)', plan, re.DOTALL).group(1)
        # check the number of scenes in the outline
        scene_number = len(re.findall(r'<SCENE_(\d+)>[^<]', scene_outline))
        # replace all spaces and special characters with underscores for file path compatibility
        file_prefix = topic.lower()
        file_prefix = re.sub(r'[^a-z0-9_]+', '_', file_prefix)
        # generate implementation plan for each scene
        all_scene_implementation_plans = []

        tasks = []
        for i in range(1, scene_number):
            print(f"Generating implementation plan for scene {i} in topic {topic}")
            scene_outline_i = re.search(r'(<SCENE_{i}>.*?</SCENE_{i}>)'.format(i=i), scene_outline, re.DOTALL).group(1)
            scene_trace_id = str(uuid.uuid4())
            task = asyncio.create_task(self._generate_scene_implementation_single(topic, description, scene_outline_i, i, file_prefix, session_id, scene_trace_id))
            tasks.append(task)

        all_scene_implementation_plans = await asyncio.gather(*tasks)
        return all_scene_implementation_plans

    async def generate_scene_implementation_concurrently(self,
                                              topic: str,
                                              description: str,
                                              plan: str,
                                              session_id: str,
                                              scene_semaphore) -> List[str]:
        """Generate detailed implementation plans for all scenes concurrently with controlled concurrency.

        Args:
            topic (str): The topic of the video
            description (str): Description of the video content
            plan (str): Overall scene plan
            session_id (str): Session identifier
            scene_semaphore: Semaphore to control concurrent scene generation

        Returns:
            List[str]: List of implementation plans for each scene
        """
        scene_outline = extract_xml(plan)
        scene_number = len(re.findall(r'<SCENE_(\d+)>[^<]', scene_outline))
        file_prefix = re.sub(r'[^a-z0-9_]+', '_', topic.lower())
        all_scene_implementation_plans = []

        async def generate_single_scene_implementation(i):
            async with scene_semaphore:  # controls parallelism
                print(f"Generating implementation plan for scene {i} in topic {topic}")
                scene_outline_i = re.search(r'(<SCENE_{i}>.*?</SCENE_{i}>)'.format(i=i), scene_outline, re.DOTALL).group(1)
                scene_trace_id = str(uuid.uuid4())  # Generate UUID here
                return await self._generate_scene_implementation_single(topic, description, scene_outline_i, i, file_prefix, session_id, scene_trace_id)

        tasks = [generate_single_scene_implementation(i + 1) for i in range(scene_number)]
        all_scene_implementation_plans = await asyncio.gather(*tasks)
        return all_scene_implementation_plans 