import os
from enum import Enum
from dataclasses import dataclass, fields, field
from typing import Any, Optional, Dict, Literal
from langchain_core.runnables import RunnableConfig

DEFAULT_REPORT_STRUCTURE = """Use this structure to create a report on the user-provided topic:

1. Introduction
    - Brief introduction of the subject
    
2. Main Body
    - 2.1 Work Responsibilities and Policy Positions
        - Conclusion
        - Evidence
    - 2.2 Stance on China
        - Overall attitude
        - Position on key issues
        - Conclusion
        - Evidence
    - 2.3 Personality Traits and Decision-Making Style
        - Conclusion
        - Evidence
    - 2.4 External Evaluation and Social Controversies
        - Conclusion
        - Evidence
        
3. Conclusion

4. Summary
    - Summarize main findings in a table or list 
    - Concise overall assessment"""

# class SearchAPI(Enum):
#     PERPLEXITY = "perplexity"
#     TAVILY = "tavily"
#     EXA = "exa"
#     ARXIV = "arxiv"
#     PUBMED = "pubmed"
#     LINKUP = "linkup"
#     DUCKDUCKGO = "duckduckgo"
#     GOOGLESEARCH = "googlesearch"
#     NONE = "none"
class SearchAPI(Enum):
    TAVILY = "tavily"
    GOOGLESEARCH = "googlesearch"
    MULTISOURCE = 'multi_source'
    LOCALJSON = 'local_json'
    WIKIPEDIA = 'wikipedia'
    NONE = "none"


@dataclass(kw_only=True)
class WorkflowConfiguration:
    """Configuration for the workflow/graph-based implementation (graph.py)."""
    # Common configuration
    report_structure: str = DEFAULT_REPORT_STRUCTURE
    search_api: SearchAPI = SearchAPI.GOOGLESEARCH
    search_api_config: Optional[Dict[str, Any]] = None
    process_search_results: Literal["summarize", "split_and_rerank"] | None = None
    summarization_model_provider: str = "anthropic"
    summarization_model: str = "claude-3-5-haiku-latest"
    max_structured_output_retries: int = 3
    include_source_str: bool = False
    
    # Workflow-specific configuration
    number_of_queries: int = 2 # Number of search queries to generate per iteration
    max_search_depth: int = 2 # Maximum number of reflection + search iterations
    planner_provider: str = "openai"
    planner_model: str = "gpt-4o"
    planner_model_kwargs: Optional[Dict[str, Any]] = field(default_factory=lambda: {"base_url": "https://pro.xiaoai.plus/v1"})
    writer_provider: str = "openai"
    writer_model: str = "gpt-4o"
    writer_model_kwargs: Optional[Dict[str, Any]] = field(default_factory=lambda: {"base_url": "https://pro.xiaoai.plus/v1"})
    trainslate_provider: str = "openai"
    trainslate_model: str = "gpt-4o"
    trainslate_model_kwargs: Optional[Dict[str, Any]] = None
    deduplicate_provider: str = "openai"
    deduplicate_model: str = "gpt-4o"
    deduplicate_model_kwargs: Optional[Dict[str, Any]] = None

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "WorkflowConfiguration":
        """Create a WorkflowConfiguration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})

@dataclass(kw_only=True)
class MultiAgentConfiguration:
    """Configuration for the multi-agent implementation (multi_agent.py)."""
    # Common configuration
    search_api: SearchAPI = SearchAPI.TAVILY
    search_api_config: Optional[Dict[str, Any]] = None
    process_search_results: Literal["summarize", "split_and_rerank"] | None = None
    summarization_model_provider: str = "anthropic"
    summarization_model: str = "claude-3-5-haiku-latest"
    include_source_str: bool = False
    
    # Multi-agent specific configuration
    number_of_queries: int = 2 # Number of search queries to generate per section
    supervisor_model: str = "anthropic:claude-3-7-sonnet-latest"
    researcher_model: str = "anthropic:claude-3-7-sonnet-latest"
    ask_for_clarification: bool = False # Whether to ask for clarification from the user
    # MCP server configuration
    mcp_server_config: Optional[Dict[str, Any]] = None
    mcp_prompt: Optional[str] = None
    mcp_tools_to_include: Optional[list[str]] = None

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "MultiAgentConfiguration":
        """Create a MultiAgentConfiguration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})

# Keep the old Configuration class for backward compatibility
Configuration = WorkflowConfiguration
