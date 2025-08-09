"""
Global warning suppression for the video generation system.
Import this module early to suppress common third-party library warnings.
"""

import warnings
import os

# Set environment variable to suppress warnings globally
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'

# Suppress specific deprecation warnings
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=".*langchain_core.pydantic_v1.*"
)

warnings.filterwarnings(
    "ignore", 
    category=DeprecationWarning,
    message=".*langchain.pydantic_v1.*"
)

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="ragas.*"
)

# Suppress specific ragas warnings about missing components
warnings.filterwarnings(
    "ignore",
    message=".*Enhanced components not fully available.*"
)

warnings.filterwarnings(
    "ignore",
    message=".*cannot import name 'AnswerAccuracy'.*"
)

# Apply the filters
warnings.simplefilter("ignore", DeprecationWarning)