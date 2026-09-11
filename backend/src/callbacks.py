"""
Compatibility shim.

The Langfuse callback manager now lives in
``src.observability.langfuse_callback`` (adapted for the langfuse 4.x SDK this
project pins). This module re-exports it so ``from src.callbacks import ...``
keeps working.
"""

from src.observability.langfuse_callback import (
    LangfuseCallbackManager,
    flush_langfuse_traces,
    get_langfuse_manager,
    setup_langfuse_callback,
)

__all__ = [
    "LangfuseCallbackManager",
    "flush_langfuse_traces",
    "get_langfuse_manager",
    "setup_langfuse_callback",
]
