"""Calute UI module for creating Gradio-based chat interfaces.

This module provides components for building interactive chat applications
with the Calute framework.
"""

try:
    from .application import create_application
except Exception:
    print("Error importing gradio install calute via `pip install calute[ui]`")
    create_application = None
__all__ = ["create_application"]
