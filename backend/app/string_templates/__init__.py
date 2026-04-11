"""Centralized string templates and literal copy.

Submodules hold only ``str`` constants and ``str.format`` placeholders — no business logic.
Application code imports specific submodules (e.g. ``app.string_templates.rag_agent``).
"""
