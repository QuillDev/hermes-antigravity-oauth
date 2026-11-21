"""What the agent this plugin is loaded into calls itself.

A rebranded build of the agent renames its command and its core modules together, so the name is
read off a core module rather than written here. Prompts, rules files and hints then name the agent
the user is actually running.
"""

from __future__ import annotations


def _name() -> str:
    try:
        import hermes_constants
    except ImportError:  # the flat source tree under test, with no core on sys.path
        return "hermes"
    return hermes_constants.__name__.partition("_")[0]


NAME = _name()
TITLE = NAME.capitalize()
