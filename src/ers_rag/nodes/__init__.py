"""V7 graph nodes — thin orchestration wrappers.

Each node reads state, calls nlp_core/hard_gates, writes back.
Per design principle 5.7: no business logic in nodes.
"""

from __future__ import annotations
