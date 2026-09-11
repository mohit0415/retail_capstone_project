"""Cost and latency optimisation: model routing.

The Sprint-9 "Gateway Router" pattern, applied to the policy graph:

* :mod:`src.llm_routing.complexity` - a heuristic, zero-latency assessment of
  how much reasoning a question needs (keyword and length rules, the risk
  level, the intent, the evidence path). No model call is spent deciding
  which model to call.
* :mod:`src.llm_routing.router` - turns that assessment, the configured
  strategy and the remaining time / token budget into a tier (``small`` or
  ``strong``) for each *routable* node. Classification nodes are always
  small; the high-risk panel is always strong.
* :mod:`src.llm_routing.pricing` and :mod:`src.llm_routing.ledger` - price
  every model call from its real token usage so the cost of a request, and of
  the whole process, is a number the SLO report can hold to a budget.

``configs.llms.routed_model`` is the entry point the nodes use.
"""

from src.llm_routing.complexity import ComplexityAssessment, assess_complexity
from src.llm_routing.router import RoutingDecision, route_tier

__all__ = ["ComplexityAssessment", "RoutingDecision", "assess_complexity", "route_tier"]
