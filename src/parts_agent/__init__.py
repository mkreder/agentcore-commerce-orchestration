"""Advisor agent + safety monitor for an auto parts store, orchestrated in code.

One turn, left to right:

    01 detector  ->  02 advisor  ->  03 cart action  ->  04 review gate  ->  05 safety monitor  ->  response
       (regex)       (agent)         (allow-listed tools)   (code)              (agent, no tools)     (SSE)

Two agents, one acts. Everything that must not fail is deterministic code.
"""
