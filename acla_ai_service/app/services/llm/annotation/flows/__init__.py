"""
Flow definitions. Each flow declares:

    build_request(...)  -> AgentRequest    domain context -> agent input
    parse(response, ...) -> typed result   raw text       -> typed output

Add a new flow by writing a new module here; nothing in the box changes.
"""
