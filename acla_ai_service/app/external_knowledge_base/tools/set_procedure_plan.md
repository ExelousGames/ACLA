---
name: set_procedure_plan
title: Setting procedure plan
description: >
  Create or replace the visible procedure plan UI with an AI-authored list of
  requests. Use this when you have decided the plan and need the frontend to
  display it.
parameters:
  goal:
    description: Short goal shown above the request list.
  current_request:
    description: Optional zero-based index of the active request. Usually 0.
  requests:
    description: >
      Ordered list of request objects. Each request must have `type` and
      `title`, and may include `detail`, `name`, `method`, `url`, and `payload`.
---

## Usage notes

Use this to make the plan visible after you have decided the plan yourself.
Do not ask the frontend to invent steps.

Use only the top-level fields `goal`, `current_request`, and `requests`.
Do not add plan-level focus fields. Section names, map ranges, API arguments,
or other request-specific data belong inside the relevant request's `payload`.

Prefer request types such as:

- `tool_call` for frontend or server tools you intend to call, with `name` and
  optional `payload`.
- `api_request` for HTTP/API work, with `method`, `url`, and optional `payload`.
- `driver_action` for something the driver needs to do before the next request.

Keep titles short and concrete. Include only requests you actually intend to
perform or monitor.
