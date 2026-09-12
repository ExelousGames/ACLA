from copy import deepcopy

import pytest


def _object_schema(properties, required=()):
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def _child_schema(name, properties, required):
    return _object_schema({
        "tool": _object_schema({"name": {"type": "string", "enum": [name]}, **properties}, ["name", *required]),
    }, ["tool"])


@pytest.fixture(params=[
    "set_procedure_plan",
    "create_repeatable_plan",
    "add_event_to_live_range_todo_list",
])
def user_workflow_case(request):
    """Representative backend descriptors and calls for the strict protocol."""
    name = request.param
    map_schema = _object_schema({
        "source_track_key": {"type": "string"},
        "section_start": {"type": "number"},
        "section_end": {"type": "number"},
    })
    map_arguments = {
        "source_track_key": "spa",
        "section_start": 0,
        "section_end": 0.2,
    }
    if name == "set_procedure_plan":
        children = [_child_schema("show_map", {
            "title": {"type": "string"},
            "arguments": map_schema,
        }, ["title", "arguments"])]
        properties = {"goal": {"type": "string"}}
        body = {
            "goal": "Review the Spa opening section",
            "tools": [{"tool": {"name": "show_map",
                "title": "Show the opening section",
                "arguments": map_arguments,
            }}],
        }
    elif name == "create_repeatable_plan":
        condition_schema = _object_schema({
            "field": {"type": "string"},
            "operator": {"type": "string", "enum": [
                "eq", "neq", "lt", "lte", "gt", "gte",
            ]},
            "value": {"type": "number"},
        }, ["field", "operator", "value"])
        children = [
            _child_schema("collect_live_baseline", {
                "id": {"type": "string"},
                "title": {"type": "string"},
                "arguments": _object_schema({
                    "query": {"oneOf": [
                        _object_schema({
                            "preset": {"type": "string", "enum": ["full_lap"]},
                        }, ["preset"]),
                        _object_schema({
                            "start_query": deepcopy(condition_schema),
                            "end_query": deepcopy(condition_schema),
                        }, ["start_query", "end_query"]),
                    ]},
                }, ["query"]),
            }, ["id", "title"]),
            _child_schema("analyze_live_recorded_analysis", {
                "id": {"type": "string"},
                "title": {"type": "string"},
                "arguments": _object_schema({"limit": {"type": "integer"}}),
            }, ["id", "title"]),
        ]
        properties = {
            "goal": {"type": "string"},
            "stop_when": _object_schema({
                "tool": {"oneOf": [_child_schema("query_lap_analysis_result", {
                    "arguments": _object_schema({
                        "query": {"type": "string", "minLength": 1},
                    }, ["query"]),
                }, [])["properties"]["tool"]]},
                "operator": {"type": "string", "enum": [
                    "eq", "neq", "lt", "lte", "gt", "gte",
                ]},
                "target": {"type": "number"},
            }, ["tool", "operator", "target"]),
        }
        body = {
            "goal": "Analyze five laps",
            "tools": [
                {"tool": {"name": "collect_live_baseline",
                    "id": "collect",
                    "title": "Record a full lap",
                    "arguments": {"query": {"preset": "full_lap"}},
                }},
                {"tool": {"name": "analyze_live_recorded_analysis",
                    "id": "analyze",
                    "title": "Analyze the recorded lap",
                }},
            ],
            "stop_when": {
                "tool": {"name": "query_lap_analysis_result",
                    "arguments": {"query": "$count(analyses)"},
                },
                "operator": "gte",
                "target": 5,
            },
        }
    else:
        children = [_child_schema("show_map", {
            "event": _object_schema({
                "id": {"type": "string", "minLength": 1},
                "normalized_position": {
                    "type": "number", "minimum": 0, "maximum": 1,
                },
                "lead_time_seconds": {"type": "number", "minimum": 0},
                "content": _object_schema({
                    "title": {"type": "string", "minLength": 1},
                    "description": {"type": "string"},
                }, ["title"]),
            }, ["id", "normalized_position", "content"]),
            "arguments": map_schema,
        }, ["event", "arguments"])]
        properties = {}
        body = {"tools": [{"tool": {"name": "show_map",
            "event": {
                "id": "spa-opening-map",
                "normalized_position": 0.1,
                "lead_time_seconds": 2,
                "content": {
                    "title": "Opening section",
                    "description": "Show the selected Spa section",
                },
            },
            "arguments": map_arguments,
        }}]}

    properties = {"name": {"type": "string", "enum": [name]}, **properties}
    body = {"name": name, **body}
    properties["tools"] = {
        "type": "array",
        "minItems": 1,
        "items": {"oneOf": children},
    }
    return deepcopy({
        "descriptor": {
            "name": name,
            "description": "Create a visible tool-only workflow.",
            "properties": {"workflow": _object_schema(properties, properties)},
            "required": ["workflow"],
        },
        "arguments": {"workflow": body},
    })
