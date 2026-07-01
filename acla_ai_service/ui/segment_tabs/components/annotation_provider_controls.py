"""Streamlit controls for annotation AI provider selection."""

from __future__ import annotations

from typing import Optional

import streamlit as st

from app.annotation_providers.registry import list_annotation_providers
from app.local_annotation_agent.workflow import AnnotationPipelineConfig


def render_annotation_provider_config(
    *,
    key_prefix: str,
    default_temperature: float,
    default_max_new_tokens: int,
    default_tool_budget: int = 3,
) -> Optional[AnnotationPipelineConfig]:
    providers = list_annotation_providers()
    if not providers:
        st.error(
            "No annotation AI providers are configured. Configure "
            "`ANNOTATION_ENABLED_PROVIDERS` and the selected provider settings."
        )
        return None

    provider = st.selectbox(
        "AI annotation provider",
        options=providers,
        format_func=lambda p: p.label,
        key=f"{key_prefix}_provider",
    )
    st.caption(provider.description)
    if not provider.configured:
        missing = ", ".join(provider.required_settings) or "provider configuration"
        st.warning(
            f"{provider.label} is available to configure, but cannot run until "
            f"{missing} is set."
        )

    model = provider.default_model_id()
    model_lookup = {m.id: m for m in provider.models}
    if provider.models:
        model_ids = [m.id for m in provider.models]
        default_idx = model_ids.index(model) if model in model_ids else 0
        model = st.selectbox(
            "Model",
            options=model_ids,
            format_func=lambda m: model_lookup[m].label,
            index=default_idx,
            key=f"{key_prefix}_model",
        )
    else:
        model = st.text_input(
            "Model",
            value=model,
            key=f"{key_prefix}_model_text",
        )

    selected_model = model_lookup.get(model)
    max_new_limit = selected_model.max_new_tokens if selected_model else None
    if max_new_limit:
        max_new_tokens = min(default_max_new_tokens, int(max_new_limit))
    else:
        max_new_tokens = int(default_max_new_tokens)

    col_a, col_b = st.columns(2)
    with col_a:
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.5,
            value=float(default_temperature),
            step=0.1,
            key=f"{key_prefix}_temperature",
        )
    with col_b:
        tool_budget = st.number_input(
            "Tool-call budget (x10)",
            min_value=1,
            max_value=10,
            value=int(default_tool_budget),
            key=f"{key_prefix}_tool_budget",
        )

    provider_options = {}
    regular_options = [opt for opt in provider.options if not opt.advanced]
    advanced_options = [opt for opt in provider.options if opt.advanced]

    def render_option(opt):
        key = f"{key_prefix}_option_{provider.id}_{opt.key}"
        value = opt.default
        if opt.kind == "checkbox":
            value = st.checkbox(opt.label, value=bool(opt.default), help=opt.help, key=key)
        elif opt.kind == "select":
            options = list(opt.options)
            idx = options.index(opt.default) if opt.default in options else 0
            value = st.selectbox(opt.label, options=options, index=idx, help=opt.help, key=key)
        elif opt.kind == "number":
            kwargs = {
                "label": opt.label,
                "value": int(opt.default or 0),
                "step": int(opt.step or 1),
                "help": opt.help,
                "key": key,
            }
            if opt.min_value is not None:
                kwargs["min_value"] = int(opt.min_value)
            max_value = opt.max_value
            if opt.key == "context_size" and selected_model and selected_model.max_context:
                max_value = selected_model.max_context
                kwargs["value"] = min(int(kwargs["value"]), int(max_value))
            if max_value is not None:
                kwargs["max_value"] = int(max_value)
            value = st.number_input(**kwargs)
        else:
            value = st.text_input(opt.label, value=str(opt.default or ""), help=opt.help, key=key)
        return value

    for opt in regular_options:
        provider_options[opt.key] = render_option(opt)
    if advanced_options:
        with st.expander("Advanced provider settings", expanded=False):
            for opt in advanced_options:
                provider_options[opt.key] = render_option(opt)

    provider_options["max_turns"] = int(tool_budget) * 10
    return AnnotationPipelineConfig(
        provider_id=provider.id,
        model=model,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        provider_options=provider_options,
    )
