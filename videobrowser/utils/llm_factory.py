import os
from langchain_openai import ChatOpenAI
from videobrowser.config import get_config

def get_llm(node_name: str = None) -> ChatOpenAI:
    """
    Factory function to get an LLM instance based on configuration.
    
    Args:
        node_name: The name of the node (e.g., 'planner', 'selector') to apply overrides.
    """
    config = get_config()
    
    # Start with default config values
    llm_params = config.llm.default.model_dump()
    
    # Apply overrides if they exist for this node
    if node_name and node_name in config.llm.overrides:
        # Merge dictionary updates
        override_params = config.llm.overrides[node_name]
        llm_params.update(override_params)
    
    # Extract params needed for instantiation.  `api_key_env`, if set, names an
    # alternate env var (e.g. "GEMINI_API_KEY") to source the key from — takes
    # precedence over any literal `api_key` or the OPENAI_API_KEY default.
    api_key_env = llm_params.get("api_key_env")
    if api_key_env:
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(
                f"api_key_env={api_key_env!r} is set in config but that env var is empty."
            )
    else:
        api_key = llm_params.get("api_key")

    if not api_key:
        raise ValueError("API Key not found in configuration or environment variable.")
    
    # Construct arguments for ChatOpenAI
    # We filter out internal config keys (like 'provider') that ChatOpenAI doesn't accept,
    # unless we want to map them specifically.
    
    init_kwargs = {
        "model": llm_params.get("model"),
        "temperature": llm_params.get("temperature"),
        "api_key": api_key,
    }

    if llm_params.get("base_url"):
        init_kwargs["base_url"] = llm_params.get("base_url")

    if llm_params.get("max_tokens"):
        init_kwargs["max_tokens"] = llm_params.get("max_tokens")

    # Forward explicit safeguard flags when configured. The `use_responses_api`
    # field gates OpenAI's Responses endpoint (the only path through which
    # server-side tools like `web_search` become available). Chat.completions
    # has no built-in web_search, so defaulting to False when the key is
    # present means any future langchain default flip won't silently enable it.
    for pass_through in ("use_responses_api", "model_kwargs", "extra_body", "disabled_params"):
        if pass_through in llm_params and llm_params[pass_through] is not None:
            init_kwargs[pass_through] = llm_params[pass_through]

    # Create the instance
    # Future: Switch on llm_params['provider'] to support other classes
    return ChatOpenAI(**init_kwargs)
