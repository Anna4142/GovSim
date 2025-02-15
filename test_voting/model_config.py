# model_config.py

def get_model_configs():
    """
    Returns a list of dictionaries. Each entry has:
      - display_name: The 'label' or 'name' you want to show in your tables/logs
      - model_id: The ACTUAL ID passed to the code (always "openai/gpt-4" here)
      - api_key: The unique API key for that 'label'
    """
    return [
        {
            "display_name": "Claude-Sonnet",
            "model_id": "gpt-4",  # same for all
            "api_key": "sk-or-v1-902243cd87dd4bf7e3ba7133ccb46a36..."
        },
        {
            "display_name": "Claude-Haiku",
            "model_id": "openai/gpt-4",  # same for all
            "api_key": "sk-or-v1-6ade5cbf24d600745f58b17235736853..."
        },
        {
            "display_name": "Claude-Opus",
            "model_id": "gpt-4",  # same for all
            "api_key": "sk-or-v1-030ba5525014b627bd820aca689ca13..."
        },
        {
            "display_name": "GPT-3.5(Turbo)",
            "model_id": "gpt-4",  # same for all
            "api_key": "sk-or-v1-eeb45853062534ca4d47329f00d2675..."
        },
        {
            "display_name": "GPT-4o",
            "model_id": "gpt-4",  # same for all
            "api_key": "sk-or-v1-5c665fb2c134d89ae8636f5283d8e00..."
        },
        {
            "display_name": "OpenAI-o3(high)",
            "model_id": "gpt-4",  # same for all
            "api_key": "sk-or-v1-595cd32f0e232fc1aa347bdd59009d56..."
        }
    ]

