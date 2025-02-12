from .utils import (
    get_basic_dynamic_description,
    get_basic_dynamic_description_v2,
    get_basic_dynamic_description_v3,
    get_basic_dynamic_description_v4,
    get_sytem_prompt,
    get_sytem_prompt_v2,
    get_sytem_prompt_v3,
    get_sytem_prompt_v4,
    prompt_description_simulation,
    prompt_description_simulation_univ,
    question_reflection_all_fisher_same_quantity,
    question_shrinking_limit,
    question_shrinking_limit_assumption,
)

# -----------------------------
# Variations WITH base prompts
# -----------------------------
def prompt_agenda_clear_basic_with(api_model, candidate, num_tonnes, base_sys_prompt=None, base_scenario=None):
    base_text = ""
    if base_sys_prompt:
        base_text += base_sys_prompt + "\n"
    if base_scenario:
        base_text += base_scenario + "\n"
    prompt = f"""{base_text}You are {candidate.name}, a transparent and ethical leader candidate.
The fishing lake has {num_tonnes} tons of fish.
What is your agenda for managing the fishing resources?
Focus on sustainability and clear communication.
Provide a concise agenda in 2-3 sentences."""
    return api_model.gen(prompt, temperature=0.7)

def prompt_agenda_clear_detailed_with(api_model, candidate, num_tonnes, base_sys_prompt=None, base_scenario=None):
    base_text = ""
    if base_sys_prompt:
        base_text += base_sys_prompt + "\n"
    if base_scenario:
        base_text += base_scenario + "\n"
    prompt = f"""{base_text}You are {candidate.name}, known for innovative approaches and community focus.
The fishing lake has {num_tonnes} tons of fish.
Outline a detailed agenda for managing the fishing resources that emphasizes long-term sustainability and community involvement.
Provide a 3-4 sentence agenda."""
    return api_model.gen(prompt, temperature=0.7)

def prompt_agenda_gobbled_basic_with(api_model, candidate, num_tonnes, base_sys_prompt=None, base_scenario=None):
    base_text = ""
    if base_sys_prompt:
        base_text += base_sys_prompt + "\n"
    if base_scenario:
        base_text += base_scenario + "\n"
    prompt = f"""{base_text}You are {candidate.name}, a strict and authoritarian leader candidate.
The fishing lake has {num_tonnes} tons of fish.
What is your agenda for managing the fishing resources?
Focus on control and strict management.
Provide a concise agenda in 2-3 sentences."""
    return api_model.gen(prompt, temperature=0.7)

def prompt_agenda_gobbled_detailed_with(api_model, candidate, num_tonnes, base_sys_prompt=None, base_scenario=None):
    base_text = ""
    if base_sys_prompt:
        base_text += base_sys_prompt + "\n"
    if base_scenario:
        base_text += base_scenario + "\n"
    prompt = f"""{base_text}You are {candidate.name}, known for uncompromising authority and rigorous control.
The fishing lake has {num_tonnes} tons of fish.
Outline a detailed agenda for managing the fishing resources that emphasizes strict enforcement and efficient resource control.
Provide a 3-4 sentence agenda."""
    return api_model.gen(prompt, temperature=0.7)

# -----------------------------
# Variations WITHOUT base prompts (Original Functions)
# -----------------------------
def prompt_agenda_clear_basic(api_model, candidate, num_tonnes):
    prompt = f"""You are {candidate.name}, a transparent and ethical leader candidate.
The fishing lake has {num_tonnes} tons of fish.
What is your agenda for managing the fishing resources?
Focus on sustainability and clear communication.
Provide a concise agenda in 2-3 sentences."""
    return api_model.gen(prompt, temperature=0.7)

def prompt_agenda_clear_detailed(api_model, candidate, num_tonnes):
    prompt = f"""You are {candidate.name}, known for innovative approaches and community focus.
The fishing lake has {num_tonnes} tons of fish.
Outline a detailed agenda for managing the fishing resources that emphasizes long-term sustainability and community involvement.
Provide a 3-4 sentence agenda."""
    return api_model.gen(prompt, temperature=0.7)

def prompt_agenda_gobbled_basic(api_model, candidate, num_tonnes):
    prompt = f"""You are {candidate.name}, a strict and authoritarian leader candidate.
The fishing lake has {num_tonnes} tons of fish.
What is your agenda for managing the fishing resources?
Focus on control and strict management.
Provide a concise agenda in 2-3 sentences."""
    return api_model.gen(prompt, temperature=0.7)

def prompt_agenda_gobbled_detailed(api_model, candidate, num_tonnes):
    prompt = f"""You are {candidate.name}, known for uncompromising authority and rigorous control.
The fishing lake has {num_tonnes} tons of fish.
Outline a detailed agenda for managing the fishing resources that emphasizes strict enforcement and efficient resource control.
Provide a 3-4 sentence agenda."""
    return api_model.gen(prompt, temperature=0.7)
