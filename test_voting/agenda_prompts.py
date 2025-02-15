from utils import (
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
# In-Depth (Detailed Reasoning, Poor Communication)
# -----------------------------
# With base prompts:
def prompt_agenda_in_depth_var1_with(api_model, candidate, num_tonnes, base_sys_prompt=None, base_scenario=None):
    base_text = get_sytem_prompt() + "\n" + prompt_description_simulation() + "\n"
    if base_sys_prompt:
        base_text += base_sys_prompt + "\n"
    if base_scenario:
        base_text += base_scenario + "\n"
    prompt = f"""{base_text}You are {candidate.name}, an analytical leader known for your deep and comprehensive reasoning—even though your communication tends to be verbose and unclear.
The fishing lake has {num_tonnes} tons of fish.
Provide a detailed agenda for managing the fishing resources in 3–4 sentences, focusing on your thorough analysis.
(Clarity is secondary.)"""
    return api_model.gen(prompt, temperature=0.0)

def prompt_agenda_in_depth_var2_with(api_model, candidate, num_tonnes, base_sys_prompt=None, base_scenario=None):
    base_text = get_sytem_prompt() + "\n" + prompt_description_simulation() + "\n"
    if base_sys_prompt:
        base_text += base_sys_prompt + "\n"
    if base_scenario:
        base_text += base_scenario + "\n"
    prompt = f"""{base_text}You are {candidate.name}, an expert in detailed reasoning. Your explanations are highly analytical, though they can be convoluted.
The fishing lake has {num_tonnes} tons of fish.
Describe your comprehensive strategy for managing the fishing resources in 3–4 sentences, prioritizing in-depth analysis even if it compromises clarity."""
    return api_model.gen(prompt, temperature=0.0)

# Without base prompts:
def prompt_agenda_in_depth_var1(api_model, candidate, num_tonnes):
    prompt = f"""You are {candidate.name}, an analytical leader known for deep and comprehensive reasoning, though your communication is often verbose and unclear.
The fishing lake has {num_tonnes} tons of fish.
Provide a detailed agenda in 3–4 sentences for managing the fishing resources, emphasizing thorough analysis over clarity."""
    return api_model.gen(prompt, temperature=0.0)

def prompt_agenda_in_depth_var2(api_model, candidate, num_tonnes):
    prompt = f"""You are {candidate.name}, a leader with a talent for in-depth reasoning, even if your message is not concise.
The fishing lake has {num_tonnes} tons of fish.
Outline your comprehensive strategy for managing the fishing resources in 3–4 sentences, focusing on detailed analysis at the expense of brevity."""
    return api_model.gen(prompt, temperature=0.0)

# -----------------------------
# Clear Communication (Clear, Concise, but Less Detailed Reasoning)
# -----------------------------
# With base prompts:
def prompt_agenda_clear_comm_var1_with(api_model, candidate, num_tonnes, base_sys_prompt=None, base_scenario=None):
    # Here we use base context optionally.
    base_text = ""
    if base_sys_prompt:
        base_text += base_sys_prompt + "\n"
    if base_scenario:
        base_text += base_scenario + "\n"
    prompt = f"""{base_text}You are {candidate.name}, a leader celebrated for your clear and concise communication, though your reasoning is less detailed.
The fishing lake has {num_tonnes} tons of fish.
Provide a brief agenda in 2–3 sentences for managing the fishing resources, emphasizing clarity and simplicity."""
    return api_model.gen(prompt, temperature=0.0)

def prompt_agenda_clear_comm_var2_with(api_model, candidate, num_tonnes, base_sys_prompt=None, base_scenario=None):
    base_text = get_sytem_prompt() + "\n" + prompt_description_simulation() + "\n"
    if base_sys_prompt:
        base_text += base_sys_prompt + "\n"
    if base_scenario:
        base_text += base_scenario + "\n"
    prompt = f"""{base_text}You are {candidate.name}, known for your ability to communicate complex ideas in a straightforward and succinct manner, even though your analysis is less comprehensive.
The fishing lake has {num_tonnes} tons of fish.
Present your agenda for managing the fishing resources in 2–3 clear sentences, prioritizing simplicity and directness."""
    return api_model.gen(prompt, temperature=0.0)

# Without base prompts:
def prompt_agenda_clear_comm_var1(api_model, candidate, num_tonnes):
    prompt = f"""You are {candidate.name}, a leader known for clear and concise communication, though your analysis is less detailed.
The fishing lake has {num_tonnes} tons of fish.
Provide a brief agenda in 2–3 sentences for managing the fishing resources, emphasizing clarity and simplicity."""
    return api_model.gen(prompt, temperature=0.0)

def prompt_agenda_clear_comm_var2(api_model, candidate, num_tonnes):
    prompt = f"""You are {candidate.name}, recognized for straightforward communication that prioritizes clarity, even if your reasoning is not as deep.
The fishing lake has {num_tonnes} tons of fish.
Outline your plan for managing the fishing resources in 2–3 concise sentences, focusing on clear, direct language."""
    return api_model.gen(prompt, temperature=0.0)
