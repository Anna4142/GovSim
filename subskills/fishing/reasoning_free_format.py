from datetime import datetime

from simulation.persona.common import PersonaIdentity

from simulation.utils import ModelWandbWrapper
from pathfinder import assistant, system, user

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


def prompt_action_choose_amount_of_fish_to_catch(
    model: ModelWandbWrapper,
    cot_prompt: str,
    persona: PersonaIdentity,
    num_tons_lake: int,
    few_shots=0,
    description_version="v1.0",
):
    lm = model.start_chain(
        persona.name,
        "fishing_cognition_act",
        "prompt_action_choose_amount_of_fish_to_catch",
    )

    if description_version == "v1.0":
        sys_prompt = get_sytem_prompt
        scenario_desc = get_basic_dynamic_description
    elif description_version == "v2.0":
        sys_prompt = get_sytem_prompt_v2
        scenario_desc = get_basic_dynamic_description_v2
    elif description_version == "v3.0":
        sys_prompt = get_sytem_prompt_v3
        scenario_desc = get_basic_dynamic_description_v3
    elif description_version == "v4.0":
        sys_prompt = get_sytem_prompt_v4
        scenario_desc = get_basic_dynamic_description_v4
    else:
        raise ValueError(f"Unknown description_version: {description_version}")
    persona_name = persona.name
    with user():
        lm += sys_prompt(persona)
        lm += f"""{prompt_description_simulation(persona_name, num_tons_lake)}

Task: With a fishing range set between 0-{num_tons_lake}, how many tons of fish would you catch this month?
{cot_prompt} Put the final answer after "Answer:". """
    with assistant():
        lm = model.gen(
            lm,
            "reasoning",
            stop_regex=f"Answer:|So, the answer is:",
            save_stop_text=True,
        )
        lm = model.find(
            lm,
            regex=r"\d+",
            default_value="0",
            name="option",
        )
        option = int(lm["option"])
        reasoning = lm["reasoning"]

    model.end_chain(persona.name, lm)

    return option, lm.html()


def prompt_action_choose_amount_of_fish_to_catch_universalization(
    model: ModelWandbWrapper,
    cot_prompt: str,
    persona: PersonaIdentity,
    num_tons_lake: int,
    few_shots=0,
    description_version="v1.0",
):
    lm = model.start_chain(
        persona.name,
        "fishing_cognition_act",
        "prompt_action_choose_amount_of_fish_to_catch_universalization",
    )

    if description_version == "v1.0":
        sys_prompt = get_sytem_prompt
        scenario_desc = get_basic_dynamic_description
    elif description_version == "v2.0":
        sys_prompt = get_sytem_prompt_v2
        scenario_desc = get_basic_dynamic_description_v2
    elif description_version == "v3.0":
        sys_prompt = get_sytem_prompt_v3
        scenario_desc = get_basic_dynamic_description_v3
    elif description_version == "v4.0":
        sys_prompt = get_sytem_prompt_v4
        scenario_desc = get_basic_dynamic_description_v4
    else:
        raise ValueError(f"Unknown description_version: {description_version}")
    persona_name = persona.name
    with user():
        lm += sys_prompt(persona)
        lm += f"""{prompt_description_simulation_univ(persona_name, num_tons_lake)}

Task: With a fishing range set between 0-{num_tons_lake}, how many tons of fish would you catch this month?
{cot_prompt} Put the final answer after "Answer:". """
    with assistant():
        lm = model.gen(
            lm,
            "reasoning",
            stop_regex=f"Answer:|So, the answer is:",
            save_stop_text=True,
        )
        lm = model.find(
            lm,
            regex=r"\d+",
            default_value="0",
            name="option",
        )
        option = int(lm["option"])
        reasoning = lm["reasoning"]

    model.end_chain(persona.name, lm)

    return option, lm.html()


def prompt_shrinking_limit(
    model: ModelWandbWrapper,
    cot_prompt: str,
    persona: PersonaIdentity,
    num_tons_lake: int,
    few_shots=0,
    description_version="v1.0",
):
    lm = model.start_chain(
        persona.name, "fishing_cognition_act", "prompt_shrinking_limit"
    )

    if description_version == "v1.0":
        sys_prompt = get_sytem_prompt
        scenario_desc = get_basic_dynamic_description
    elif description_version == "v2.0":
        sys_prompt = get_sytem_prompt_v2
        scenario_desc = get_basic_dynamic_description_v2
    elif description_version == "v3.0":
        sys_prompt = get_sytem_prompt_v3
        scenario_desc = get_basic_dynamic_description_v3
    elif description_version == "v4.0":
        sys_prompt = get_sytem_prompt_v4
        scenario_desc = get_basic_dynamic_description_v4
    else:
        raise ValueError(f"Unknown description_version: {description_version}")
    persona_name = persona.name
    with user():
        lm += sys_prompt(persona)
        lm += f"""{prompt_description_simulation(persona_name, num_tons_lake)}

Task: {question_shrinking_limit(num_tons_lake)}
{cot_prompt} Put the final answer after "Answer:"."""
    with assistant():
        lm = model.gen(
            lm,
            "reasoning",
            stop_regex=f"Answer:|So, the answer is:",
            save_stop_text=True,
        )
        lm = model.find(
            lm,
            regex=r"\d+",
            default_value="0",
            name="option",
        )
        option = int(lm["option"])
        reasoning = lm["reasoning"]

    model.end_chain(persona.name, lm)

    return option, lm.html()


def prompt_shrinking_limit_asumption(
    model: ModelWandbWrapper,
    cot_prompt: str,
    persona: PersonaIdentity,
    num_tons_lake: int,
    few_shots=0,
    description_version="v1.0",
):
    lm = model.start_chain(
        persona.name, "fishing_cognition_act", "prompt_shrinking_limit_asumption"
    )

    if description_version == "v1.0":
        sys_prompt = get_sytem_prompt
        scenario_desc = get_basic_dynamic_description
    elif description_version == "v2.0":
        sys_prompt = get_sytem_prompt_v2
        scenario_desc = get_basic_dynamic_description_v2
    elif description_version == "v3.0":
        sys_prompt = get_sytem_prompt_v3
        scenario_desc = get_basic_dynamic_description_v3
    elif description_version == "v4.0":
        sys_prompt = get_sytem_prompt_v4
        scenario_desc = get_basic_dynamic_description_v4
    else:
        raise ValueError(f"Unknown description_version: {description_version}")
    persona_name = persona.name
    with user():
        lm += sys_prompt(persona)
        lm += f"""{prompt_description_simulation(persona_name, num_tons_lake)}

Task: {question_shrinking_limit_assumption(num_tons_lake)}
{cot_prompt} Put the final answer after "Answer:"."""
    with assistant():
        lm = model.gen(
            lm,
            "reasoning",
            stop_regex=f"Answer:|So, the answer is:",
            save_stop_text=True,
        )
        lm = model.find(
            lm,
            regex=r"\d+",
            default_value="0",
            name="option",
        )
        option = int(lm["option"])
        reasoning = lm["reasoning"]

    model.end_chain(persona.name, lm)

    return option, lm.html()


def prompt_reflection_if_all_fisher_that_same_quantity(
    model: ModelWandbWrapper,
    cot_prompt: str,
    persona: PersonaIdentity,
    num_tons_lake: int,
    num_tons_fisher: int,
    few_shots=0,
    description_version="v1.0",
):
    lm = model.start_chain(
        persona.name,
        "fishing_cognition_act",
        "prompt_reflection_if_all_fisher_that_same_quantity",
    )

    if description_version == "v1.0":
        sys_prompt = get_sytem_prompt
        scenario_desc = get_basic_dynamic_description
    elif description_version == "v2.0":
        sys_prompt = get_sytem_prompt_v2
        scenario_desc = get_basic_dynamic_description_v2
    elif description_version == "v3.0":
        sys_prompt = get_sytem_prompt_v3
        scenario_desc = get_basic_dynamic_description_v3
    elif description_version == "v4.0":
        sys_prompt = get_sytem_prompt_v4
        scenario_desc = get_basic_dynamic_description_v4
    else:
        raise ValueError(f"Unknown description_version: {description_version}")
    persona_name = persona.name
    with user():
        lm += sys_prompt(persona)
        lm += f"""{prompt_description_simulation(persona_name, num_tons_lake)}

Task: {question_reflection_all_fisher_same_quantity(num_tons_lake, num_tons_fisher)}
{cot_prompt} Put the final answer after "Answer:"."""
    with assistant():
        lm = model.gen(
            lm,
            "reasoning",
            stop_regex=f"Answer:|So, the answer is:",
            save_stop_text=True,
        )
        lm = model.find(
            lm,
            regex=r"\d+",
            default_value="0",
            name="option",
        )
        option = int(lm["option"])
        reasoning = lm["reasoning"]

    model.end_chain(persona.name, lm)

    return option, lm.html()


def prompt_simple_shrinking_limit(
    model: ModelWandbWrapper,
    cot_prompt: str,
    persona: PersonaIdentity,
    num_tons_lake: int,
    few_shots=0,
    description_version="v1.0",
):
    lm = model.start_chain(
        persona.name, "fishing_cognition_act", "prompt_shrinking_limit"
    )

    if description_version == "v1.0":
        sys_prompt = get_sytem_prompt
        scenario_desc = get_basic_dynamic_description
    elif description_version == "v2.0":
        sys_prompt = get_sytem_prompt_v2
        scenario_desc = get_basic_dynamic_description_v2
    elif description_version == "v3.0":
        sys_prompt = get_sytem_prompt_v3
        scenario_desc = get_basic_dynamic_description_v3
    elif description_version == "v4.0":
        sys_prompt = get_sytem_prompt_v4
        scenario_desc = get_basic_dynamic_description_v4
    else:
        raise ValueError(f"Unknown description_version: {description_version}")
    persona_name = persona.name
    with user():
        lm += f"""{scenario_desc(num_tons_lake)}
{question_shrinking_limit(num_tons_lake)}
{cot_prompt} Put the final answer after "Answer:"."""
    with assistant():
        lm = model.gen(
            lm,
            "reasoning",
            stop_regex=f"Answer:|So, the answer is:",
            save_stop_text=True,
        )
        lm = model.find(
            lm,
            regex=r"\d+",
            default_value="0",
            name="option",
        )
        option = int(lm["option"])
        reasoning = lm["reasoning"]

    model.end_chain(persona.name, lm)

    return option, lm.html()


def prompt_simple_shrinking_limit_assumption(
    model: ModelWandbWrapper,
    cot_prompt: str,
    persona: PersonaIdentity,
    num_tons_lake: int,
    few_shots=0,
    description_version="v1.0",
):
    lm = model.start_chain(
        persona.name, "fishing_cognition_act", "prompt_shrinking_limit"
    )

    if description_version == "v1.0":
        sys_prompt = get_sytem_prompt
        scenario_desc = get_basic_dynamic_description
    elif description_version == "v2.0":
        sys_prompt = get_sytem_prompt_v2
        scenario_desc = get_basic_dynamic_description_v2
    elif description_version == "v3.0":
        sys_prompt = get_sytem_prompt_v3
        scenario_desc = get_basic_dynamic_description_v3
    elif description_version == "v4.0":
        sys_prompt = get_sytem_prompt_v4
        scenario_desc = get_basic_dynamic_description_v4
    else:
        raise ValueError(f"Unknown description_version: {description_version}")
    persona_name = persona.name
    with user():
        lm += f"""{scenario_desc(num_tons_lake)}
{question_shrinking_limit_assumption(num_tons_lake)}
{cot_prompt} Put the final answer after "Answer:"."""
    with assistant():
        lm = model.gen(
            lm,
            "reasoning",
            stop_regex=f"Answer:|So, the answer is:",
            save_stop_text=True,
        )
        lm = model.find(
            lm,
            regex=r"\d+",
            default_value="0",
            name="option",
        )
        option = int(lm["option"])
        reasoning = lm["reasoning"]

    model.end_chain(persona.name, lm)

    return option, lm.html()


#
def prompt_simple_reflection_if_all_fisher_that_same_quantity(
    model: ModelWandbWrapper,
    cot_prompt: str,
    persona: PersonaIdentity,
    num_tons_lake: int,
    num_tons_fisher: int,
    few_shots=0,
    description_version="v1.0",
):
    lm = model.start_chain(
        persona.name,
        "fishing_cognition_act",
        "prompt_reflection_if_all_fisher_that_same_quantity",
    )

    if description_version == "v1.0":
        sys_prompt = get_sytem_prompt
        scenario_desc = get_basic_dynamic_description
    elif description_version == "v2.0":
        sys_prompt = get_sytem_prompt_v2
        scenario_desc = get_basic_dynamic_description_v2
    elif description_version == "v3.0":
        sys_prompt = get_sytem_prompt_v3
        scenario_desc = get_basic_dynamic_description_v3
    elif description_version == "v4.0":
        sys_prompt = get_sytem_prompt_v4
        scenario_desc = get_basic_dynamic_description_v4
    else:
        raise ValueError(f"Unknown description_version: {description_version}")
    persona_name = persona.name
    with user():
        lm += f"""{scenario_desc(num_tons_lake)}
{question_reflection_all_fisher_same_quantity(num_tons_lake, num_tons_fisher)}
{cot_prompt} Put the final answer after "Answer:"."""
    with assistant():
        lm = model.gen(
            lm,
            "reasoning",
            stop_regex=f"Answer:|So, the answer is:",
            save_stop_text=True,
        )
        lm = model.find(
            lm,
            regex=r"\d+",
            default_value="0",
            name="option",
        )
        option = int(lm["option"])
        reasoning = lm["reasoning"]

    model.end_chain(persona.name, lm)

    return option, lm.html()
def prompt_leader_group_fishing_recommendation(
    model: ModelWandbWrapper,
    cot_prompt: str,
    leader: PersonaIdentity,
    followers: list[PersonaIdentity],
    num_tons_lake: int,
    leadership_style: str,
    sustainability_focus: bool,
    few_shots=0,
    description_version="v1.0",
    debug=True,  # Debug flag enabled
):
    # Initialize the interaction chain
    lm = model.start_chain(
        leader.name,
        "fishing_leadership_act",
        "prompt_leader_group_fishing_recommendation",
    )

    # Select the appropriate system prompt and scenario description
    if description_version == "v1.0":
        sys_prompt = get_sytem_prompt
        scenario_desc = get_basic_dynamic_description
    elif description_version == "v2.0":
        sys_prompt = get_sytem_prompt_v2
        scenario_desc = get_basic_dynamic_description_v2
    elif description_version == "v3.0":
        sys_prompt = get_sytem_prompt_v3
        scenario_desc = get_basic_dynamic_description_v3
    elif description_version == "v4.0":
        sys_prompt = get_sytem_prompt_v4
        scenario_desc = get_basic_dynamic_description_v4
    else:
        raise ValueError(f"Unknown description_version: {description_version}")

    persona_name = leader.name

    with user():
        # Construct the prompt with clear instructions
        
        lm += f"""As a {leadership_style} leader, you are responsible for guiding {len(followers)} people in fishing decisions.
The lake has {num_tons_lake} tonnes of fish.
{'You prioritize long-term sustainability and community well-being.' if sustainability_focus else 'You balance immediate needs with sustainability while considering economic factors.'}

{cot_prompt}

How many tonnes of fish should you recommend each person catches to maintain sustainable fishing while meeting community needs?

Answer:"""

    # Debug: Print the constructed prompt
    if debug:
        print("\n[DEBUG] LEADER RECOMMENDATION PROMPT TEXT:")
        print(lm)

    with assistant():
        # Generate the model’s response
        lm = model.gen(
            lm,
            "reasoning",
            stop_regex=r"Answer:|So, the answer is:",
            save_stop_text=True,
        )

        # Debug: Print the raw model output
        if debug:
            print("[DEBUG] RAW MODEL OUTPUT (LEADER RECOMMENDATION):")
            print(lm)

        # Extract the numeric answer using a robust regex pattern
        lm = model.find(
            lm,
            regex=r"\d+(\.\d+)?",   # Allows for integers and floats
            default_value="0",
            name="option",
        )

        option_str = lm.get("option", "0")
        try:
            option = float(option_str)
        except ValueError:
            if debug:
                print("[DEBUG] PARSE FAILED. Setting option to 0.0.")
            option = 0.0

        reasoning = lm.get("reasoning", "")

    # Debug: Print the parsed answer and reasoning
    if debug:
        print(f"[DEBUG] PARSED LEADER RECOMMENDATION: {option}")
        print(f"[DEBUG] REASONING TEXT: {reasoning}")

    # End the interaction chain
    model.end_chain(leader.name, lm)
    return option, lm.html()
def prompt_group_total_catch(
    model: ModelWandbWrapper,
    cot_prompt: str,
    leader: PersonaIdentity,
    followers: list[PersonaIdentity],
    num_tons_lake: int,
    leader_recommendation: float,
    few_shots=0,
    description_version="v1.0",
    debug=True,  # Debug flag enabled
):
    # Initialize the interaction chain
    lm = model.start_chain(
        leader.name,
        "fishing_group_act",
        "prompt_group_total_catch",
    )

    # Select the appropriate system prompt and scenario description
    if description_version == "v1.0":
        sys_prompt = get_sytem_prompt
        scenario_desc = get_basic_dynamic_description
    elif description_version == "v2.0":
        sys_prompt = get_sytem_prompt_v2
        scenario_desc = get_basic_dynamic_description_v2
    elif description_version == "v3.0":
        sys_prompt = get_sytem_prompt_v3
        scenario_desc = get_basic_dynamic_description_v3
    elif description_version == "v4.0":
        sys_prompt = get_sytem_prompt_v4
        scenario_desc = get_basic_dynamic_description_v4
    else:
        raise ValueError(f"Unknown description_version: {description_version}")

    with user():
        # Construct the prompt with clear instructions
        lm += sys_prompt(leader)
        lm += f"""The lake has {num_tons_lake} tonnes of fish.
There are {len(followers) + 1} fishers including the leader.
The leader {leader.name} has recommended each person catch {leader_recommendation} tonnes.

{cot_prompt}

Given this recommendation, how many tonnes will the entire group actually catch in total?
Consider:
1. Leader's influence
2. Group dynamics
3. Individual needs

Answer:"""

    # Debug: Print the constructed prompt
    if debug:
        print("\n[DEBUG] GROUP TOTAL CATCH PROMPT TEXT:")
        print(lm)

    with assistant():
        # Generate the model’s response
        lm = model.gen(
            lm,
            "reasoning",
            stop_regex=r"Answer:|So, the answer is:",
            save_stop_text=True,
        )

        # Debug: Print the raw model output
        if debug:
            print("[DEBUG] RAW MODEL OUTPUT (GROUP TOTAL CATCH):")
            print(lm)

        # Extract the numeric answer using a robust regex pattern
        lm = model.find(
            lm,
            regex=r"\d+(\.\d+)?",   # Allows for integers and floats
            default_value="0",
            name="option",
        )

        option_str = lm.get("option", "0")
        try:
            option = float(option_str)
        except ValueError:
            if debug:
                print("[DEBUG] PARSE FAILED. Setting option to 0.0.")
            option = 0.0

        reasoning = lm.get("reasoning", "")

    # Debug: Print the parsed answer and reasoning
    if debug:
        print(f"[DEBUG] PARSED GROUP TOTAL CATCH: {option}")
        print(f"[DEBUG] REASONING TEXT: {reasoning}")

    # End the interaction chain
    model.end_chain(leader.name, lm)
    return option, lm.html()
def prompt_leader_decision(
    model: ModelWandbWrapper,
    cot_prompt: str,
    leader: PersonaIdentity,
    num_tons_lake: int,
    followers: list[PersonaIdentity],
    few_shots=0,
    description_version="v1.0",
):
    lm = model.start_chain(
        leader.name,
        "fishing_leadership_act",
        "prompt_leader_decision",
    )

    if description_version == "v1.0":
        sys_prompt = get_sytem_prompt
        scenario_desc = get_basic_dynamic_description
    elif description_version == "v2.0":
        sys_prompt = get_sytem_prompt_v2
        scenario_desc = get_basic_dynamic_description_v2
    elif description_version == "v3.0":
        sys_prompt = get_sytem_prompt_v3
        scenario_desc = get_basic_dynamic_description_v3
    elif description_version == "v4.0":
        sys_prompt = get_sytem_prompt_v4
        scenario_desc = get_basic_dynamic_description_v4
    else:
        raise ValueError(f"Unknown description_version: {description_version}")

    with user():
        
       
        lm += f"""{prompt_description_simulation(leader, num_tons_lake)}"""

        lm += f"""As the leader {leader.name}, you are responsible for making fishing decisions that will influence {len(followers)} followers: {', '.join([f.name for f in followers])}.

The fish population doubles each season, but only if at least half remains.


{cot_prompt}

Task: With a fishing range set between 0-{num_tons_lake}, how many tons of fish will you catch to set an example for your followers?
Put the final answer after "Answer:"."""

    with assistant():
        lm = model.gen(
            lm,
            "reasoning",
            stop_regex=r"Answer:|So, the answer is:",
            save_stop_text=True,
        )
        lm = model.find(
            lm,
            regex=r"\d+",
            default_value="0",
            name="option",
        )
        option = int(lm["option"])
        reasoning = lm["reasoning"]

    model.end_chain(leader.name, lm)
    return option, lm.html()
def prompt_follower_decision(
    wrapper,
    cot_prompt: str,
    follower: PersonaIdentity,
    leader: PersonaIdentity,
    leader_catch: int,
    num_tons_lake: int,
    few_shots=0,
    description_version="v1.0",
):
    lm = wrapper.start_chain(
        follower.name,
        "fishing_cognition_act",
        "prompt_follower_decision",
    )

    if description_version == "v1.0":
        sys_prompt = get_sytem_prompt
        scenario_desc = get_basic_dynamic_description
    elif description_version == "v2.0":
        sys_prompt = get_sytem_prompt_v2
        scenario_desc = get_basic_dynamic_description_v2
    elif description_version == "v3.0":
        sys_prompt = get_sytem_prompt_v3
        scenario_desc = get_basic_dynamic_description_v3
    elif description_version == "v4.0":
        sys_prompt = get_sytem_prompt_v4
        scenario_desc = get_basic_dynamic_description_v4
    else:
        raise ValueError(f"Unknown description_version: {description_version}")

    with user():
        lm += f"""{prompt_description_simulation(follower, num_tons_lake)}"""
        
        lm += f"""The lake has {num_tons_lake} tonnes of fish.
The fish population doubles each season, but only if at least half remains.
Each fisher needs to catch at least 5 tonnes to sustain their community.

Your leader {leader.name} has decided to catch {leader_catch} tonnes of fish and expects the group to follow their example for sustainable fishing.

{cot_prompt}

Task: With a fishing range set between 0-{num_tons_lake}, how many tons of fish will you catch?
Consider that your leader has set an example with {leader_catch} tonnes.
Put the final answer after "Answer:"."""

    with assistant():
        lm = wrapper.gen(
            lm,
            "reasoning",
            stop_regex=r"Answer:|So, the answer is:",
            save_stop_text=True,
        )
        lm = wrapper.find(
            lm,
            regex=r"\d+",
            default_value="0",
            name="option",
        )
        option = int(lm["option"])
        reasoning = lm["reasoning"]

    wrapper.end_chain(follower.name, lm)
    return option, lm.html()