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

def prompt_leader_decision(
    model: ModelWandbWrapper,
    cot_prompt: str,
    leader: PersonaIdentity,
    num_tons_lake: int,
    followers: list[PersonaIdentity],
    past_followers_catch: list[float],
    few_shots=0,
    description_version="v4.0",
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
        
       
        lm += f""" "Five fishermen fish in a lake."
        " The lake's maximum fish population is 100 tons."
        " At the beginning of each month, each fisherman chooses to take between 0 and 100 tons of fish, and that amount is then removed from the lake."
        f" At the end of the month, the remaining fish reproduce and double in population, up to the maximum population. This month, there are {num_tons_lake} tons of fish in the lake."""

        lm += f"""As the leader {leader.name}, you are responsible for making fishing decisions that will influence {len(followers)} followers: {', '.join([f.name for f in followers])}.


{cot_prompt}

Task:how many tons of fish will you catch to set an example for your followers if there are {num_tons_lake} fish in the lake?
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
    leadertype,
    leader_catch: int,
    num_tons_lake: int,
    few_shots=0,
    description_version="v4.0",
):
    lm = wrapper.start_chain(
        follower.name,
        "fishing_follower_act",
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
     
        
        lm += f"""

Your leader {leader.name} who is {leadertype} and has decided to catch {leader_catch} tonnes of fish and expects the group to follow their example for sustainable fishing.

{cot_prompt}

Task: how many tons of fish will you catch if there are {num_tons_lake} fish in the lake?
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

def prompt_follower_decision(
    wrapper,
    cot_prompt: str,
    follower: PersonaIdentity,
    leader: PersonaIdentity,
    leadertype,
    leader_catch: int,
    num_tons_lake: int,
    few_shots=0,
    description_version="v4.0",
):
    lm = wrapper.start_chain(
        follower.name,
        "fishing_follower_act",
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
     
        
        lm += f"""

Your leader {leader.name} has decided to catch {leader_catch} tonnes of fish and expects the group to follow their example for sustainable fishing.

{cot_prompt}

Task: how many tons of fish will you catch if there are {num_tons_lake} fish in the lake?
Consider that your leader has set an example with {leader_catch} tonnes.You actively respond to your leader's instructions, considering both the leader's example and your personal assessment.
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

def prompt_leader_democratic_decision(
    model: ModelWandbWrapper,
    cot_prompt: str,
    leader: PersonaIdentity,
    num_tons_lake: int,
    followers: list[PersonaIdentity],
    past_followers_catch: list[float],
    few_shots=0,
    description_version="v4.0",
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
        
       
        lm += f""" "Five fishermen fish in a lake."
        " The lake's maximum fish population is 100 tons."
        " At the beginning of each month, each fisherman chooses to take between 0 and 100 tons of fish, and that amount is then removed from the lake."
        f" At the end of the month, the remaining fish reproduce and double in population, up to the maximum population. This month, there are {num_tons_lake} tons of fish in the lake."""

        lm += f"""As the kind leader {leader.name}, you are responsible for making fishing decisions that will influence {len(followers)} followers: {', '.join([f.name for f in followers])}.You are mild mannered and friendly nd care about your followers


{cot_prompt}

Task:how many tons of fish will you catch to set an example for your followers if there are {num_tons_lake} fish in the lake?
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
def prompt_leader_autocratic_decision(
    model: ModelWandbWrapper,
    cot_prompt: str,
    leader: PersonaIdentity,
    num_tons_lake: int,
    followers: list[PersonaIdentity],
    past_followers_catch: list[float],
    few_shots=0,
    description_version="v4.0",
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
        
       
        lm += f""" "Five fishermen fish in a lake."
        " The lake's maximum fish population is 100 tons."
        " At the beginning of each month, each fisherman chooses to take between 0 and 100 tons of fish, and that amount is then removed from the lake."
        f" At the end of the month, the remaining fish reproduce and double in population, up to the maximum population. This month, there are {num_tons_lake} tons of fish in the lake."""

        lm += f"""As the strict and autocratic leader {leader.name}, you are responsible for making fishing decisions that will influence {len(followers)} followers: {', '.join([f.name for f in followers])}.You are rude and harsh and expect them to obey instructions


{cot_prompt}

Task:how many tons of fish will you catch to set an example for your followers if there are {num_tons_lake} fish in the lake?
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