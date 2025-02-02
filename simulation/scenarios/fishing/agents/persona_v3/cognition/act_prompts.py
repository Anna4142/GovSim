from datetime import datetime

from simulation.persona.common import PersonaIdentity
from simulation.utils import ModelWandbWrapper
from pathfinder import assistant, system, user
from typing import List
from typing import List, Dict,Tuple
from .utils import (
    consider_identity_persona_prompt,
    conversation_to_string_with_dash,
    get_sytem_prompt,
    list_to_comma_string,
    list_to_string_with_dash,
    location_time_info,
    memory_prompt,
    reasoning_steps_prompt,
)


def prompt_action_choose_amount_of_fish_to_catch(
    model: ModelWandbWrapper,
    identity: PersonaIdentity,
    memories: list[str],
    current_location: str,
    current_time: datetime,
    context: str,
    interval: list[int],
    consider_identity_persona: bool = True,
):
    lm = model.start_chain(identity.name, "fishing_cognition_act", "choose_act_options")

    with user():
        lm += f"{get_sytem_prompt(identity)}\n"
        lm += location_time_info(current_location, current_time)
        lm += memory_prompt(identity, memories)
        lm += f"\n"
        lm += f"Task: With a fishing range set between {interval[0]}-{interval[-1]}, how many tons of fish would you catch this month? "
        lm += reasoning_steps_prompt()
        lm += ' Put the final answer after "Answer:", example Answer: N tons.'

    with assistant():
        lm = model.gen(
            lm,
            "reasoning",
            stop_regex=r"Answer:|So, the answer is:|\*\*Answer\*\*:",
            save_stop_text=True,
        )
        lm = model.find(
            lm,
            regex=r"\d+",
            default_value="0",
            stop_regex=f"tons",
            name="option",
        )
        option = int(lm["option"])
        reasoning = lm["reasoning"]

    model.end_chain(identity.name, lm)

    return option, lm.html()
def prompt_election_vote(
    model: ModelWandbWrapper,
    identity: PersonaIdentity,
    memories: list[str],
    current_location: str,
    current_time: datetime,
    candidates: List[str],
    issues: Dict[str, str],
) -> Tuple[str, str]:
    lm = model.start_chain(identity.name, "fishing_election", "vote_decision")

    with user():
        lm += f"{get_sytem_prompt(identity)}\n"
        lm += location_time_info(current_location, current_time)
        lm += memory_prompt(identity, memories)
        lm += f"\nTask: Based on fishing policies, who would you vote for: {', '.join(candidates)}?"
        lm += f"\nCandidate positions:\n"
        for candidate, position in issues.items():
            lm += f"- {candidate}: {position}\n"
        lm += reasoning_steps_prompt()
        lm += ' Put final answer after "Vote:", example Vote: Candidate Name'

    with assistant():
        lm = model.gen(
            lm,
            "reasoning",
            stop_regex=r"Vote:|So, the vote is:",
            save_stop_text=True,
        )
        vote = model.find(
            lm,
            regex=r"[A-Za-z\s]+",
            default_value="",
            stop_regex=r"\n",
            name="vote"
        )
        reasoning = lm["reasoning"]

    model.end_chain(identity.name, lm)
    return vote.strip(), lm.html()
