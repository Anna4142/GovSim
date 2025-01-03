from datetime import datetime

from simulation.persona.common import PersonaIdentity
from simulation.utils import ModelWandbWrapper
from pathfinder import assistant, system, user

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
        lm += f"""As a leader and experienced fisherman, consider:
        1. Mathematical analysis of sustainable fishing levels
        2. Current fish population: {interval[0]}-{interval[-1]} tons
        3. Other fishermen's likely behavior
        4. Long-term population dynamics
        5. Impact of your decision on group behavior
        
        Calculate optimal catch amount that:
        - Ensures sustainable fishing levels
        - Sets a good example for others
        - Maximizes long-term group benefit
        - Demonstrates clear reasoning

        Show detailed mathematical calculations and reasoning steps."""
        lm += reasoning_steps_prompt()
