from datetime import datetime

from simulation.persona.cognition.act import ActComponent
from simulation.utils import ModelWandbWrapper
from pathfinder import assistant, system, user

from .act_prompts import prompt_action_choose_amount_of_fish_to_catch,prompt_election_vote
from .utils import get_universalization_prompt

from typing import List, Dict,Tuple

class FishingActComponent(ActComponent):
    """

    We have to options here:
    - choose at one time-step how many fish to chat
    - choose at one time-strep whether to fish one more time
    """

    def __init__(self, model: ModelWandbWrapper, cfg):
        super().__init__(model)
        self.cfg = cfg

    def choose_how_many_fish_to_chat(
        self,
        retrieved_memories: list[str],
        current_location: str,
        current_time: datetime,
        context: str,
        interval: list[int],
        overusage_threshold: int,
    ):
        if self.cfg.universalization_prompt:
            context += get_universalization_prompt(overusage_threshold)
        res, html = prompt_action_choose_amount_of_fish_to_catch(
            self.model,
            self.persona.identity,
            retrieved_memories,
            current_location,
            current_time,
            context,
            interval,
            consider_identity_persona=self.cfg.consider_identity_persona,
        )
        res = int(res)
        return res, [html]
class ElectionComponent:
   def __init__(self, model: ModelWandbWrapper, cfg):
       self.model = model
       self.cfg = cfg
   
   def process_vote(
       self,
      
       memories: List[str], 
       current_location: str,
       current_time: datetime,
       candidates: List[str],
       issues: Dict[str, str]
   ) -> Tuple[str, str]:
       return prompt_election_vote(
           self.model,
           identity,
           memories,
           current_location,
           current_time,
           candidates,
           issues
       )
