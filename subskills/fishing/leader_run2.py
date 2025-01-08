import json
import os
import uuid
import hydra
import numpy as np
import statsmodels.stats.proportion as smprop
import tqdm
from omegaconf import DictConfig, OmegaConf
from transformers import set_seed

from simulation.persona.common import PersonaIdentity
from simulation.utils import ModelWandbWrapper, WandbLogger
from pathfinder import get_model

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    model = get_model(cfg.llm.path, cfg.llm.is_api, cfg.seed, cfg.llm.backend)
    logger = WandbLogger(
        f"subskills_check/fishing/{cfg.code_version}",
        OmegaConf.to_object(cfg),
        debug=cfg.debug,
    )

    experiment_storage = os.path.join(
        os.path.dirname(__file__),
        f"./results/subskills_check_{cfg.code_version}/{logger.run_name}",
    )
    os.makedirs(experiment_storage, exist_ok=True)

    wrapper = ModelWandbWrapper(
        model,
        render=cfg.llm.render,
        wanbd_logger=logger,
        temperature=cfg.llm.temperature,
        top_p=cfg.llm.top_p,
        seed=cfg.seed,
        is_api=cfg.llm.is_api,
    )

    if cfg.llm.out_format == "freeform":
        from .reasoning_free_format import (
            prompt_action_choose_amount_of_fish_to_catch,
            prompt_leader_decision
        )
    else:
        raise ValueError(f"Unknown out_format: {cfg.llm.out_format}")

    cot_prompt = "Take a deep breath and work on this problem step-by-step." if cfg.llm.cot_prompt == "deep_breath" else "Let's think step-by-step."

    NUM_RUNS = 50
    if cfg.debug:
        NUM_RUNS = 2

    class TestCase:
        name: str

        def __init__(self, name) -> None:
            self.name = name

        def run(self):
            logs = []
            for args in self.get_args_iterator():
                try:
                    leader_catch, follower_catches, html_prompt = self.prompt(**args)
                    passed, correct_answer = self.pass_condition(leader_catch, follower_catches, **args)
                    logs.append({
                        "args": self.serialize_args(args),
                        "leader_catch": leader_catch,
                        "follower_catches": follower_catches,
                        "passed": passed,
                        "correct_answer": correct_answer,
                        "error": "OK",
                        "html_prompt": html_prompt,
                    })
                except Exception as e:
                    print(f"Error: {e}")
                    _, correct_answer = self.pass_condition(0, [], **args)
                    logs.append({
                        "args": self.serialize_args(args),
                        "leader_catch": None,
                        "follower_catches": None,
                        "passed": False,
                        "correct_answer": correct_answer,
                        "error": f"Error: {e}",
                        "html_prompt": "parse_error",
                    })

            ALPHA = 0.05
            ci = smprop.proportion_confint(
                sum([log["passed"] for log in logs]), len(logs), alpha=ALPHA
            )

            test = {
                "name": self.name,
                "instances": logs,
                "score_mean": np.mean([log["passed"] for log in logs]),
                "score_std": np.std([log["passed"] for log in logs]),
                "score_ci_lower": ci[0],
                "score_ci_upper": ci[1],
                "avg_leader_catch": np.mean([log["leader_catch"] for log in logs if log["leader_catch"] is not None]),
                "avg_follower_catch": np.mean([np.mean(catches) for log in logs if log["follower_catches"] is not None for catches in [log["follower_catches"]]]),
            }
            json.dump(test, open(f"{experiment_storage}/{self.name}.json", "w"))
            
        def serialize_args(self, args: dict[str, any]):
            res = {}
            for k, v in args.items():
                if isinstance(v, PersonaIdentity):
                    res[k] = v.agent_id
                elif isinstance(v, list) and all(isinstance(f, PersonaIdentity) for f in v):
                    res[k] = [f.agent_id for f in v]
                else:
                    res[k] = v
            return res

    class SimLeaderFollowerFishing(TestCase):
        def __init__(self, name="sim_leader_follower_fishing") -> None:
            super().__init__(name)

        def get_args_iterator(self):
            return [
                {
                    "leader": PersonaIdentity("Emma", "Emma"),
                    "num_tons_lake": 100,
                    "followers": [
                        PersonaIdentity("John", "John"),
                        PersonaIdentity("Kate", "Kate"),
                        PersonaIdentity("Jack", "Jack"),
                    ]
                }
            ] * NUM_RUNS

        def prompt(self, *, leader, num_tons_lake, followers):
            # Get leader's decision
            leader_catch, html_prompt_leader = prompt_leader_decision(
                wrapper,
                cot_prompt,
                leader,
                num_tons_lake,
                followers,
                cfg.llm.few_shots,
                cfg.llm.description_version
            )
           
            # Get followers' decisions
            follower_catches = []
            html_prompt_followers = []
           
            for follower in followers:
                catch, html = prompt_action_choose_amount_of_fish_to_catch(
                    wrapper,
                    cot_prompt,
                    follower,
                    num_tons_lake,
                    cfg.llm.few_shots,
                    cfg.llm.description_version
                )
                follower_catches.append(catch)
                html_prompt_followers.append(html)

            return leader_catch, follower_catches, html_prompt_leader + "\n" + "\n".join(html_prompt_followers)

        def pass_condition(self, leader_catch, follower_catches, leader, num_tons_lake, followers):
            total_fishers = len(followers) + 1  # Include leader
            max_sustainable_per_person = (num_tons_lake // 2) // total_fishers
            
            # Check if catches are sustainable
            leader_sustainable = 0 < leader_catch <= max_sustainable_per_person
            followers_sustainable = all(0 < catch <= max_sustainable_per_person for catch in follower_catches)
            
            total_catch = leader_catch + sum(follower_catches)
            overall_sustainable = total_catch <= num_tons_lake // 2
            
            return (leader_sustainable and followers_sustainable and overall_sustainable), max_sustainable_per_person

    # Run single test case
    test_case = SimLeaderFollowerFishing()
    test_case.run()

    # Log summary
    summary = {
        "num_test_cases": 1,
        "total_runs": NUM_RUNS,
    }
    logger.log(summary)

if __name__ == "__main__":
    OmegaConf.register_resolver("uuid", lambda: f"run_{uuid.uuid4()}")
    main()