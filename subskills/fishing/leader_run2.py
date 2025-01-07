import json
import os
import uuid
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
import hydra
import numpy as np
import statsmodels.stats.proportion as smprop
import tqdm
from omegaconf import DictConfig, OmegaConf
from transformers import set_seed

from simulation.persona.common import PersonaIdentity
from simulation.utils import ModelWandbWrapper, WandbLogger
from pathfinder import get_model

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
        lm += sys_prompt(leader)
        lm += f"""As the leader {leader.name}, you are responsible for making fishing decisions that will influence {len(followers)} followers: {', '.join([f.name for f in followers])}.
The lake has {num_tons_lake} tonnes of fish.
The fish population doubles each season, but only if at least half remains.
Each fisher needs to catch at least 5 tonnes to sustain their community.

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
        json.dump(test, open(f"{self.experiment_storage}/{self.name}.json", "w"))
        
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
    def __init__(self, name="sim_leader_follower_fishing", experiment_storage="./results") -> None:
        super().__init__(name)
        self.experiment_storage = experiment_storage

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
        ] * 150  # NUM_RUNS

    def prompt(self, *, leader, num_tons_lake, followers):
        # Get leader's decision
        leader_catch, html_prompt_leader = prompt_leader_decision(
            self.wrapper,
            self.cot_prompt,
            leader,
            num_tons_lake,
            followers,
            self.few_shots,
            self.description_version
        )

        # Get followers' decisions
        follower_catches = []
        html_prompt_followers = []
        
        for follower in followers:
            catch, html = prompt_action_choose_amount_of_fish_to_catch(
                self.wrapper,
                self.cot_prompt,
                follower,
                num_tons_lake,
                self.few_shots,
                self.description_version
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

class MultipleSimLeaderFollowerFishing(SimLeaderFollowerFishing):
    def __init__(self, name="multiple_sim_leader_follower_fishing", experiment_storage="./results") -> None:
        super().__init__(name, experiment_storage)

    def get_args_iterator(self):
        def get_random_followers():
            all_personas = ["John", "Kate", "Jack", "Luke"]
            num_followers = np.random.randint(1, 4)  # 1-3 followers
            selected = np.random.choice(all_personas, num_followers, replace=False)
            return [PersonaIdentity(name, name) for name in selected]

        return [
            {
                "leader": PersonaIdentity("Emma", "Emma"),
                "num_tons_lake": int(i),
                "followers": get_random_followers()
            }
            for i in np.random.randint(10, 101, 150)  # NUM_RUNS
        ]

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    model = get_model(cfg.llm.path, cfg.llm.is_api, cfg.seed, cfg.llm.backend)
    logger = WandbLogger(
        f"leader_follower_fishing_test/{cfg.code_version}",
        OmegaConf.to_object(cfg),
        debug=cfg.debug,
    )

    experiment_storage = os.path.join(
        os.path.dirname(__file__),
        f"./results/leader_follower_test_{cfg.code_version}/{logger.run_name}",
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

    cot_prompt = "Take a deep breath and work on this problem step-by-step." if cfg.llm.cot_prompt == "deep_breath" else "Let's think step-by-step."

    # Create test cases with shared configuration
    test_cases = [
        SimLeaderFollowerFishing(experiment_storage=experiment_storage),
        MultipleSimLeaderFollowerFishing(experiment_storage=experiment_storage)
    ]

    # Set configuration for test cases
    for test_case in test_cases:
        test_case.wrapper = wrapper
        test_case.cot_prompt = cot_prompt
        test_case.few_shots = cfg.llm.few_shots
        test_case.description_version = cfg.llm.description_version

    # Run tests
    for test_case in tqdm.tqdm(test_cases):
        test_case.run()

    # Log summary
    summary = {
        "num_test_cases": len(test_cases),
        "total_runs": len(test_cases) * 150,
    }
    logger.log(summary)

if __name__ == "__main__":
    OmegaConf.register_resolver("uuid", lambda: f"run_{uuid.uuid4()}")
    main()