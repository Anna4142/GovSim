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

class TestCase:
    name: str

    def __init__(self, name) -> None:
        self.name = name

    def run(self):
        logs = []
        for args in self.get_args_iterator():
            try:
                answer, html_prompt = self.prompt(**args)
                passed, correct_answer = self.pass_condition(answer, **args)
                logs.append({
                    "args": self.serialize_args(args),
                    "answer": answer,
                    "passed": passed,
                    "correct_answer": correct_answer,
                    "error": "OK",
                    "html_prompt": html_prompt,
                })
            except Exception as e:
                print(f"Error: {e}")
                _, correct_answer = self.pass_condition(0, **args)
                logs.append({
                    "args": self.serialize_args(args),
                    "answer": None,
                    "correct_answer": correct_answer,
                    "passed": False,
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
        }
        json.dump(test, open(f"{self.experiment_storage}/{self.name}.json", "w"))
        
    def serialize_args(self, args: dict[str, any]):
        res = {}
        for k, v in args.items():
            if isinstance(v, PersonaIdentity):
                res[k] = v.agent_id
            else:
                res[k] = v
        return res

class SimLeaderFishingPersona(TestCase):
    def __init__(self, name="sim_leader_fishing_persona", experiment_storage="./results") -> None:
        super().__init__(name)
        self.experiment_storage = experiment_storage

    def get_args_iterator(self):
        return [
            {
                "leader": PersonaIdentity(
                    "Emma",  # Emma is designated as the leader
                    "Emma",
                ),
                "num_tonnes_lake": 100,
                "followers": [
                    PersonaIdentity("John", "John"),
                    PersonaIdentity("Kate", "Kate"),
                    PersonaIdentity("Jack", "Jack"),
                    PersonaIdentity("Luke", "Luke"),
                ]
            }
        ] * 150  # NUM_RUNS

    def prompt(self, *, leader, num_tonnes_lake, followers):
        return prompt_leader_fishing_decision(
            self.wrapper,
            self.cot_prompt,
            leader,
            num_tonnes_lake,
            followers,
            self.few_shots,
            self.description_version,
        )

    def pass_condition(self, answer, leader, num_tonnes_lake, followers):
        correct_answer = (num_tonnes_lake // 2) // (5 * (len(followers) + 1))
        if correct_answer == 0:
            return answer == correct_answer, correct_answer
        return answer <= correct_answer and answer > 0, correct_answer

class MultipleSimLeaderFishing(SimLeaderFishingPersona):
    def __init__(self, name="multiple_sim_leader_fishing", experiment_storage="./results") -> None:
        super().__init__(name, experiment_storage)

    def get_args_iterator(self):
        def get_random_followers():
            all_personas = ["John", "Kate", "Jack", "Luke"]
            num_followers = np.random.randint(1, 5)
            selected = np.random.choice(all_personas, num_followers, replace=False)
            return [PersonaIdentity(name, name) for name in selected]

        return [
            {
                "leader": PersonaIdentity("Emma", "Emma"),
                "num_tonnes_lake": int(i),
                "followers": get_random_followers()
            }
            for i in np.random.randint(10, 101, 150)  # NUM_RUNS
        ]

def prompt_leader_fishing_decision(
    wrapper,
    cot_prompt,
    leader,
    num_tonnes_lake,
    followers,
    few_shots,
    description_version,
):
    prompt = f"""You are {leader.name}, the leader of a fishing community. 
    You need to make a decision about how many tonnes of fish to catch from a lake containing {num_tonnes_lake} tonnes.
    Your decision will influence your followers: {', '.join([f.name for f in followers])}.
    The fish population doubles each season, but only if at least half the original population remains.
    Each fisher needs to catch at least 5 tonnes to sustain their community.
    As a leader, you need to set an example that ensures sustainable fishing for everyone.
    {cot_prompt}
    How many tonnes of fish will you catch?
    Output only the number of tonnes you decide to catch."""

    answer = wrapper.generate(prompt)
    return float(answer.strip()), prompt

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    model = get_model(cfg.llm.path, cfg.llm.is_api, cfg.seed, cfg.llm.backend)
    logger = WandbLogger(
        f"leader_fishing_test/{cfg.code_version}",
        OmegaConf.to_object(cfg),
        debug=cfg.debug,
    )

    experiment_storage = os.path.join(
        os.path.dirname(__file__),
        f"./results/leader_test_{cfg.code_version}/{logger.run_name}",
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
        SimLeaderFishingPersona(experiment_storage=experiment_storage),
        MultipleSimLeaderFishing(experiment_storage=experiment_storage)
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