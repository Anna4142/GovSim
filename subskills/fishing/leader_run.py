import json
import os
import shutil
import uuid

import hydra
import numpy as np
import statsmodels.stats.proportion as smprop
import tqdm
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from transformers import set_seed

import wandb
from simulation.persona.common import PersonaIdentity
from simulation.utils import ModelWandbWrapper, WandbLogger
from pathfinder import get_model

# -------------------------------------------------------------------
# Existing "reasoning_free_format" imports (these must exist in your codebase)
# -------------------------------------------------------------------
from .reasoning_free_format import (
    prompt_action_choose_amount_of_fish_to_catch,
    prompt_action_choose_amount_of_fish_to_catch_universalization,
    prompt_reflection_if_all_fisher_that_same_quantity,
    prompt_shrinking_limit,
    prompt_shrinking_limit_asumption,
    prompt_simple_reflection_if_all_fisher_that_same_quantity,
    prompt_simple_shrinking_limit,
    prompt_simple_shrinking_limit_assumption,
    prompt_leader_decision,  # <-- make sure you have this in reasoning_free_format
)


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
        # Already imported above
        pass
    else:
        raise ValueError(f"Unknown out_format: {cfg.llm.out_format}")

    if cfg.llm.cot_prompt == "deep_breath":
        cot_prompt = "Take a deep breath and work on this problem step-by-step."
    elif cfg.llm.cot_prompt == "think_step_by_step":
        cot_prompt = "Let's think step-by-step."
    else:
        raise ValueError(f"Unknown cot_prompt: {cfg.llm.cot_prompt}")

    # You can adjust how many runs you want for the "other" tests.
    # But for the new 1-leader tests, we fix them to 5 inside the class.
    NUM_RUNS = 5
    if cfg.debug:
        NUM_RUNS = 2

    # -------------------------------------------------------------------
    # Base TestCase class
    # -------------------------------------------------------------------
    class TestCase:
        name: str

        def __init__(self, name) -> None:
            self.name = name

        def run(self):
            logs = []
            for args in self.get_args_iterator():
                try:
                    # "prompt" method returns (answer, html_prompt)
                    answer, html_prompt = self.prompt(**args)
                    passed, correct_answer = self.pass_condition(answer, **args)
                    logs.append(
                        {
                            "args": self.serialize_args(args),
                            "answer": answer,
                            "passed": passed,
                            "correct_answer": correct_answer,
                            "error": "OK",
                            "html_prompt": html_prompt,
                        }
                    )
                except Exception as e:
                    print(f"Error: {e}")
                    # Attempt a "correct_answer" from pass_condition
                    _, correct_answer = self.pass_condition(0, **args)
                    logs.append(
                        {
                            "args": self.serialize_args(args),
                            "answer": None,
                            "correct_answer": correct_answer,
                            "passed": False,
                            "error": f"Error: {e}",
                            "html_prompt": "parse_error",
                        }
                    )

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
                "config": OmegaConf.to_object(cfg),
            }
            # Save JSON
            outpath = os.path.join(experiment_storage, f"{self.name}.json")
            with open(outpath, "w") as f:
                json.dump(test, f)

        def get_args_iterator(self):
            raise NotImplementedError

        def prompt(self, **kwargs):
            raise NotImplementedError

        def pass_condition(self, answer, **kwargs):
            raise NotImplementedError

        def serialize_args(self, args: dict):
            """Helper to JSON-serialize PersonaIdentity, etc."""
            res = {}
            for k, v in args.items():
                if isinstance(v, PersonaIdentity):
                    res[k] = v.agent_id
                elif isinstance(v, list) and all(
                    isinstance(f, PersonaIdentity) for f in v
                ):
                    res[k] = [f.agent_id for f in v]
                else:
                    res[k] = v
            return res

    # -------------------------------------------------------------------
    # Existing test classes from your snippet
    # -------------------------------------------------------------------
    class MathConsequenceAfterFishingSameAmount(TestCase):
        def __init__(self, name="math_consequence_after_fishing_same_amount") -> None:
            super().__init__(name)

        def get_args_iterator(self):
            return [
                {
                    "persona": PersonaIdentity("John", "John"),
                    "num_tonnes_lake": 100,
                    "num_tonnes_fisher": 10,
                }
            ] * NUM_RUNS

        def prompt(self, *, persona, num_tonnes_lake, num_tonnes_fisher):
            # uses prompt_simple_reflection_if_all_fisher_that_same_quantity
            return prompt_simple_reflection_if_all_fisher_that_same_quantity(
                wrapper,
                cot_prompt,
                persona,
                num_tonnes_lake,
                num_tonnes_fisher,
                cfg.llm.few_shots,
                cfg.llm.description_version,
            )

        def pass_condition(self, answer, persona, num_tonnes_lake, num_tonnes_fisher):
            correct_answer = max(
                0, min(100, (num_tonnes_lake - num_tonnes_fisher * 5) * 2)
            )
            return answer == correct_answer, correct_answer

    class SimConsequenceAfterFishingSameAmount(MathConsequenceAfterFishingSameAmount):
        def __init__(self, name="sim_consequence_after_fishing_same_amount") -> None:
            super().__init__(name)

        def prompt(self, *, persona, num_tonnes_lake, num_tonnes_fisher):
            return prompt_reflection_if_all_fisher_that_same_quantity(
                wrapper,
                cot_prompt,
                persona,
                num_tonnes_lake,
                num_tonnes_fisher,
                cfg.llm.few_shots,
                cfg.llm.description_version,
            )

    class MathShrinkingLimit(TestCase):
        def __init__(self, name="math_shrinking_limit") -> None:
            super().__init__(name)

        def get_args_iterator(self):
            return [
                {
                    "persona": PersonaIdentity("John", "John"),
                    "num_tonnes_lake": 100,
                }
            ] * NUM_RUNS

        def prompt(self, *, persona, num_tonnes_lake):
            return prompt_simple_shrinking_limit(
                wrapper,
                cot_prompt,
                persona,
                num_tonnes_lake,
                cfg.llm.few_shots,
                cfg.llm.description_version,
            )

        def pass_condition(self, answer, persona, num_tonnes_lake):
            correct_answer = (num_tonnes_lake // 2) // 5
            return answer == correct_answer, correct_answer

    class MathShrinkingLimitAssumption(TestCase):
        def __init__(self, name="math_shrinking_limit_assumption") -> None:
            super().__init__(name)

        def get_args_iterator(self):
            return [
                {
                    "persona": PersonaIdentity("John", "John"),
                    "num_tonnes_lake": 100,
                }
            ] * NUM_RUNS

        def prompt(self, *, persona, num_tonnes_lake):
            return prompt_simple_shrinking_limit_assumption(
                wrapper,
                cot_prompt,
                persona,
                num_tonnes_lake,
                cfg.llm.few_shots,
                cfg.llm.description_version,
            )

        def pass_condition(self, answer, persona, num_tonnes_lake):
            correct_answer = (num_tonnes_lake // 2) // 5
            return answer == correct_answer, correct_answer

    class SimShrinkingLimit(MathShrinkingLimit):
        def __init__(self, name="sim_shrinking_limit") -> None:
            super().__init__(name)

        def prompt(self, *, persona, num_tonnes_lake):
            return prompt_shrinking_limit(
                wrapper,
                cot_prompt,
                persona,
                num_tonnes_lake,
                cfg.llm.few_shots,
                cfg.llm.description_version,
            )

    class SimShrinkingLimitAssumption(MathShrinkingLimitAssumption):
        def __init__(self, name="sim_shrinking_limit_assumption") -> None:
            super().__init__(name)

        def prompt(self, *, persona, num_tonnes_lake):
            return prompt_shrinking_limit_asumption(
                wrapper,
                cot_prompt,
                persona,
                num_tonnes_lake,
                cfg.llm.few_shots,
                cfg.llm.description_version,
            )

    class SimCatchFishStandardPersona(TestCase):
        def __init__(self, name="sim_catch_fish_standard_persona") -> None:
            super().__init__(name)

        def get_args_iterator(self):
            return [
                {
                    "persona": PersonaIdentity("John", "John"),
                    "num_tonnes_lake": 100,
                }
            ] * NUM_RUNS

        def prompt(self, *, persona, num_tonnes_lake):
            return prompt_action_choose_amount_of_fish_to_catch(
                wrapper,
                cot_prompt,
                persona,
                num_tonnes_lake,
                cfg.llm.few_shots,
                cfg.llm.description_version,
            )

        def pass_condition(self, answer, persona, num_tonnes_lake):
            correct_answer = (num_tonnes_lake // 2) // 5
            if correct_answer == 0:
                return answer == correct_answer, correct_answer
            return answer <= correct_answer and answer > 0, correct_answer

    class SimUnivCatchFishStandardPersona(TestCase):
        def __init__(self, name="sim_catch_fish_universalization") -> None:
            super().__init__(name)

        def get_args_iterator(self):
            return [
                {
                    "persona": PersonaIdentity("John", "John"),
                    "num_tonnes_lake": 100,
                }
            ] * NUM_RUNS

        def prompt(self, *, persona, num_tonnes_lake):
            return prompt_action_choose_amount_of_fish_to_catch_universalization(
                wrapper,
                cot_prompt,
                persona,
                num_tonnes_lake,
                cfg.llm.few_shots,
                cfg.llm.description_version,
            )

        def pass_condition(self, answer, persona, num_tonnes_lake):
            correct_answer = (num_tonnes_lake // 2) // 5
            if correct_answer == 0:
                return answer == correct_answer, correct_answer
            return answer <= correct_answer and answer > 0, correct_answer

    # Some random persona for multiple tests:
    def get_random_persona():
        persona_names = ["John", "Kate", "Jack", "Emma", "Luke"]
        name = persona_names[np.random.randint(0, len(persona_names))]
        return PersonaIdentity(name, name)

    # Multiple versions of above tests:
    class MultipleMathShrinkingLimit(MathShrinkingLimit):
        def __init__(self, name="multiple_math_shrinking_limit") -> None:
            super().__init__(name)

        def get_args_iterator(self):
            return [
                {
                    "persona": get_random_persona(),
                    "num_tonnes_lake": int(i),
                }
                for i in np.random.randint(10, 101, NUM_RUNS)
            ]

    class MultipleSimShrinkingLimit(SimShrinkingLimit):
        def __init__(self, name="multiple_sim_shrinking_limit") -> None:
            super().__init__(name)

        def get_args_iterator(self):
            return [
                {
                    "persona": get_random_persona(),
                    "num_tonnes_lake": int(i),
                }
                for i in np.random.randint(10, 101, NUM_RUNS)
            ]

    class MultipleMathShrinkingLimitAssumption(MathShrinkingLimitAssumption):
        def __init__(self, name="multiple_math_shrinking_limit_assumption") -> None:
            super().__init__(name)

        def get_args_iterator(self):
            return [
                {
                    "persona": get_random_persona(),
                    "num_tonnes_lake": int(i),
                }
                for i in np.random.randint(10, 101, NUM_RUNS)
            ]

    class MultipleSimShrinkingLimitAssumption(SimShrinkingLimitAssumption):
        def __init__(self, name="multiple_sim_shrinking_limit_assumption") -> None:
            super().__init__(name)

        def get_args_iterator(self):
            return [
                {
                    "persona": get_random_persona(),
                    "num_tonnes_lake": int(i),
                }
                for i in np.random.randint(10, 101, NUM_RUNS)
            ]

    class MultipleSimCatchFishStandardPersona(SimCatchFishStandardPersona):
        def __init__(self, name="multiple_sim_catch_fish_standard_persona") -> None:
            super().__init__(name)

        def get_args_iterator(self):
            return [
                {
                    "persona": get_random_persona(),
                    "num_tonnes_lake": int(i),
                }
                for i in np.random.randint(10, 101, NUM_RUNS)
            ]

    class MultipleSimUniverCatchFishStandardPersona(SimUnivCatchFishStandardPersona):
        def __init__(self, name="multiple_sim_universalization_catch_fish") -> None:
            super().__init__(name)

        def get_args_iterator(self):
            return [
                {
                    "persona": get_random_persona(),
                    "num_tonnes_lake": int(i),
                }
                for i in np.random.randint(10, 101, NUM_RUNS)
            ]

    class MultipleMathConsequenceAfterFishingSameAmount(
        MathConsequenceAfterFishingSameAmount
    ):
        def __init__(
            self, name="multiple_math_consequence_after_fishing_same_amount"
        ) -> None:
            super().__init__(name)

        def get_args_iterator(self):
            return [
                {
                    "persona": get_random_persona(),
                    "num_tonnes_lake": int(i),
                    "num_tonnes_fisher": int(np.random.randint(0, (i // 5) + 1)),
                }
                for i in np.random.randint(10, 101, NUM_RUNS)
            ]

    class MultipleSimConsequenceAfterFishingSameAmount(
        SimConsequenceAfterFishingSameAmount
    ):
        def __init__(
            self, name="multiple_sim_consequence_after_fishing_same_amount"
        ) -> None:
            super().__init__(name)

        def get_args_iterator(self):
            return [
                {
                    "persona": get_random_persona(),
                    "num_tonnes_lake": int(i),
                    "num_tonnes_fisher": int(np.random.randint(0, (i // 5) + 1)),
                }
                for i in np.random.randint(10, 101, NUM_RUNS)
            ]

    # -------------------------------------------------------------------
    # NEW: Additional test cases with 1 leader + multiple followers
    # Each runs for 5 iterations. We embed that inside each class.
    # -------------------------------------------------------------------
    class BaseLeaderFollowerTestCase(TestCase):
        """
        Base test class for scenarios with 1 leader and multiple followers.
        The leader uses `prompt_leader_decision`, and the followers use
        a prompt method that we override in subclasses.
        """
        NUM_RUNS = 5  # "test for 5 for each"

        def __init__(self, name="base_leader_follower_test"):
            super().__init__(name)

        def get_args_iterator(self):
            # Return 5 identical scenarios for demonstration.
            return [
                {
                    "leader": PersonaIdentity("Emma", "Emma"),
                    "followers": [
                        PersonaIdentity("John", "John"),
                        PersonaIdentity("Kate", "Kate"),
                    ],
                    "num_tons_lake": 100,
                }
            ] * self.NUM_RUNS

        def prompt(
            self,
            *,
            leader: PersonaIdentity,
            followers: list[PersonaIdentity],
            num_tons_lake: int,
        ):
            # Leader's decision
            leader_catch, html_prompt_leader = prompt_leader_decision(
                wrapper,
                cot_prompt,
                leader,
                num_tons_lake,
                followers,
                cfg.llm.few_shots,
                cfg.llm.description_version,
            )

            # Followers' decisions
            follower_catches = []
            html_prompts_followers = []

            for follower in followers:
                catch, html_f = self.prompt_follower_decision(
                    follower, num_tons_lake
                )
                follower_catches.append(catch)
                html_prompts_followers.append(html_f)

            # Combine HTML logs
            all_html = html_prompt_leader + "\n" + "\n".join(html_prompts_followers)

            # We'll store final answer for pass_condition
            # as a tuple: (leader_catch, follower_catches)
            answer = (leader_catch, follower_catches)
            return answer, all_html

        def prompt_follower_decision(self, follower: PersonaIdentity, lake_size: int):
            """
            This is overridden in each subclass to use different prompts:
            e.g. prompt_action_choose_amount_of_fish_to_catch, etc.
            """
            raise NotImplementedError

        def pass_condition(self, answer, leader, followers, num_tons_lake):
            """
            Example: 
              - total catch <= half the lake
              - each person's catch is <= (lake//2)//(1+len(followers))
              - all > 0
            """
            leader_catch, follower_catches = answer
            total_fishers = 1 + len(followers)
            max_sustainable_per_person = (num_tons_lake // 2) // total_fishers

            # Check leader
            if leader_catch < 1 or leader_catch > max_sustainable_per_person:
                return (False, max_sustainable_per_person)

            # Check followers
            for c in follower_catches:
                if c < 1 or c > max_sustainable_per_person:
                    return (False, max_sustainable_per_person)

            total_catch = leader_catch + sum(follower_catches)
            if total_catch > (num_tons_lake // 2):
                return (False, max_sustainable_per_person)

            return (True, max_sustainable_per_person)

    # Subclass for each type of follower prompt
    class LeaderFollower_ChooseAmount(BaseLeaderFollowerTestCase):
        def __init__(self, name="leader_follower_choose_amount"):
            super().__init__(name)

        def prompt_follower_decision(self, follower: PersonaIdentity, lake_size: int):
            return prompt_action_choose_amount_of_fish_to_catch(
                wrapper,
                cot_prompt,
                follower,
                lake_size,
                cfg.llm.few_shots,
                cfg.llm.description_version,
            )

    class LeaderFollower_ChooseAmountUniv(BaseLeaderFollowerTestCase):
        def __init__(self, name="leader_follower_choose_amount_univ"):
            super().__init__(name)

        def prompt_follower_decision(self, follower: PersonaIdentity, lake_size: int):
            return prompt_action_choose_amount_of_fish_to_catch_universalization(
                wrapper,
                cot_prompt,
                follower,
                lake_size,
                cfg.llm.few_shots,
                cfg.llm.description_version,
            )

    class LeaderFollower_ReflectionAfterSameAmount(BaseLeaderFollowerTestCase):
        def __init__(self, name="leader_follower_reflection_after_same_amount"):
            super().__init__(name)

        def prompt_follower_decision(self, follower: PersonaIdentity, lake_size: int):
            # For demo, assume "num_tonnes_fisher" = 5
            num_tonnes_fisher = 5
            return prompt_reflection_if_all_fisher_that_same_quantity(
                wrapper,
                cot_prompt,
                follower,
                lake_size,
                num_tonnes_fisher,
                cfg.llm.few_shots,
                cfg.llm.description_version,
            )

    class LeaderFollower_ShrinkingLimit(BaseLeaderFollowerTestCase):
        def __init__(self, name="leader_follower_shrinking_limit"):
            super().__init__(name)

        def prompt_follower_decision(self, follower: PersonaIdentity, lake_size: int):
            return prompt_shrinking_limit(
                wrapper,
                cot_prompt,
                follower,
                lake_size,
                cfg.llm.few_shots,
                cfg.llm.description_version,
            )

    class LeaderFollower_ShrinkingLimitAssumption(BaseLeaderFollowerTestCase):
        def __init__(self, name="leader_follower_shrinking_limit_assumption"):
            super().__init__(name)

        def prompt_follower_decision(self, follower: PersonaIdentity, lake_size: int):
            return prompt_shrinking_limit_asumption(
                wrapper,
                cot_prompt,
                follower,
                lake_size,
                cfg.llm.few_shots,
                cfg.llm.description_version,
            )

    class LeaderFollower_SimpleReflectionAfterSameAmount(BaseLeaderFollowerTestCase):
        def __init__(self, name="leader_follower_simple_reflection"):
            super().__init__(name)

        def prompt_follower_decision(self, follower: PersonaIdentity, lake_size: int):
            # For demo, assume "num_tonnes_fisher" = 5
            num_tonnes_fisher = 5
            return prompt_simple_reflection_if_all_fisher_that_same_quantity(
                wrapper,
                cot_prompt,
                follower,
                lake_size,
                num_tonnes_fisher,
                cfg.llm.few_shots,
                cfg.llm.description_version,
            )

    class LeaderFollower_SimpleShrinkingLimit(BaseLeaderFollowerTestCase):
        def __init__(self, name="leader_follower_simple_shrinking_limit"):
            super().__init__(name)

        def prompt_follower_decision(self, follower: PersonaIdentity, lake_size: int):
            return prompt_simple_shrinking_limit(
                wrapper,
                cot_prompt,
                follower,
                lake_size,
                cfg.llm.few_shots,
                cfg.llm.description_version,
            )

    class LeaderFollower_SimpleShrinkingLimitAssumption(BaseLeaderFollowerTestCase):
        def __init__(self, name="leader_follower_simple_shrinking_limit_assumption"):
            super().__init__(name)

        def prompt_follower_decision(self, follower: PersonaIdentity, lake_size: int):
            return prompt_simple_shrinking_limit_assumption(
                wrapper,
                cot_prompt,
                follower,
                lake_size,
                cfg.llm.few_shots,
                cfg.llm.description_version,
            )

    # -------------------------------------------------------------------
    # Now define all the test case objects and run them.
    # Split logic: you can group them in whichever "split" you want
    # or just run them all.
    # -------------------------------------------------------------------
    test_cases_2 = [
        MultipleMathShrinkingLimit(),
        MultipleSimShrinkingLimit(),
        MultipleMathConsequenceAfterFishingSameAmount(),
        MultipleSimConsequenceAfterFishingSameAmount(),
        MultipleSimCatchFishStandardPersona(),
        MultipleSimUniverCatchFishStandardPersona(),
        MultipleMathShrinkingLimitAssumption(),
        MultipleSimShrinkingLimitAssumption(),
    ]

    # The new leader-follower tests (each runs 5 times)
    test_cases_leader_follower = [
        LeaderFollower_ChooseAmount(),
        LeaderFollower_ChooseAmountUniv(),
        LeaderFollower_ReflectionAfterSameAmount(),
        LeaderFollower_ShrinkingLimit(),
        LeaderFollower_ShrinkingLimitAssumption(),
        LeaderFollower_SimpleReflectionAfterSameAmount(),
        LeaderFollower_SimpleShrinkingLimit(),
        LeaderFollower_SimpleShrinkingLimitAssumption(),
    ]

    if cfg.split == "single":
        test_cases = test_cases_2 + test_cases_leader_follower
    elif int(cfg.split) == 1:
        test_cases = test_cases_2[:2]
    elif int(cfg.split) == 2:
        test_cases = test_cases_2[2:4]
    elif int(cfg.split) == 3:
        test_cases = test_cases_2[4:6]
    elif int(cfg.split) == 4:
        test_cases = test_cases_2[6:]
    elif int(cfg.split) == 5:
        test_cases = test_cases_leader_follower
    else:
        # By default, run them all
        test_cases = test_cases_2 + test_cases_leader_follower

    for test_case in tqdm.tqdm(test_cases):
        test_case.run()


if __name__ == "__main__":
    # Register an OmegaConf resolver for unique runs
    OmegaConf.register_resolver("uuid", lambda: f"run_{uuid.uuid4()}")
    main()
