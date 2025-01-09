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
# "reasoning_free_format" imports (must exist in your codebase)
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
    prompt_leader_decision,  # Ensure this is present in reasoning_free_format
)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    # Initialize model + logger
    model = get_model(cfg.llm.path, cfg.llm.is_api, cfg.seed, cfg.llm.backend)
    logger = WandbLogger(
        f"subskills_check/fishing/{cfg.code_version}",
        OmegaConf.to_object(cfg),
        debug=cfg.debug,
    )

    # Set up results directory
    experiment_storage = os.path.join(
        os.path.dirname(__file__),
        f"./results/subskills_check_{cfg.code_version}/{logger.run_name}",
    )
    os.makedirs(experiment_storage, exist_ok=True)

    # Wrap model for logging/temperature/etc.
    wrapper = ModelWandbWrapper(
        model,
        render=cfg.llm.render,
        wanbd_logger=logger,
        temperature=cfg.llm.temperature,
        top_p=cfg.llm.top_p,
        seed=cfg.seed,
        is_api=cfg.llm.is_api,
    )

    # Check out_format
    if cfg.llm.out_format == "freeform":
        pass  # We already have the imports above
    else:
        raise ValueError(f"Unknown out_format: {cfg.llm.out_format}")

    # Handle CoT prompt
    if cfg.llm.cot_prompt == "deep_breath":
        cot_prompt = "Take a deep breath and work on this problem step-by-step."
    elif cfg.llm.cot_prompt == "think_step_by_step":
        cot_prompt = "Let's think step-by-step."
    else:
        raise ValueError(f"Unknown cot_prompt: {cfg.llm.cot_prompt}")

    # By default, we'll do 5 runs for each test
    NUM_RUNS = 5
    if cfg.debug:
        NUM_RUNS = 2

    # -------------------------------------------------------------------
    # Base TestCase
    # -------------------------------------------------------------------
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
                sum(log["passed"] for log in logs), len(logs), alpha=ALPHA
            )

            test = {
                "name": self.name,
                "instances": logs,
                "score_mean": float(np.mean([log["passed"] for log in logs])),
                "score_std": float(np.std([log["passed"] for log in logs])),
                "score_ci_lower": ci[0],
                "score_ci_upper": ci[1],
                "config": OmegaConf.to_object(cfg),
            }

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
            res = {}
            for k, v in args.items():
                if isinstance(v, PersonaIdentity):
                    res[k] = v.agent_id
                elif isinstance(v, list) and all(isinstance(f, PersonaIdentity) for f in v):
                    res[k] = [f.agent_id for f in v]
                else:
                    res[k] = v
            return res

    # -------------------------------------------------------------------
    # Existing test classes (unchanged)
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
            correct_answer = max(0, min(100, (num_tonnes_lake - num_tonnes_fisher * 5) * 2))
            return (answer == correct_answer, correct_answer)

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
            return (answer == correct_answer, correct_answer)

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
            return (answer == correct_answer, correct_answer)

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
                return (answer == correct_answer, correct_answer)
            return (answer <= correct_answer and answer > 0, correct_answer)

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
                return (answer == correct_answer, correct_answer)
            return (answer <= correct_answer and answer > 0, correct_answer)

    # Some random persona for "multiple" tests:
    def get_random_persona():
        persona_names = ["John", "Kate", "Jack", "Emma", "Luke"]
        name = persona_names[np.random.randint(0, len(persona_names))]
        return PersonaIdentity(name, name)

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
    # Leader-Follower Base (Static, Single-run lake = 100)
    # -------------------------------------------------------------------
    class BaseLeaderFollowerTestCase(TestCase):
        """
        1 leader + multiple followers, each run resets the lake to the same value.
        """
        NUM_RUNS = 5

        def __init__(self, name="base_leader_follower_test"):
            super().__init__(name)

        def get_args_iterator(self):
            return [
                {
                    "leader": PersonaIdentity("Emma", "Emma"),
                    "followers": [PersonaIdentity("John", "John"), PersonaIdentity("Kate", "Kate")],
                    "num_tons_lake": 100,
                }
            ] * self.NUM_RUNS

        def prompt(self, *, leader, followers, num_tons_lake):
            leader_catch, html_leader = prompt_leader_decision(
                wrapper,
                cot_prompt,
                leader,
                num_tons_lake,
                followers,
                cfg.llm.few_shots,
                cfg.llm.description_version,
            )
            follower_catches = []
            html_followers = []
            for f in followers:
                catch, html_f = self.prompt_follower_decision(f, num_tons_lake)
                follower_catches.append(catch)
                html_followers.append(html_f)

            all_html = html_leader + "\n" + "\n".join(html_followers)
            answer = (leader_catch, follower_catches)
            return answer, all_html

        def prompt_follower_decision(self, follower, lake_size):
            raise NotImplementedError

        def pass_condition(self, answer, leader, followers, num_tons_lake):
            leader_catch, follower_catches = answer
            total_fishers = 1 + len(followers)
            max_sustain = (num_tons_lake // 2) // total_fishers

            if leader_catch < 1 or leader_catch > max_sustain:
                return (False, max_sustain)
            for c in follower_catches:
                if c < 1 or c > max_sustain:
                    return (False, max_sustain)

            total_catch = leader_catch + sum(follower_catches)
            if total_catch > (num_tons_lake // 2):
                return (False, max_sustain)

            return (True, max_sustain)

    class LeaderFollower_ChooseAmount(BaseLeaderFollowerTestCase):
        def __init__(self, name="leader_follower_choose_amount"):
            super().__init__(name)

        def prompt_follower_decision(self, follower, lake_size):
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

        def prompt_follower_decision(self, follower, lake_size):
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

        def prompt_follower_decision(self, follower, lake_size):
            # Hard-coded example:
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

        def prompt_follower_decision(self, follower, lake_size):
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

        def prompt_follower_decision(self, follower, lake_size):
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

        def prompt_follower_decision(self, follower, lake_size):
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

        def prompt_follower_decision(self, follower, lake_size):
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

        def prompt_follower_decision(self, follower, lake_size):
            return prompt_simple_shrinking_limit_assumption(
                wrapper,
                cot_prompt,
                follower,
                lake_size,
                cfg.llm.few_shots,
                cfg.llm.description_version,
            )

    # -------------------------------------------------------------------
    # NEW: A stateful test class where the lake updates each instance.
    # If fish deplete (<=0), we end early. The fish also doubles after each step.
    # -------------------------------------------------------------------
    class LeaderFollower_ShrinkingLakeStateful(TestCase):
        """
        Similar to the BaseLeaderFollowerTestCase, but we keep track of the
        lake across runs. Each run uses the updated fish count from the previous run.
        If it hits 0 or below, we stop the scenario.
        """

        def __init__(self, name="leader_follower_shrinking_lake_stateful"):
            super().__init__(name)
            self.current_lake = 100
            self.max_capacity = 100
            self.num_runs = 5
            self.leader = PersonaIdentity("Emma", "Emma")
            self.followers = [
                PersonaIdentity("John", "John"),
                PersonaIdentity("Kate", "Kate"),
            ]

        def get_args_iterator(self):
            """
            We'll create a list of dummy arguments for how many
            runs we plan (self.num_runs). The real "state" is in self.current_lake.
            """
            return [{} for _ in range(self.num_runs)]

        def prompt(self, **kwargs):
            """
            1) Leader's decision with current_lake
            2) Each follower's decision
            3) Remove fish
            4) Double what's left (capped at 100)
            """
            # 1) Leader
            leader_catch, html_leader = prompt_leader_decision(
                wrapper,
                cot_prompt,
                self.leader,
                self.current_lake,
                self.followers,
                cfg.llm.few_shots,
                cfg.llm.description_version,
            )

            # 2) Followers
            follower_catches = []
            html_followers = []
            for f in self.followers:
                catch, html_f = prompt_action_choose_amount_of_fish_to_catch(
                    wrapper,
                    cot_prompt,
                    f,
                    self.current_lake,
                    cfg.llm.few_shots,
                    cfg.llm.description_version,
                )
                follower_catches.append(catch)
                html_followers.append(html_f)

            # 3) Remove fish from lake
            total_catch = leader_catch + sum(follower_catches)
            remainder = self.current_lake - total_catch
            if remainder < 0:
                remainder = 0

            # 4) Double remainder, capped
            new_lake = min(remainder * 2, self.max_capacity)

            # update state for next iteration
            self.current_lake = new_lake

            # combine HTML
            all_html = html_leader + "\n" + "\n".join(html_followers)

            # Return the final answer: (leader_catch, follower_catches, new_lake)
            return (leader_catch, follower_catches, new_lake), all_html

        def pass_condition(self, answer, **kwargs):
            """
            For this example, let's say the test 'passes' if the
            leader + followers remain in a 'sustainable' range. Also,
            if the lake hits 0, we consider it a fail or 'episode end'.
            """
            leader_catch, follower_catches, new_lake = answer

            # If the lake is depleted at any stage, we fail
            if new_lake <= 0:
                return (False, "Lake depleted!")

            # e.g. We require each to catch <= 16
            # and total <= 50, as a simple criterion
            max_sustain = 16
            if leader_catch < 0 or leader_catch > max_sustain:
                return (False, max_sustain)
            for c in follower_catches:
                if c < 0 or c > max_sustain:
                    return (False, max_sustain)

            total_catch = leader_catch + sum(follower_catches)
            if total_catch > 50:
                return (False, max_sustain)

            # Otherwise pass
            return (True, max_sustain)

        def run(self):
            """
            Override run to stop early if fish is depleted.
            """
            logs = []
            for idx, args in enumerate(self.get_args_iterator()):
                # If lake is already 0 or negative, break
                if self.current_lake <= 0:
                    print(f"Lake is depleted at iteration {idx}. Ending.")
                    break

                try:
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
                    # If we consider failing once -> end the entire episode, you can do:
                    # if not passed:
                    #     print("Failed, ending episode now.")
                    #     break
                    # But if you want to keep going, omit it.
                except Exception as e:
                    print(f"Error: {e}")
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

            # Summaries
            ALPHA = 0.05
            ci = smprop.proportion_confint(
                sum(log["passed"] for log in logs), len(logs), alpha=ALPHA
            )
            test = {
                "name": self.name,
                "instances": logs,
                "score_mean": float(np.mean([log["passed"] for log in logs])),
                "score_std": float(np.std([log["passed"] for log in logs])),
                "score_ci_lower": ci[0],
                "score_ci_upper": ci[1],
                "config": OmegaConf.to_object(cfg),
            }

            outpath = os.path.join(experiment_storage, f"{self.name}.json")
            with open(outpath, "w") as f:
                json.dump(test, f)

    # -------------------------------------------------------------------
    # Assemble test cases, including the new "LeaderFollower_ShrinkingLakeStateful"
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

    # NEW stateful lake scenario
    stateful_test = [LeaderFollower_ShrinkingLakeStateful()]

    # Decide which to run (depending on cfg.split, or just run them all)
    if cfg.split == "single":
        test_cases = test_cases_2 + test_cases_leader_follower + stateful_test
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
    elif int(cfg.split) == 6:
        # Example: run only the new stateful scenario
        test_cases = stateful_test
    else:
        # By default, run everything
        test_cases = test_cases_2 + test_cases_leader_follower + stateful_test

    # Run
    for test_case in tqdm.tqdm(test_cases):
        test_case.run()


if __name__ == "__main__":
    OmegaConf.register_resolver("uuid", lambda: f"run_{uuid.uuid4()}")
    main()
