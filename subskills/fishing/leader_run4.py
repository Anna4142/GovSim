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
    prompt_follower_decision,
    prompt_leader_autocratic_decision,
    prompt_leader_democratic_decision,
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
    NUM_RUNS = 50
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

     ##BASE LEADER FOLLOER   
    class BaseLeaderFollowerTest(TestCase):
        """Base class for all leader-follower test scenarios"""
        def __init__(self, name):
            super().__init__(name)
            self.current_lake = 100
            self.max_capacity = 100
            self.num_runs = 5
            self._setup_personas()
            self._setup_prompts()

        def _setup_personas(self):
            """Set up leader and follower personas"""
            self.leader = PersonaIdentity("Leader", "Leader")
            self.followers = [
                PersonaIdentity("Follower1", "Follower1"),
                PersonaIdentity("Follower2", "Follower2")
            ]

        def _setup_prompts(self):
            """Set up which prompts to use - override in derived classes"""
            self.leadership_style = None  # "democratic" or "autocratic"
            self.leader_decision_prompt = None
            self.follower_prompt_func = None

        def get_args_iterator(self):
            """Base implementation returning simple args for each run"""
            return [{"run": i} for i in range(self.num_runs)]

        def prompt(self, **kwargs):
            """Execute fishing decisions using configured prompts"""
            try:
                html_prompts = []

                # 1) Leader's decision combining leadership style and decision method
                if self.leadership_style == "democratic":
                    leader_catch, html_leader = prompt_leader_democratic_decision(
                        wrapper,
                        cot_prompt,
                        self.leader,
                        self.current_lake,
                        self.followers,
                        cfg.llm.few_shots,
                        cfg.llm.description_version,
                    )
                    # After democratic leadership decision, apply decision method
                    leader_catch, html_decision = self.leader_decision_prompt(
                        wrapper,
                        cot_prompt,
                        self.leader,
                        self.current_lake,
                        cfg.llm.few_shots,
                        cfg.llm.description_version,
                    )
                else:  # autocratic
                    leader_catch, html_leader = prompt_leader_autocratic_decision(
                        wrapper,
                        cot_prompt,
                        self.leader,
                        self.current_lake,
                        self.followers,
                        cfg.llm.few_shots,
                        cfg.llm.description_version,
                    )
                    # After autocratic leadership decision, apply decision method
                    leader_catch, html_decision = self.leader_decision_prompt(
                        wrapper,
                        cot_prompt,
                        self.leader,
                        self.current_lake,
                        cfg.llm.few_shots,
                        cfg.llm.description_version,
                    )

                html_prompts.extend([str(h) for h in [html_leader, html_decision] if h is not None])

                # 2) Followers' decisions
                follower_catches = []
                for f in self.followers:
                    catch, html_f = self.follower_prompt_func(
                        wrapper,
                        cot_prompt,
                        f,
                        self.leader,
                        leader_catch,
                        self.current_lake,
                        self.leadership_style,
                        cfg.llm.few_shots,
                        cfg.llm.description_version,
                    )
                    follower_catches.append(catch)
                    html_prompts.append(str(html_f) if html_f is not None else "")

                # 3) Update lake population
                total_catch = leader_catch + sum(follower_catches)
                remainder = max(0, self.current_lake - total_catch)
                self.current_lake = remainder

                # Combine HTML safely
                all_html = "\n".join(filter(None, html_prompts))

                return (leader_catch, follower_catches, remainder), all_html

            except Exception as e:
                print(f"Error in prompt execution: {e}")
                return (0, [0] * len(self.followers), self.current_lake), "Error in prompt execution"

        def pass_condition(self, answer, **kwargs):
            """Default evaluation criteria"""
            leader_catch, follower_catches, new_lake = answer

            if new_lake <= 0:
                return (False, {"reason": "Lake depleted!"})

            old_lake_size = kwargs.get("old_lake_size", 100)
            correct_answer = (old_lake_size // 2) // 5

            def is_catch_ok(catch_val):
                if correct_answer == 0:
                    return catch_val == 0
                return 0 < catch_val <= correct_answer

            leader_ok = is_catch_ok(leader_catch)
            followers_ok_list = [is_catch_ok(fc) for fc in follower_catches]
            
            passed = leader_ok and all(followers_ok_list)
            
            details = {
                "correct_answer": correct_answer,
                "leader_ok": leader_ok,
                "followers_ok_list": followers_ok_list,
                "leadership_style": self.leadership_style,
                "prompt_combination": self.get_prompt_combination_name()
            }
            return passed, details

        def get_prompt_combination_name(self):
            """Get descriptive name of the prompt combination"""
            return f"{self.leadership_style}_base"

        def run(self):
            """Execute test case with enhanced logging"""
            logs = []
            for idx, args in enumerate(self.get_args_iterator()):
                if self.current_lake <= 0:
                    print(f"Lake depleted at iteration {idx}. Ending scenario.")
                    break

                old_lake_size = self.current_lake

                try:
                    answer, html_prompt = self.prompt(**args)
                    passed, pc_details = self.pass_condition(
                        answer, old_lake_size=old_lake_size, **args
                    )

                    logs.append({
                        "day": idx + 1,
                        "args": self.serialize_args(args),
                        "answer": answer,
                        "passed": passed,
                        "correct_answer": pc_details.get("correct_answer"),
                        "leader_set_right_example": pc_details.get("leader_ok"),
                        "followers_obeyed": all(pc_details.get("followers_ok_list", [])),
                        "followers_ok_list": pc_details.get("followers_ok_list", []),
                        "error": "OK",
                        "html_prompt": html_prompt,
                        "remaining_lake_after_run": self.current_lake,
                        "leadership_style": self.leadership_style,
                        "prompt_combination": self.get_prompt_combination_name()
                    })

                except Exception as e:
                    print(f"Error in run {idx}: {e}")
                    fallback_answer = (0, [0] * len(self.followers), self.current_lake)
                    passed, pc_details = self.pass_condition(
                        fallback_answer, old_lake_size=old_lake_size, **args
                    )

                    logs.append({
                        "day": idx + 1,
                        "args": self.serialize_args(args),
                        "answer": None,
                        "passed": False,
                        "correct_answer": pc_details.get("correct_answer"),
                        "leader_set_right_example": False,
                        "followers_obeyed": False,
                        "followers_ok_list": [],
                        "error": str(e),
                        "html_prompt": "Error in test execution",
                        "remaining_lake_after_run": self.current_lake,
                        "leadership_style": self.leadership_style,
                        "prompt_combination": self.get_prompt_combination_name()
                    })

            # Calculate statistics and save results
            if len(logs) > 0:
                ci = smprop.proportion_confint(
                    sum(log["passed"] for log in logs), len(logs), alpha=0.05
                )
                score_mean = float(np.mean([log["passed"] for log in logs]))
                score_std = float(np.std([log["passed"] for log in logs]))
                score_ci_lower, score_ci_upper = ci
            else:
                score_mean = score_std = score_ci_lower = score_ci_upper = 0.0

            test = {
                "name": self.name,
                "instances": logs,
                "score_mean": score_mean,
                "score_std": score_std,
                "score_ci_lower": score_ci_lower,
                "score_ci_upper": score_ci_upper,
                "config": OmegaConf.to_object(cfg)
            }

            outpath = os.path.join(experiment_storage, f"{self.name}.json")
            with open(outpath, "w") as f:
                json.dump(test, f, indent=2)
            print(f"Wrote results to {outpath}")



        # Democratic Leadership Combinations
    class DemocraticBasicChoice(BaseLeaderFollowerTest):
            """Democratic leader with basic choice for decisions"""
            def __init__(self, name="democratic_basic_choice"):
                super().__init__(name)

            def _setup_prompts(self):
                self.leadership_style = "democratic"
                self.leader_decision_prompt = prompt_action_choose_amount_of_fish_to_catch
                self.follower_prompt_func = prompt_action_choose_amount_of_fish_to_catch

            def get_prompt_combination_name(self):
                return "democratic_basic_choice_both"


    class DemocraticUniversalChoice(BaseLeaderFollowerTest):
            """Democratic leader with universal choice for decisions"""
            def __init__(self, name="democratic_universal_choice"):
                super().__init__(name)

            def _setup_prompts(self):
                self.leadership_style = "democratic"
                self.leader_decision_prompt = prompt_action_choose_amount_of_fish_to_catch_universalization
                self.follower_prompt_func = prompt_action_choose_amount_of_fish_to_catch_universalization

            def get_prompt_combination_name(self):
                return "democratic_universal_choice_both"


        # Autocratic Leadership Combinations
    class AutocraticBasicChoice(BaseLeaderFollowerTest):
            """Autocratic leader with basic choice for decisions"""
            def __init__(self, name="autocratic_basic_choice"):
                super().__init__(name)

            def _setup_prompts(self):
                self.leadership_style = "autocratic"
                self.leader_decision_prompt = prompt_action_choose_amount_of_fish_to_catch
                self.follower_prompt_func = prompt_action_choose_amount_of_fish_to_catch

            def get_prompt_combination_name(self):
                return "autocratic_basic_choice_both"


    class AutocraticUniversalChoice(BaseLeaderFollowerTest):
            """Autocratic leader with universal choice for decisions"""
            def __init__(self, name="autocratic_universal_choice"):
                super().__init__(name)

            def _setup_prompts(self):
                self.leadership_style = "autocratic"
                self.leader_decision_prompt = prompt_action_choose_amount_of_fish_to_catch_universalization
                self.follower_prompt_func = prompt_action_choose_amount_of_fish_to_catch_universalization

            def get_prompt_combination_name(self):
                return "autocratic_universal_choice_both"


        # Mixed Leadership-Decision Combinations
    class DemocraticMixedChoice(BaseLeaderFollowerTest):
            """Democratic leader using universal choice, followers using basic choice"""
            def __init__(self, name="democratic_mixed_choice"):
                super().__init__(name)

            def _setup_prompts(self):
                self.leadership_style = "democratic"
                self.leader_decision_prompt = prompt_action_choose_amount_of_fish_to_catch_universalization
                self.follower_prompt_func = prompt_action_choose_amount_of_fish_to_catch

            def get_prompt_combination_name(self):
                return "democratic_universal_leader_basic_followers"


    class AutocraticMixedChoice(BaseLeaderFollowerTest):
            """Autocratic leader using basic choice, followers using universal choice"""
            def __init__(self, name="autocratic_mixed_choice"):
                super().__init__(name)

            def _setup_prompts(self):
                self.leadership_style = "autocratic"
                self.leader_decision_prompt = prompt_action_choose_amount_of_fish_to_catch
                self.follower_prompt_func = prompt_action_choose_amount_of_fish_to_catch_universalization

            def get_prompt_combination_name(self):
                return "autocratic_basic_leader_universal_followers"


        # Update test cases to include all combinations
    test_cases = [
            # Democratic combinations
            DemocraticBasicChoice(),
            DemocraticUniversalChoice(),
            DemocraticMixedChoice(),
            
            # Autocratic combinations
            AutocraticBasicChoice(),
            AutocraticUniversalChoice(),
            AutocraticMixedChoice()
        ]

        # Run tests
    for test_case in tqdm.tqdm(test_cases):
            test_case.run()
if __name__ == "__main__":
    OmegaConf.register_resolver("uuid", lambda: f"run_{uuid.uuid4()}")
    main()
