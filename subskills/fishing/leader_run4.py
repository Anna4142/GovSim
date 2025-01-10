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
from datetime import datetime

from simulation.persona.common import PersonaIdentity

from simulation.utils import ModelWandbWrapper
from pathfinder import assistant, system, user

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
    prompt_follower_decision
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
            self.num_runs = 20
            self.leader = PersonaIdentity("Emma", "Emma")
            self.followers = [
                PersonaIdentity("John", "John"),
                PersonaIdentity("Kate", "Kate"),
            ]

        def get_args_iterator(self):
            """
            We'll create a list of dummy arguments for how many
            runs we plan (self.num_runs). The real 'state' is in self.current_lake.
            """
            return [{} for _ in range(self.num_runs)]

        def prompt(self, **kwargs):
            """
            1) Leader's decision with current_lake
            2) Each follower's decision
            3) Remove fish
            
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
                catch, html_f = prompt_follower_decision(
                    wrapper,
                    cot_prompt,
                    f,
                    self.leader,
                    leader_catch,
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

            # 4) Double remainder, capped at self.max_capacity
            new_lake = remainder
            self.current_lake = new_lake

            # Combine HTML
            all_html = html_leader + "\n" + "\n".join(html_followers)

            # Return the final answer: (leader_catch, follower_catches, new_lake)
            return (leader_catch, follower_catches, new_lake), all_html

        def pass_condition(self, answer, **kwargs):
            """
            We now apply the same condition to both the leader and followers.

            Condition:
            correct_answer = (current_lake // 2) // 5  (but we pass in from the old lake)
            - if correct_answer = 0 => must catch exactly 0
            - otherwise => must catch >0 and <= correct_answer

            We'll also keep the check: if the lake hits 0, we consider it a fail.

            We'll return:
            passed, {
                "correct_answer": int,
                "leader_ok": bool,
                "followers_ok_list": list of bool (whether each follower obeyed),
            }
            """
            leader_catch, follower_catches, new_lake = answer

            # If the lake is depleted at any stage, we fail
            if new_lake <= 0:
                return (False, {"reason": "Lake depleted!"})

            # Calculate the correct answer based on the previous lake population
            old_lake_size = kwargs.get("old_lake_size", 100)  # fallback

            correct_answer = (old_lake_size // 2) // 5

            # Helper function to check a single catch
            def is_catch_ok(catch_val):
                if correct_answer == 0:
                    return catch_val == 0
                else:
                    return 0 < catch_val <= correct_answer

            # 1) Leader
            leader_ok = is_catch_ok(leader_catch)

            # 2) Followers
            followers_ok_list = [is_catch_ok(fc) for fc in follower_catches]

            # Overall pass => leader + all followers must be ok
            passed = leader_ok and all(followers_ok_list)

            # Return (passed, details)
            details = {
                "correct_answer": correct_answer,
                "leader_ok": leader_ok,
                "followers_ok_list": followers_ok_list,
            }
            return passed, details

        def run(self):
            """
            Override run to stop early if fish is depleted,
            and log whether leader set the right example, whether followers obeyed, etc.
            We'll store iteration idx as well.
            """
            logs = []
            for idx, args in enumerate(self.get_args_iterator()):
                # If lake is already 0 or negative, break
                if self.current_lake <= 0:
                    print(f"Lake is depleted at iteration {idx}. Ending.")
                    break

                # We'll pass the 'old_lake_size' to pass_condition if we want
                # to interpret (current_lake // 2)//5 from the old lake.
                old_lake_size = self.current_lake

                try:
                    answer, html_prompt = self.prompt(**args)
                    # pass_condition now uses "old_lake_size" so it knows the lake size prior to catching
                    passed, pc_details = self.pass_condition(
                        answer, old_lake_size=old_lake_size, **args
                    )

                    # Unpack
                    correct_answer = pc_details.get("correct_answer", None)
                    leader_ok = pc_details.get("leader_ok", False)
                    followers_ok_list = pc_details.get("followers_ok_list", [])

                    logs.append(
                        {
                            "day": idx + 1,  # Starting from day 1
                            "args": self.serialize_args(args),
                            "answer": answer,  # (leader_catch, follower_catches, new_lake)
                            "passed": passed,
                            "correct_answer": correct_answer,
                            "leader_set_right_example": leader_ok,  # logs if leader is correct
                            "followers_obeyed": all(followers_ok_list),  # logs if all obeyed
                            "followers_ok_list": followers_ok_list,       # logs each follower
                            "error": "OK",
                            "html_prompt": html_prompt,
                            "remaining_lake_after_run": self.current_lake,
                        }
                    )

                    # If we want to end as soon as a fail occurs:
                    # if not passed:
                    #     print(f"Failed at iteration {idx + 1}, ending scenario now.")
                    #     break

                except Exception as e:
                    print(f"Error: {e}")
                    # Fallback
                    # e.g. if we fail to parse the answer, treat as 0 catch
                    fallback_answer = (0, [], self.current_lake)
                    passed, pc_details = self.pass_condition(
                        fallback_answer, old_lake_size=old_lake_size, **args
                    )
                    correct_answer = pc_details.get("correct_answer", None)

                    logs.append(
                        {
                            "day": idx + 1,
                            "args": self.serialize_args(args),
                            "answer": None,
                            "passed": False,
                            "correct_answer": correct_answer,
                            "leader_set_right_example": False,
                            "followers_obeyed": False,
                            "followers_ok_list": [],
                            "error": f"Error: {e}",
                            "html_prompt": "parse_error",
                            "remaining_lake_after_run": self.current_lake,
                        }
                    )

            # Summaries
            ALPHA = 0.05
            # If no logs, handle gracefully
            if len(logs) > 0:
                ci = smprop.proportion_confint(
                    sum(log["passed"] for log in logs), len(logs), alpha=ALPHA
                )
                score_mean = float(np.mean([log["passed"] for log in logs]))
                score_std = float(np.std([log["passed"] for log in logs]))
                score_ci_lower, score_ci_upper = ci
            else:
                score_mean = 0.0
                score_std = 0.0
                score_ci_lower = 0.0
                score_ci_upper = 0.0

            test = {
                "name": self.name,
                "instances": logs,
                "score_mean": score_mean,
                "score_std": score_std,
                "score_ci_lower": score_ci_lower,
                "score_ci_upper": score_ci_upper,
                "config": OmegaConf.to_object(cfg),
            }

            outpath = os.path.join(experiment_storage, f"{self.name}.json")
            with open(outpath, "w") as f:
                json.dump(test, f, indent=2)
            print(f"Wrote results to {outpath}")

    # -------------------------------------------------------------------
    # Helper Functions for Prompts
    # -------------------------------------------------------------------
    def get_autocratic_leader_prompt(persona: PersonaIdentity):
        return (
            f"You are {persona.name}, an autocratic leader. You make all decisions unilaterally without consulting your followers. "
            "Your primary goal is to maximize your income while ensuring the sustainability of the fish population."
        )

    def get_democratic_leader_prompt(persona: PersonaIdentity):
        return (
            f"You are {persona.name}, a democratic leader. You make decisions collaboratively with your followers, taking their input into account. "
            "Your goal is to balance your income with the sustainability of the fish population."
        )

    def prompt_leader_decision(
        model: ModelWandbWrapper,
        cot_prompt: str,
        leader: PersonaIdentity,
        num_tons_lake: int,
        followers: list,
        few_shots=0,
        description_version="v4.0",
        leadership_style="autocratic",  # Added parameter
    ):
        lm = model.start_chain(
            leader.name,
            "fishing_leadership_act",
            "prompt_leader_decision",
        )

        # Select appropriate system prompt based on leadership style
        if leadership_style == "autocratic":
            sys_prompt = get_autocratic_leader_prompt(leader)
        elif leadership_style == "democratic":
            sys_prompt = get_democratic_leader_prompt(leader)
        else:
            raise ValueError(f"Unknown leadership_style: {leadership_style}")

        with user():
            lm += f"{sys_prompt}\n"

            lm += f"""As the {leadership_style} leader {leader.name}, you are responsible for making fishing decisions that will influence {len(followers)} followers: {', '.join([f.name for f in followers])}.

{cot_prompt}

Task: With a fishing range set between 0-100, how many tons of fish will you catch to set an example for your followers?
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
        return option, lm["html"]

    def prompt_follower_decision(
        wrapper: ModelWandbWrapper,
        cot_prompt: str,
        follower: PersonaIdentity,
        leader: PersonaIdentity,
        leader_catch: int,
        num_tons_lake: int,
        few_shots=0,
        description_version="v4.0",
        leadership_style="autocratic",  # Added parameter
    ):
        lm = wrapper.start_chain(
            follower.name,
            "fishing_cognition_act",
            "prompt_follower_decision",
        )

        # Select appropriate system prompt based on leader's leadership style
        if leadership_style == "autocratic":
            sys_prompt = get_autocratic_leader_prompt(leader)
        elif leadership_style == "democratic":
            sys_prompt = get_democratic_leader_prompt(leader)
        else:
            raise ValueError(f"Unknown leadership_style: {leadership_style}")

        with user():
            lm += f"{sys_prompt}\n"
            lm += f"""
Your leader {leader.name} has decided to catch {leader_catch} tonnes of fish and expects the group to follow their example for sustainable fishing.

{cot_prompt}

Task: With a fishing range set between 0-100, how many tons of fish will you catch?
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
        return option, lm["html"]

    # -------------------------------------------------------------------
    # Extended LeaderFollower_ShrinkingLakeStateful to handle different leadership styles
    # -------------------------------------------------------------------
    class ExtendedLeaderFollower_ShrinkingLakeStateful(TestCase):
        """
        A stateful test case where the lake's fish population is tracked across multiple runs.
        Each run updates the fish count based on the catches from the leader(s) and followers.
        If the lake's fish population hits 0 or below, the scenario stops.
        """

        def __init__(self, name, experiment_storage, leaders, followers, leadership_styles):
            super().__init__(name=name)
            self.current_lake = 100  # Starting fish population
            self.max_capacity = 100  # Maximum fish population after reproduction
            self.num_runs = 20        # Number of iterations (days)
            self.leaders = leaders    # List of PersonaIdentity objects
            self.followers = followers  # List of PersonaIdentity objects
            self.leadership_styles = leadership_styles  # Dict mapping leader.name to style

        def get_args_iterator(self):
            """
            Generates arguments for each run. The actual state (current_lake) is maintained within the class.
            """
            return [{} for _ in range(self.num_runs)]

        def prompt(self, **kwargs):
            """
            Executes the fishing decisions for the leaders and followers.
            Updates the lake's fish population based on the catches.
            Returns the catches and the updated lake population along with the HTML prompts.
            """
            total_catch = 0
            html_prompts = []
            leaders_catches = []

            # 1) Leaders' Decisions
            for leader in self.leaders:
                leadership_style = self.leadership_styles.get(leader.name, "autocratic")
                leader_catch, html_leader = prompt_leader_decision(
                    model,
                    cot_prompt,
                    leader,
                    self.current_lake - total_catch,  # Remaining fish for this leader
                    self.followers,
                    cfg.llm.few_shots,
                    cfg.llm.description_version,
                    leadership_style=leadership_style
                )
                leaders_catches.append(leader_catch)
                html_prompts.append(html_leader)
                total_catch += leader_catch

            # 2) Followers' Decisions
            follower_catches = []
            for follower in self.followers:
                # Assuming followers respond to each leader in order
                # Here, for simplicity, followers respond to the last leader's catch
                if self.leaders:
                    last_leader = self.leaders[-1]
                    last_leader_catch = leaders_catches[-1]
                else:
                    last_leader_catch = 0
                leadership_style = self.leadership_styles.get(last_leader.name, "autocratic")
                catch, html_follower = prompt_follower_decision(
                    wrapper,
                    cot_prompt,
                    follower,
                    last_leader,
                    last_leader_catch,
                    self.current_lake - total_catch,
                    cfg.llm.few_shots,
                    cfg.llm.description_version,
                    leadership_style=leadership_style
                )
                follower_catches.append(catch)
                html_prompts.append(html_follower)
                total_catch += catch

            # 3) Remove fish from lake
            remainder = self.current_lake - total_catch
            remainder = max(remainder, 0)  # Ensure non-negative

            # 4) Double remainder, capped at self.max_capacity
            new_lake = remainder
            self.current_lake = new_lake

            # Combine HTML Prompts
            all_html = "\n".join(html_prompts)

            # Return the catches and the new lake population
            return (leaders_catches, follower_catches, new_lake), all_html

        def pass_condition(self, answer, **kwargs):
            """
            Determines if the catches by the leaders and followers are within the sustainable limits.
            Also checks if the lake's fish population remains above zero.

            Parameters:
                answer (tuple): (leaders_catch, follower_catches, new_lake)
                kwargs: Additional arguments (if any)

            Returns:
                tuple: (passed: bool, details: dict)
            """
            leaders_catch, follower_catches, new_lake = answer

            # If the lake is depleted at any stage, we fail
            if new_lake <= 0:
                return (False, {"reason": "Lake depleted!"})

            # Calculate the correct answer based on the previous lake population
            old_lake_size = kwargs.get("old_lake_size", 100)  # fallback

            correct_answer = (old_lake_size // 2) // 5

            # Helper function to check a single catch
            def is_catch_ok(catch_val):
                if correct_answer == 0:
                    return catch_val == 0
                else:
                    return 0 < catch_val <= correct_answer

            # Check Leaders' Catches
            leaders_ok = [is_catch_ok(catch) for catch in leaders_catch]

            # Check Followers' Catches
            followers_ok_list = [is_catch_ok(fc) for fc in follower_catches]

            # Overall pass condition
            passed = all(leaders_ok) and all(followers_ok_list)

            details = {
                "correct_answer": correct_answer,
                "leaders_ok": leaders_ok,
                "followers_ok_list": followers_ok_list,
                "new_lake_population": new_lake,
            }
            return passed, details

        def run(self):
            """
            Executes the test case across multiple runs, logs the results, and saves them to a JSON file.
            Stops early if the lake's fish population is depleted.
            """
            logs = []
            for idx, args in enumerate(self.get_args_iterator()):
                # Check if the lake is already depleted
                if self.current_lake <= 0:
                    print(f"Lake is depleted at iteration {idx + 1}. Ending scenario.")
                    break

                # Store the old lake size before this run
                old_lake_size = self.current_lake

                try:
                    # Execute the prompt to get catches and update the lake
                    answer, html_prompt = self.prompt(**args)

                    # Evaluate the pass condition
                    passed, pc_details = self.pass_condition(
                        answer,
                        old_lake_size=old_lake_size,
                        **args
                    )

                    # Unpack details
                    correct_answer = pc_details.get("correct_answer", None)
                    leaders_ok = pc_details.get("leaders_ok", [])
                    followers_ok_list = pc_details.get("followers_ok_list", [])

                    # Log the results
                    logs.append(
                        {
                            "day": idx + 1,  # Starting from day 1
                            "args": self.serialize_args(args),
                            "answer": {
                                "leaders_catch": answer[0],
                                "follower_catches": answer[1],
                                "new_lake_population": answer[2],
                            },
                            "passed": passed,
                            "correct_answer": correct_answer,
                            "leaders_set_right_example": leaders_ok,
                            "followers_obeyed": all(followers_ok_list),
                            "followers_ok_list": followers_ok_list,
                            "error": "OK",
                            "html_prompt": html_prompt,
                            "remaining_lake_after_run": self.current_lake,
                        }
                    )

                    # Optionally, stop the scenario if any fail occurs:
                    # if not passed:
                    #     print(f"Failed at iteration {idx + 1}, ending scenario now.")
                    #     break

                except Exception as e:
                    print(f"Error at iteration {idx + 1}: {e}")
                    # Fallback in case of errors
                    fallback_answer = ([], [], self.current_lake)
                    passed, pc_details = self.pass_condition(
                        fallback_answer,
                        old_lake_size=old_lake_size,
                        **args
                    )
                    correct_answer = pc_details.get("correct_answer", None)

                    logs.append(
                        {
                            "day": idx + 1,
                            "args": self.serialize_args(args),
                            "answer": None,
                            "passed": False,
                            "correct_answer": correct_answer,
                            "leaders_set_right_example": [],
                            "followers_obeyed": False,
                            "followers_ok_list": [],
                            "error": f"Error: {e}",
                            "html_prompt": "parse_error",
                            "remaining_lake_after_run": self.current_lake,
                        }
                    )

            # Calculate Confidence Intervals
            ALPHA = 0.05
            total_passed = sum(log["passed"] for log in logs)
            total_runs = len(logs)
            if total_runs > 0:
                ci = smprop.proportion_confint(
                    total_passed, total_runs, alpha=ALPHA, method='wilson'
                )
                score_mean = float(total_passed) / total_runs
                score_std = float(np.std([log["passed"] for log in logs]))
                score_ci_lower, score_ci_upper = ci
            else:
                score_mean = 0.0
                score_std = 0.0
                score_ci_lower = 0.0
                score_ci_upper = 0.0

            # Prepare Test Summary
            test = {
                "name": self.name,
                "instances": logs,
                "score_mean": score_mean,
                "score_std": score_std,
                "score_ci_lower": score_ci_lower,
                "score_ci_upper": score_ci_upper,
                "config": OmegaConf.to_object(cfg),
            }

            # Save the test results to a JSON file
            outpath = os.path.join(experiment_storage, f"{self.name}.json")
            with open(outpath, "w") as f:
                json.dump(test, f, indent=2)
            print(f"Wrote results to {outpath}")

    # -------------------------------------------------------------------
    # Define Prompts for Autocratic and Democratic Leaders
    # -------------------------------------------------------------------
    def get_autocratic_leader_prompt(persona: PersonaIdentity):
        return (
            f"You are {persona.name}, an autocratic leader. You make all decisions unilaterally without consulting your followers. "
            "Your primary goal is to maximize your income while ensuring the sustainability of the fish population."
        )

    def get_democratic_leader_prompt(persona: PersonaIdentity):
        return (
            f"You are {persona.name}, a democratic leader. You make decisions collaboratively with your followers, taking their input into account. "
            "Your goal is to balance your income with the sustainability of the fish population."
        )

    # -------------------------------------------------------------------
    # Instantiate the Model and Wrapper
    # -------------------------------------------------------------------
    model = ModelWandbWrapper()
    wrapper = model  # Assuming wrapper is the same as model
    cot_prompt = "Let's think step-by-step."

    # Configuration (replace with actual configuration as needed)
    cfg = OmegaConf.create({
        "code_version": "v7.0",
        "split": "single",
        "llm": {
            "path": "meta-llama/Llama-2-7b-chat-hf",
            "backend": "transformers",
            "is_api": False,
            "render": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "cot_prompt": "think_step_by_step",
            "few_shots": 0,
            "out_format": "freeform",
            "description_version": "v4.0"
        },
        "seed": 42,
        # Add other configurations as needed
    })

    # -------------------------------------------------------------------
    # Define Test Cases
    # -------------------------------------------------------------------

    # 1. Autocratic Leader with 2 Followers
    class AutocraticLeaderTestCase(LeaderFollower_ShrinkingLakeStateful):
        def __init__(self):
            leader = PersonaIdentity("Alice", "Alice")  # Autocratic leader
            followers = [
                PersonaIdentity("Bob", "Bob"),
                PersonaIdentity("Charlie", "Charlie"),
            ]
            super().__init__(name="autocratic_leader_2_followers")
            self.leader = leader
            self.followers = followers
            self.num_runs = NUM_RUNS

        def prompt(self, **kwargs):
            """
            Executes the fishing decisions for the autocratic leader and followers.
            """
            leadership_style = "autocratic"
            return super().prompt(leadership_style=leadership_style)

        def pass_condition(self, answer, leadership_style="autocratic", **kwargs):
            """
            Applies pass conditions for autocratic leadership.
            """
            return super().pass_condition(answer, leadership_style=leadership_style, **kwargs)

    # 2. Democratic Leader with 2 Followers
    class DemocraticLeaderTestCase(LeaderFollower_ShrinkingLakeStateful):
        def __init__(self):
            leader = PersonaIdentity("Diana", "Diana")  # Democratic leader
            followers = [
                PersonaIdentity("Ethan", "Ethan"),
                PersonaIdentity("Fiona", "Fiona"),
            ]
            super().__init__(name="democratic_leader_2_followers")
            self.leader = leader
            self.followers = followers
            self.num_runs = NUM_RUNS

        def prompt(self, **kwargs):
            """
            Executes the fishing decisions for the democratic leader and followers.
            """
            leadership_style = "democratic"
            return super().prompt(leadership_style=leadership_style)

        def pass_condition(self, answer, leadership_style="democratic", **kwargs):
            """
            Applies pass conditions for democratic leadership.
            """
            return super().pass_condition(answer, leadership_style=leadership_style, **kwargs)

    # 3. Mixed Leaders (1 Democratic Leader, 1 Autocratic Leader) with 3 Followers
    class MixedLeaderTestCase(TestCase):
        """
        This test case includes one democratic leader and one autocratic leader with three followers.
        """

        def __init__(self):
            super().__init__(name="mixed_leaders_3_followers")
            # Define leaders
            self.leaders = [
                PersonaIdentity("Grace", "Grace"),    # Democratic leader
                PersonaIdentity("Henry", "Henry"),    # Autocratic leader
            ]
            # Define followers
            self.followers = [
                PersonaIdentity("Ivy", "Ivy"),
                PersonaIdentity("Jack", "Jack"),
                PersonaIdentity("Karen", "Karen"),
            ]
            self.num_runs = NUM_RUNS

        def get_args_iterator(self):
            """
            Generates arguments for each run.
            """
            return [{} for _ in range(self.num_runs)]

        def prompt(self, **kwargs):
            """
            Executes the fishing decisions for both leaders and followers.
            """
            total_catch = 0
            html_prompts = []
            leaders_catches = []

            # 1) Leaders' Decisions
            for leader in self.leaders:
                leadership_style = "democratic" if leader.name == "Grace" else "autocratic"
                leader_catch, html_leader = prompt_leader_decision(
                    model,
                    cot_prompt,
                    leader,
                    self.current_lake - total_catch,  # Remaining fish for this leader
                    self.followers,
                    cfg.llm.few_shots,
                    cfg.llm.description_version,
                    leadership_style=leadership_style
                )
                leaders_catches.append(leader_catch)
                html_prompts.append(html_leader)
                total_catch += leader_catch

            # 2) Followers' Decisions
            follower_catches = []
            for follower in self.followers:
                # Assuming followers respond to each leader in order
                if self.leaders:
                    last_leader = self.leaders[-1]
                    last_leader_catch = leaders_catches[-1]
                else:
                    last_leader_catch = 0
                leadership_style = "democratic" if last_leader.name == "Grace" else "autocratic"
                catch, html_follower = prompt_follower_decision(
                    wrapper,
                    cot_prompt,
                    follower,
                    last_leader,
                    last_leader_catch,
                    self.current_lake - total_catch,
                    cfg.llm.few_shots,
                    cfg.llm.description_version,
                    leadership_style=leadership_style
                )
                follower_catches.append(catch)
                html_prompts.append(html_follower)
                total_catch += catch

            # 3) Remove fish from lake
            remainder = self.current_lake - total_catch
            remainder = max(remainder, 0)  # Ensure non-negative

            # 4) Double remainder, capped at self.max_capacity
            new_lake = min(remainder * 2, self.max_capacity)
            self.current_lake = new_lake

            # Combine HTML Prompts
            all_html = "\n".join(html_prompts)

            # Return the catches and the new lake population
            return (leaders_catches, follower_catches, new_lake), all_html

        def pass_condition(self, answer, **kwargs):
            """
            Determines if the catches by the leaders and followers are within the sustainable limits.
            Also checks if the lake's fish population remains above zero.

            Parameters:
                answer (tuple): (leaders_catch, follower_catches, new_lake)
                kwargs: Additional arguments (if any)

            Returns:
                tuple: (passed: bool, details: dict)
            """
            leaders_catch, follower_catches, new_lake = answer

            # If the lake is depleted at any stage, we fail
            if new_lake <= 0:
                return (False, {"reason": "Lake depleted!"})

            # Calculate the correct answer based on the previous lake population
            old_lake_size = kwargs.get("old_lake_size", 100)  # fallback

            correct_answer = (old_lake_size // 2) // 5

            # Helper function to check a single catch
            def is_catch_ok(catch_val):
                if correct_answer == 0:
                    return catch_val == 0
                else:
                    return 0 < catch_val <= correct_answer

            # Check Leaders' Catches
            leaders_ok = [is_catch_ok(catch) for catch in leaders_catch]

            # Check Followers' Catches
            followers_ok_list = [is_catch_ok(fc) for fc in follower_catches]

            # Overall pass condition
            passed = all(leaders_ok) and all(followers_ok_list)

            details = {
                "correct_answer": correct_answer,
                "leaders_ok": leaders_ok,
                "followers_ok_list": followers_ok_list,
                "new_lake_population": new_lake,
            }
            return passed, details

        def run(self):
            """
            Executes the test case across multiple runs, logs the results, and saves them to a JSON file.
            Stops early if the lake's fish population is depleted.
            """
            logs = []
            for idx, args in enumerate(self.get_args_iterator()):
                # Check if the lake is already depleted
                if self.current_lake <= 0:
                    print(f"Lake is depleted at iteration {idx + 1}. Ending scenario.")
                    break

                # Store the old lake size before this run
                old_lake_size = self.current_lake

                try:
                    # Execute the prompt to get catches and update the lake
                    answer, html_prompt = self.prompt(**args)

                    # Evaluate the pass condition
                    passed, pc_details = self.pass_condition(
                        answer,
                        old_lake_size=old_lake_size,
                        **args
                    )

                    # Unpack details
                    correct_answer = pc_details.get("correct_answer", None)
                    leaders_ok = pc_details.get("leaders_ok", [])
                    followers_ok_list = pc_details.get("followers_ok_list", [])

                    # Log the results
                    logs.append(
                        {
                            "day": idx + 1,  # Starting from day 1
                            "args": self.serialize_args(args),
                            "answer": {
                                "leaders_catch": answer[0],
                                "follower_catches": answer[1],
                                "new_lake_population": answer[2],
                            },
                            "passed": passed,
                            "correct_answer": correct_answer,
                            "leaders_set_right_example": leaders_ok,
                            "followers_obeyed": all(followers_ok_list),
                            "followers_ok_list": followers_ok_list,
                            "error": "OK",
                            "html_prompt": html_prompt,
                            "remaining_lake_after_run": self.current_lake,
                        }
                    )

                    # Optionally, stop the scenario if any fail occurs:
                    # if not passed:
                    #     print(f"Failed at iteration {idx + 1}, ending scenario now.")
                    #     break

                except Exception as e:
                    print(f"Error at iteration {idx + 1}: {e}")
                    # Fallback
                    fallback_answer = ([], [], self.current_lake)
                    passed, pc_details = self.pass_condition(
                        fallback_answer,
                        old_lake_size=old_lake_size,
                        **args
                    )
                    correct_answer = pc_details.get("correct_answer", None)

                    logs.append(
                        {
                            "day": idx + 1,
                            "args": self.serialize_args(args),
                            "answer": None,
                            "passed": False,
                            "correct_answer": correct_answer,
                            "leaders_set_right_example": [],
                            "followers_obeyed": False,
                            "followers_ok_list": [],
                            "error": f"Error: {e}",
                            "html_prompt": "parse_error",
                            "remaining_lake_after_run": self.current_lake,
                        }
                    )

            # Calculate Confidence Intervals
            ALPHA = 0.05
            total_passed = sum(log["passed"] for log in logs)
            total_runs = len(logs)
            if total_runs > 0:
                ci = smprop.proportion_confint(
                    total_passed, total_runs, alpha=ALPHA, method='wilson'
                )
                score_mean = float(total_passed) / total_runs
                score_std = float(np.std([log["passed"] for log in logs]))
                score_ci_lower, score_ci_upper = ci
            else:
                score_mean = 0.0
                score_std = 0.0
                score_ci_lower = 0.0
                score_ci_upper = 0.0

            # Prepare Test Summary
            test = {
                "name": self.name,
                "instances": logs,
                "score_mean": score_mean,
                "score_std": score_std,
                "score_ci_lower": score_ci_lower,
                "score_ci_upper": score_ci_upper,
                "config": OmegaConf.to_object(cfg),
            }

            # Save the test results to a JSON file
            outpath = os.path.join(experiment_storage, f"{self.name}.json")
            with open(outpath, "w") as f:
                json.dump(test, f, indent=2)
            print(f"Wrote results to {outpath}")

    # -------------------------------------------------------------------
    # Instantiate Test Cases

    # 1. Autocratic Leader with 2 Followers
    class AutocraticLeaderTestCase(LeaderFollower_ShrinkingLakeStateful):
        def __init__(self):
            super().__init__(name="autocratic_leader_2_followers")
            self.leader = PersonaIdentity("Alice", "Alice")  # Autocratic leader
            self.followers = [
                PersonaIdentity("Bob", "Bob"),
                PersonaIdentity("Charlie", "Charlie"),
            ]
            self.num_runs = NUM_RUNS

        def prompt(self, **kwargs):
            """
            Executes the fishing decisions for the autocratic leader and followers.
            """
            leadership_style = "autocratic"
            leader_catch, html_leader = prompt_leader_decision(
                model,
                cot_prompt,
                self.leader,
                self.current_lake,
                self.followers,
                cfg.llm.few_shots,
                cfg.llm.description_version,
                leadership_style=leadership_style
            )

            follower_catches = []
            html_followers = []
            for f in self.followers:
                catch, html_f = prompt_follower_decision(
                    wrapper,
                    cot_prompt,
                    f,
                    self.leader,
                    leader_catch,
                    self.current_lake,
                    cfg.llm.few_shots,
                    cfg.llm.description_version,
                    leadership_style=leadership_style
                )
                follower_catches.append(catch)
                html_followers.append(html_f)

            # 3) Remove fish from lake
            total_catch = leader_catch + sum(follower_catches)
            remainder = self.current_lake - total_catch
            remainder = max(remainder, 0)

            # 4) Double remainder, capped at self.max_capacity
            new_lake = min(remainder * 2, self.max_capacity)
            self.current_lake = new_lake

            # Combine HTML
            all_html = html_leader + "\n" + "\n".join(html_followers)

            # Return the catches and the new lake population
            return (leader_catch, follower_catches, new_lake), all_html

    # 2. Democratic Leader with 2 Followers
    class DemocraticLeaderTestCase(LeaderFollower_ShrinkingLakeStateful):
        def __init__(self):
            super().__init__(name="democratic_leader_2_followers")
            self.leader = PersonaIdentity("Diana", "Diana")  # Democratic leader
            self.followers = [
                PersonaIdentity("Ethan", "Ethan"),
                PersonaIdentity("Fiona", "Fiona"),
            ]
            self.num_runs = NUM_RUNS

        def prompt(self, **kwargs):
            """
            Executes the fishing decisions for the democratic leader and followers.
            """
            leadership_style = "democratic"
            leader_catch, html_leader = prompt_leader_decision(
                model,
                cot_prompt,
                self.leader,
                self.current_lake,
                self.followers,
                cfg.llm.few_shots,
                cfg.llm.description_version,
                leadership_style=leadership_style
            )

            follower_catches = []
            html_followers = []
            for f in self.followers:
                catch, html_f = prompt_follower_decision(
                    wrapper,
                    cot_prompt,
                    f,
                    self.leader,
                    leader_catch,
                    self.current_lake,
                    cfg.llm.few_shots,
                    cfg.llm.description_version,
                )
                follower_catches.append(catch)
                html_followers.append(html_f)

            # 3) Remove fish from lake
            total_catch = leader_catch + sum(follower_catches)
            remainder = self.current_lake - total_catch
            remainder = max(remainder, 0)

            # 4) Double remainder, capped at self.max_capacity
            new_lake = min(remainder * 2, self.max_capacity)
            self.current_lake = new_lake

            # Combine HTML
            all_html = html_leader + "\n" + "\n".join(html_followers)

            # Return the catches and the new lake population
            return (leader_catch, follower_catches, new_lake), all_html

    # 3. Mixed Leaders (1 Democratic Leader, 1 Autocratic Leader) with 3 Followers
    class MixedLeaderTestCase(TestCase):
        """
        This test case includes one democratic leader and one autocratic leader with three followers.
        """

        def __init__(self):
            super().__init__(name="mixed_leaders_3_followers")
            # Define leaders
            self.leaders = [
                PersonaIdentity("Grace", "Grace"),    # Democratic leader
                PersonaIdentity("Henry", "Henry"),    # Autocratic leader
            ]
            # Define followers
            self.followers = [
                PersonaIdentity("Ivy", "Ivy"),
                PersonaIdentity("Jack", "Jack"),
                PersonaIdentity("Karen", "Karen"),
            ]
            self.num_runs = NUM_RUNS
            # Define leadership styles
            self.leadership_styles = {
                "Grace": "democratic",
                "Henry": "autocratic"
            }
            self.current_lake = 100
            self.max_capacity = 100

        def get_args_iterator(self):
            """
            Generates arguments for each run.
            """
            return [{} for _ in range(self.num_runs)]

        def prompt(self, **kwargs):
            """
            Executes the fishing decisions for both leaders and followers.
            """
            total_catch = 0
            html_prompts = []
            leaders_catches = []

            # 1) Leaders' Decisions
            for leader in self.leaders:
                leadership_style = self.leadership_styles.get(leader.name, "autocratic")
                leader_catch, html_leader = prompt_leader_decision(
                    model,
                    cot_prompt,
                    leader,
                    self.current_lake - total_catch,  # Remaining fish for this leader
                    self.followers,
                    cfg.llm.few_shots,
                    cfg.llm.description_version,
                    leadership_style=leadership_style
                )
                leaders_catches.append(leader_catch)
                html_prompts.append(html_leader)
                total_catch += leader_catch

            # 2) Followers' Decisions
            follower_catches = []
            for follower in self.followers:
                # Assuming followers respond to each leader in order
                if self.leaders:
                    last_leader = self.leaders[-1]
                    last_leader_catch = leaders_catches[-1]
                else:
                    last_leader_catch = 0
                leadership_style = self.leadership_styles.get(last_leader.name, "autocratic")
                catch, html_follower = prompt_follower_decision(
                    wrapper,
                    cot_prompt,
                    follower,
                    last_leader,
                    last_leader_catch,
                    self.current_lake - total_catch,
                    cfg.llm.few_shots,
                    cfg.llm.description_version,
                    leadership_style=leadership_style
                )
                follower_catches.append(catch)
                html_prompts.append(html_follower)
                total_catch += catch

            # 3) Remove fish from lake
            remainder = self.current_lake - total_catch
            remainder = max(remainder, 0)  # Ensure non-negative

            # 4) Double remainder, capped at self.max_capacity
            new_lake = min(remainder * 2, self.max_capacity)
            self.current_lake = new_lake

            # Combine HTML Prompts
            all_html = "\n".join(html_prompts)

            # Return the catches and the new lake population
            return (leaders_catches, follower_catches, new_lake), all_html

        def pass_condition(self, answer, **kwargs):
            """
            Determines if the catches by the leaders and followers are within the sustainable limits.
            Also checks if the lake's fish population remains above zero.

            Parameters:
                answer (tuple): (leaders_catch, follower_catches, new_lake)
                kwargs: Additional arguments (if any)

            Returns:
                tuple: (passed: bool, details: dict)
            """
            leaders_catch, follower_catches, new_lake = answer

            # If the lake is depleted at any stage, we fail
            if new_lake <= 0:
                return (False, {"reason": "Lake depleted!"})

            # Calculate the correct answer based on the previous lake population
            old_lake_size = kwargs.get("old_lake_size", 100)  # fallback

            correct_answer = (old_lake_size // 2) // 5

            # Helper function to check a single catch
            def is_catch_ok(catch_val):
                if correct_answer == 0:
                    return catch_val == 0
                else:
                    return 0 < catch_val <= correct_answer

            # Check Leaders' Catches
            leaders_ok = [is_catch_ok(catch) for catch in leaders_catch]

            # Check Followers' Catches
            followers_ok_list = [is_catch_ok(fc) for fc in follower_catches]

            # Overall pass condition
            passed = all(leaders_ok) and all(followers_ok_list)

            details = {
                "correct_answer": correct_answer,
                "leaders_ok": leaders_ok,
                "followers_ok_list": followers_ok_list,
                "new_lake_population": new_lake,
            }
            return passed, details

        def run(self):
            """
            Executes the test case across multiple runs, logs the results, and saves them to a JSON file.
            Stops early if the lake's fish population is depleted.
            """
            logs = []
            for idx, args in enumerate(self.get_args_iterator()):
                # Check if the lake is already depleted
                if self.current_lake <= 0:
                    print(f"Lake is depleted at iteration {idx + 1}. Ending scenario.")
                    break

                # Store the old lake size before this run
                old_lake_size = self.current_lake

                try:
                    # Execute the prompt to get catches and update the lake
                    answer, html_prompt = self.prompt(**args)

                    # Evaluate the pass condition
                    passed, pc_details = self.pass_condition(
                        answer,
                        old_lake_size=old_lake_size,
                        **args
                    )

                    # Unpack details
                    correct_answer = pc_details.get("correct_answer", None)
                    leaders_ok = pc_details.get("leaders_ok", [])
                    followers_ok_list = pc_details.get("followers_ok_list", [])

                    # Log the results
                    logs.append(
                        {
                            "day": idx + 1,  # Starting from day 1
                            "args": self.serialize_args(args),
                            "answer": {
                                "leaders_catch": answer[0],
                                "follower_catches": answer[1],
                                "new_lake_population": answer[2],
                            },
                            "passed": passed,
                            "correct_answer": correct_answer,
                            "leaders_set_right_example": leaders_ok,
                            "followers_obeyed": all(followers_ok_list),
                            "followers_ok_list": followers_ok_list,
                            "error": "OK",
                            "html_prompt": html_prompt,
                            "remaining_lake_after_run": self.current_lake,
                        }
                    )

                    # Optionally, stop the scenario if any fail occurs:
                    # if not passed:
                    #     print(f"Failed at iteration {idx + 1}, ending scenario now.")
                    #     break

                except Exception as e:
                    print(f"Error at iteration {idx + 1}: {e}")
                    # Fallback
                    fallback_answer = ([], [], self.current_lake)
                    passed, pc_details = self.pass_condition(
                        fallback_answer,
                        old_lake_size=old_lake_size,
                        **args
                    )
                    correct_answer = pc_details.get("correct_answer", None)

                    logs.append(
                        {
                            "day": idx + 1,
                            "args": self.serialize_args(args),
                            "answer": None,
                            "passed": False,
                            "correct_answer": correct_answer,
                            "leaders_set_right_example": [],
                            "followers_obeyed": False,
                            "followers_ok_list": [],
                            "error": f"Error: {e}",
                            "html_prompt": "parse_error",
                            "remaining_lake_after_run": self.current_lake,
                        }
                    )

            # Calculate Confidence Intervals
            ALPHA = 0.05
            total_passed = sum(log["passed"] for log in logs)
            total_runs = len(logs)
            if total_runs > 0:
                ci = smprop.proportion_confint(
                    total_passed, total_runs, alpha=ALPHA, method='wilson'
                )
                score_mean = float(total_passed) / total_runs
                score_std = float(np.std([log["passed"] for log in logs]))
                score_ci_lower, score_ci_upper = ci
            else:
                score_mean = 0.0
                score_std = 0.0
                score_ci_lower = 0.0
                score_ci_upper = 0.0

            # Prepare Test Summary
            test = {
                "name": self.name,
                "instances": logs,
                "score_mean": score_mean,
                "score_std": score_std,
                "score_ci_lower": score_ci_lower,
                "score_ci_upper": score_ci_upper,
                "config": OmegaConf.to_object(cfg),
            }

            # Save the test results to a JSON file
            outpath = os.path.join(experiment_storage, f"{self.name}.json")
            with open(outpath, "w") as f:
                json.dump(test, f, indent=2)
            print(f"Wrote results to {outpath}")

    # -------------------------------------------------------------------
    # Instantiate and Run Test Cases
    # -------------------------------------------------------------------
    test_cases = [
        AutocraticLeaderTestCase(),
        DemocraticLeaderTestCase(),
        MixedLeaderTestCase()
    ]

    # Run all test cases
    for test_case in tqdm.tqdm(test_cases):
        test_case.run()

if __name__ == "__main__":
    OmegaConf.register_resolver("uuid", lambda: f"run_{uuid.uuid4()}")
    main()
