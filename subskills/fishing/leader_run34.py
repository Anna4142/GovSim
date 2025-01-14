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
    NUM_RUNS = 100
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
    #
    # -------------------------------------------------------------------
    class LeaderFollower_TestMechanism(TestCase):
                """
                Similar to the BaseLeaderFollowerTestCase, but we keep track of the
                lake across runs. Each run uses the updated fish count from the previous run.
                If it hits 0 or below, we stop the scenario.
                """

                def __init__(self, name="leader_follower_shrinking_lake_stateful"):
                    super().__init__(name)
                    self.current_lake = 100
                    self.max_capacity = 100
                    self.num_runs = 100
                    self.leader = PersonaIdentity("Emma", "Emma")
                    self.followers = [
                        PersonaIdentity("Follower1", "Folllower1"),
                        PersonaIdentity("Follower2", "Folllower2"),
                        PersonaIdentity("Follower3", "Folllower3"),
                        PersonaIdentity("Follower4", "Folllower4"),
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
                    4) Double what's left (capped at 100)
                    """
                    # 1) Leader
                    self.leader_type="neutral"
                    follower_catches = []
                    leader_catch, html_leader = prompt_leader_decision(
                        wrapper,
                        cot_prompt,
                        self.leader,
                        self.current_lake,
                        self.followers,
                        follower_catches,
                        cfg.llm.few_shots,
                        cfg.llm.description_version,
                    )

                    # 2) Followers
                   
                    html_followers = []
                    for f in self.followers:
                        catch, html_f = prompt_follower_decision(
                            wrapper,
                            cot_prompt,
                            f,
                            self.leader,
                            self.leader_type,
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
                    new_lake = np.random.randint(10, 101)
                    self.current_lake = new_lake

                    # Combine HTML
                    all_html = html_leader + "\n" + "\n".join(html_followers)

                    # Return the final answer: (leader_catch, follower_catches, new_lake)
                    return (leader_catch, follower_catches, self.current_lake), all_html

                def pass_condition(self, answer, **kwargs):
                    """
                    Apply passing conditions for leader and followers who fish simultaneously.
                    Each person's correct answer depends on their own catch and the new lake size.
                    
                    Args:
                        answer: Tuple of (leader_catch, follower_catches, new_lake)
                        
                    Returns:
                        tuple: (passed, details_dict)
                    """
                    leader_catch, follower_catches, new_lake = answer
                    
                    # Calculate correct answer for leader using leader's catch
                    leader_correct_answer = max(
                        0, min(100, (new_lake - leader_catch * 5) * 2)
                    )
                    
                    # Helper function to check if a catch is valid
                    def is_catch_ok(catch_val, correct_val):
                        if correct_val == 0:
                            return catch_val == 0
                        else:
                            return 0 < catch_val <= correct_val
                    
                    # Check leader's catch
                    leader_ok = is_catch_ok(leader_catch, leader_correct_answer)
                    
                    # Calculate and check each follower's catch using their own catch
                    followers_ok_list = []
                    follower_correct_answers = []
                    
                    for fc in follower_catches:
                        follower_correct = max(
                            0, min(100, (new_lake - fc * 5) * 2)
                        )
                        follower_correct_answers.append(follower_correct)
                        followers_ok_list.append(is_catch_ok(fc, follower_correct))
                    
                    # Pass only if leader and all followers pass
                    passed = leader_ok and all(followers_ok_list)
                    
                    details = {
                        "leader_correct_answer": leader_correct_answer,
                        "follower_correct_answers": follower_correct_answers,
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
                                    "day": idx,
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
                            #     print(f"Failed at iteration {idx}, ending scenario now.")
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
                                    "day": idx,
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

    class DemocraticLeader_TestMechanism(TestCase):
        """
            Test case with one democratic leader and two followers.
            Tracks lake population across runs.
            """
        def __init__(self, name="democratic_leader_stateful"):
            super().__init__(name)
            self.current_lake = 100
            self.max_capacity = 100
            self.num_runs = 100
            # Democratic leader
            self.leader = PersonaIdentity("Diana", "Diana")
            # Followers
            self.followers = [
                PersonaIdentity("Follower1", "Folllower1"),
                PersonaIdentity("Follower2", "Folllower2"),
                PersonaIdentity("Follower3", "Folllower3"),
                PersonaIdentity("Follower4", "Folllower4")
            ]

        def get_args_iterator(self):
            return [{} for _ in range(self.num_runs)]

        def prompt(self, **kwargs):
            """
            Execute fishing decisions with democratic leadership style
            """
            follower_catches = []
            self.leader_type="democratic and kind"
            # 1) Leader's democratic decision
            leader_catch, html_leader = prompt_leader_democratic_decision(
                wrapper,
                cot_prompt,
                self.leader,
                self.current_lake,
                self.followers,
                follower_catches,
                cfg.llm.few_shots,
                cfg.llm.description_version,
            )

           
            # 2) Followers' decisions
            
            html_followers = []
            for f in self.followers:
                catch, html_f = prompt_follower_decision(
                    wrapper,
                    cot_prompt,
                    f,
                    self.leader,
                    self.leader_type,
                    leader_catch,
                    self.current_lake,
                    cfg.llm.few_shots,
                    cfg.llm.description_version,
                )
                follower_catches.append(catch)
                html_followers.append(html_f)

            # 3) Update lake population
            total_catch = leader_catch + sum(follower_catches)
            remainder = self.current_lake - total_catch
            if remainder < 0:
                remainder = 0
            
            new_lake = np.random.randint(10, 101)
            self.current_lake = new_lake

            # Combine HTML
            all_html = html_leader + "\n" + "\n".join(html_followers)

            return (leader_catch, follower_catches, new_lake), all_html

        def pass_condition(self, answer, **kwargs):
                    """
                    Apply passing conditions for leader and followers who fish simultaneously.
                    Each person's correct answer depends on their own catch and the new lake size.
                    
                    Args:
                        answer: Tuple of (leader_catch, follower_catches, new_lake)
                        
                    Returns:
                        tuple: (passed, details_dict)
                    """
                    leader_catch, follower_catches, new_lake = answer
                    
                    # Calculate correct answer for leader using leader's catch
                    leader_correct_answer = max(
                        0, min(100, (new_lake - leader_catch * 5) * 2)
                    )
                    
                    # Helper function to check if a catch is valid
                    def is_catch_ok(catch_val, correct_val):
                        if correct_val == 0:
                            return catch_val == 0
                        else:
                            return 0 < catch_val <= correct_val
                    
                    # Check leader's catch
                    leader_ok = is_catch_ok(leader_catch, leader_correct_answer)
                    
                    # Calculate and check each follower's catch using their own catch
                    followers_ok_list = []
                    follower_correct_answers = []
                    
                    for fc in follower_catches:
                        follower_correct = max(
                            0, min(100, (new_lake - fc * 5) * 2)
                        )
                        follower_correct_answers.append(follower_correct)
                        followers_ok_list.append(is_catch_ok(fc, follower_correct))
                    
                    # Pass only if leader and all followers pass
                    passed = leader_ok and all(followers_ok_list)
                    
                    details = {
                        "leader_correct_answer": leader_correct_answer,
                        "follower_correct_answers": follower_correct_answers,
                        "leader_ok": leader_ok,
                        "followers_ok_list": followers_ok_list,
                    }
                    
                    return passed, details
        def run(self):
            """
            Execute test case with logging and tracking
            """
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

                    correct_answer = pc_details.get("correct_answer", None)
                    leader_ok = pc_details.get("leader_ok", False)
                    followers_ok_list = pc_details.get("followers_ok_list", [])

                    logs.append({
                        "day": idx,
                        "args": self.serialize_args(args),
                        "answer": answer,
                        "passed": passed,
                        "correct_answer": correct_answer,
                        "leader_set_right_example": leader_ok,
                        "followers_obeyed": all(followers_ok_list),
                        "followers_ok_list": followers_ok_list,
                        "error": "OK",
                        "html_prompt": html_prompt,
                        "remaining_lake_after_run": self.current_lake,
                        "leadership_style": "democratic"
                    })

                except Exception as e:
                    print(f"Error: {e}")
                    fallback_answer = (0, [], self.current_lake)
                    passed, pc_details = self.pass_condition(
                        fallback_answer, old_lake_size=old_lake_size, **args
                    )
                    correct_answer = pc_details.get("correct_answer", None)

                    logs.append({
                        "day": idx,
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
                        "leadership_style": "democratic"
                    })

            # Calculate statistics
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


    class AutocraticLeader_TestMechanism(DemocraticLeader_TestMechanism):
        """
        Test case with one autocratic leader and two followers.
        Inherits from DemocraticLeaderStateful but modifies the prompts and leadership style.
        """
        def __init__(self, name="autocratic_leader_stateful"):
            super().__init__(name)
            # Override with autocratic leader
            self.leader = PersonaIdentity("Alexander", "Alexander")
            # Different followers
            self.followers = [
                PersonaIdentity("Follower1", "Folllower1"),
                PersonaIdentity("Follower2", "Folllower2"),
                PersonaIdentity("Follower3", "Folllower3"),
                PersonaIdentity("Follower4", "Folllower4")
                
            ]

        def prompt(self, **kwargs):
            """
            Modified prompt method to emphasize autocratic leadership style
            """
            follower_catches = []
            html_followers = []
            self.leader_type="autocratic and strict"
            # 1) Leader's autocratic decision
            leader_catch, html_leader = prompt_leader_autocratic_decision(
                wrapper,
                cot_prompt,
                self.leader,
                self.current_lake,
                self.followers,
                follower_catches,
                cfg.llm.few_shots,
                cfg.llm.description_version,
            )

            

            # Rest of the method remains the same as parent class
           
            for f in self.followers:
                catch, html_f = prompt_follower_decision(
                    wrapper,
                    cot_prompt,
                    f,
                    self.leader,
                    self.leader_type,
                    leader_catch,
                    self.current_lake,
                    cfg.llm.few_shots,
                    cfg.llm.description_version,
                )
                follower_catches.append(catch)
                html_followers.append(html_f)

            total_catch = leader_catch + sum(follower_catches)
            remainder = self.current_lake - total_catch
            if remainder < 0:
                remainder = 0
            
            new_lake = np.random.randint(10, 101)
            self.current_lake = new_lake

            all_html = html_leader + "\n" + "\n".join(html_followers)

            return (leader_catch, follower_catches, new_lake), all_html


# Update the test cases list to include both new test cases

    # -------------------------------------------------------------------
    # Assemble test cases, including the new "LeaderFollower_ShrinkingLakeStateful"
    # -------------------------------------------------------------------
    
    test_cases = [
    #LeaderFollower_TestMechanism(),
    DemocraticLeader_TestMechanism(),
    AutocraticLeader_TestMechanism()
]
    
    # Run
    for test_case in tqdm.tqdm(test_cases):
        test_case.run()


if __name__ == "__main__":
    OmegaConf.register_resolver("uuid", lambda: f"run_{uuid.uuid4()}")
    main()
''''
Storage name: feasible-moon-68-j0mosyd9
  0% 0/3 [00:00<?, ?it/s]Lake is depleted at iteration 3. Ending.
Wrote results to /content/GovSim/subskills/fishing/./results/subskills_check_v7.0/feasible-moon-68/leader_follower_shrinking_lake_stateful.json
 33% 1/3 [02:06<04:13, 126.74s/it]Lake depleted at iteration 1. Ending scenario.
Wrote results to /content/GovSim/subskills/fishing/./results/subskills_check_v7.0/feasible-moon-68/democratic_leader_stateful.json
 67% 2/3 [02:49<01:17, 77.49s/it] Lake depleted at iteration 1. Ending scenario.
Wrote results to /content/GovSim/subskills/fishing/./results/subskills_check_v7.0/feasible-moon-68/autocratic_leader_stateful.json
100% 3/3 [03:16<00:00, 65.54s/it]
'''