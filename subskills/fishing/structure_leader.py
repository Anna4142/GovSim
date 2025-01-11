import json
import os
import shutil
import uuid
import logging
import traceback

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
   
    prompt_simple_reflection_if_all_fisher_that_same_quantity,
    prompt_simple_shrinking_limit,
    prompt_simple_shrinking_limit_assumption,
    prompt_follower_decision,
    prompt_leader_autocratic_decision,
    prompt_leader_democratic_decision
)

# -------------------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all levels of logs
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        # Uncomment the following line to log to a file
        # logging.FileHandler("simulation_debug.log"),
    ]
)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Main Simulation Function
# -------------------------------------------------------------------
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    logger.debug("Starting main function with configuration:")
    logger.debug(OmegaConf.to_yaml(cfg))
    
    set_seed(cfg.seed)

    # Initialize model + wandb_logger
    try:
        model = get_model(cfg.llm.path, cfg.llm.is_api, cfg.seed, cfg.llm.backend)
        logger.debug(f"Model initialized with path: {cfg.llm.path}, is_api: {cfg.llm.is_api}, backend: {cfg.llm.backend}")
    except Exception as e:
        logger.error("Failed to initialize the model.", exc_info=True)
        raise e

    try:
        wandb_logger = WandbLogger(
            f"subskills_check/fishing/{cfg.code_version}",
            OmegaConf.to_object(cfg),
            debug=cfg.debug,
        )
        logger.debug("WandbLogger initialized successfully.")
    except Exception as e:
        logger.error("Failed to initialize WandbLogger.", exc_info=True)
        raise e

    # Set up results directory
    try:
        experiment_storage = os.path.join(
            os.path.dirname(__file__),
            f"./results/subskills_check_{cfg.code_version}/{wandb_logger.run_name}",
        )
        os.makedirs(experiment_storage, exist_ok=True)
        logger.debug(f"Experiment storage directory created at: {experiment_storage}")
    except Exception as e:
        logger.error("Failed to create experiment storage directory.", exc_info=True)
        raise e

    # Wrap model for logging/temperature/etc.
    try:
        wrapper = ModelWandbWrapper(
            model,
            render=cfg.llm.render,
            wanbd_logger=wandb_logger,
            temperature=cfg.llm.temperature,
            top_p=cfg.llm.top_p,
            seed=cfg.seed,
            is_api=cfg.llm.is_api,
        )
        logger.debug("ModelWandbWrapper initialized successfully.")
    except Exception as e:
        logger.error("Failed to initialize ModelWandbWrapper.", exc_info=True)
        raise e

    # Check out_format
    if cfg.llm.out_format == "freeform":
        logger.debug("Output format set to 'freeform'.")
    else:
        logger.error(f"Unknown out_format: {cfg.llm.out_format}")
        raise ValueError(f"Unknown out_format: {cfg.llm.out_format}")

    # Handle CoT prompt
    if cfg.llm.cot_prompt == "deep_breath":
        cot_prompt = "Take a deep breath and work on this problem step-by-step."
        logger.debug("CoT prompt set to 'deep_breath'.")
    elif cfg.llm.cot_prompt == "think_step_by_step":
        cot_prompt = "Let's think step-by-step."
        logger.debug("CoT prompt set to 'think_step_by_step'.")
    else:
        logger.error(f"Unknown cot_prompt: {cfg.llm.cot_prompt}")
        raise ValueError(f"Unknown cot_prompt: {cfg.llm.cot_prompt}")

    # By default, we'll do 5 runs for each test
    NUM_RUNS = 5
    if cfg.debug:
        NUM_RUNS = 2
        logger.debug("Debug mode is ON. Setting NUM_RUNS to 2.")
    else:
        logger.debug(f"Setting NUM_RUNS to {NUM_RUNS}.")

    # -------------------------------------------------------------------
    # Base TestCase
    # -------------------------------------------------------------------
    class TestCase:
        name: str

        def __init__(self, name) -> None:
            self.name = name
            logger.debug(f"Initialized TestCase with name: {self.name}")

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
                    logger.error(f"Error: {e}", exc_info=True)
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
            try:
                passed_count = sum(log["passed"] for log in logs)
                total = len(logs)
                ci = smprop.proportion_confint(passed_count, total, alpha=ALPHA)
                score_mean = float(np.mean([log["passed"] for log in logs]))
                score_std = float(np.std([log["passed"] for log in logs]))
                logger.debug(f"TestCase '{self.name}' results - Passed: {passed_count}/{total}, "
                             f"Mean: {score_mean}, Std: {score_std}, CI: {ci}")
            except Exception as e:
                logger.error(f"Error calculating confidence intervals: {e}", exc_info=True)
                ci = (0.0, 0.0)
                score_mean = score_std = 0.0

            test = {
                "name": self.name,
                "instances": logs,
                "score_mean": score_mean,
                "score_std": score_std,
                "score_ci_lower": ci[0],
                "score_ci_upper": ci[1],
                "config": OmegaConf.to_object(cfg),
            }

            try:
                outpath = os.path.join(experiment_storage, f"{self.name}.json")
                with open(outpath, "w") as f:
                    json.dump(test, f, indent=2)
                logger.info(f"Wrote TestCase results to {outpath}")
            except Exception as e:
                logger.error(f"Failed to write TestCase results to {outpath}: {e}", exc_info=True)

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
            logger.debug(f"Serialized args: {res}")
            return res

    # -------------------------------------------------------------------
    # Base Leader-Follower Test Case with Prompt Function Selection
    # -------------------------------------------------------------------
    class LeaderFollowerTestCase(TestCase):
        """
        Base class for leader-follower test scenarios.
        Allows selection of prompt functions for leader and followers.
        """

        def __init__(self, name, leader_prompt_funcs, follower_prompt_funcs) -> None:
            super().__init__(name)
            self.current_lake = 100
            self.max_capacity = 100
            self.num_runs = NUM_RUNS
            self.leader = PersonaIdentity("Leader", "Leader")
            self.followers = [
                PersonaIdentity("Follower1", "Follower1"),
                PersonaIdentity("Follower2", "Follower2"),
            ]
            self.leader_prompt_funcs = leader_prompt_funcs  # List of leader prompt functions
            self.follower_prompt_funcs = follower_prompt_funcs  # List of follower prompt functions
            self.run_counter = 0
            logger.debug(f"{self.name} initialized with current_lake={self.current_lake}, num_runs={self.num_runs}")

        def get_args_iterator(self):
            return [{"run": i} for i in range(self.num_runs)]

        def prompt(self, **kwargs):
            """
            Execute fishing decisions using selected prompt functions.
            Applies one set of prompt functions per run in a cyclic manner.
            """
            try:
                # Select prompt functions based on run counter
                leader_prompt = self.leader_prompt_funcs[self.run_counter % len(self.leader_prompt_funcs)]
                follower_prompt = self.follower_prompt_funcs[self.run_counter % len(self.follower_prompt_funcs)]

                logger.debug(f"{self.name} - Run {self.run_counter + 1}: Using leader prompt '{leader_prompt.__name__}' and follower prompt '{follower_prompt.__name__}'.")

                # 1) Leader's decision
                leader_catch, html_leader = leader_prompt(
                    model=wrapper,
                    cot_prompt=cot_prompt,
                    leader=self.leader,
                    num_tons_lake=self.current_lake,
                    followers=self.followers,
                    few_shots=cfg.llm.few_shots,
                    description_version=cfg.llm.description_version,
                )

                # 2) Followers' decisions
                follower_catches = []
                html_followers = []
                for f in self.followers:
                    catch, html_f = follower_prompt(
                        model=wrapper,
                        cot_prompt=cot_prompt,
                        leader=self.leader,
                        num_tons_lake=self.current_lake,
                        followers=self.followers,
                        few_shots=cfg.llm.few_shots,
                        description_version=cfg.llm.description_version,
                    )
                    follower_catches.append(catch)
                    html_followers.append(html_f)

                # 3) Update lake population
                total_catch = leader_catch + sum(follower_catches)
                remainder = self.current_lake - total_catch
                if remainder < 0:
                    remainder = 0

                # 4) Update lake (could include doubling logic or other state changes)
                self.current_lake = remainder

                # 5) Combine HTML prompts
                all_html = html_leader + "\n" + "\n".join(html_followers)

                # Increment run counter
                self.run_counter += 1

                return (leader_catch, follower_catches, self.current_lake), all_html

            except Exception as e:
                logger.error(f"{self.name} - Error in prompt execution: {e}", exc_info=True)
                return (0, [0] * len(self.followers), self.current_lake), "Error in prompt execution"

        def pass_condition(self, answer, **kwargs):
            """
            Evaluation criteria:
            - correct_answer = (old_lake_size // 2) // 5
            - if correct_answer == 0 => must catch exactly 0
            - else => must catch >0 and <= correct_answer
            """
            leader_catch, follower_catches, new_lake = answer

            if new_lake <= 0:
                logger.warning(f"{self.name} - Lake depleted!")
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
            logger.debug(f"{self.name} - Pass condition: leader_ok={leader_ok}, followers_ok_list={followers_ok_list}, passed={passed}")

            details = {
                "correct_answer": correct_answer,
                "leader_ok": leader_ok,
                "followers_ok_list": followers_ok_list,
            }
            return passed, details

    # -------------------------------------------------------------------
    # Subclasses for Different Prompt Function Combinations
    # -------------------------------------------------------------------
    class LeaderFollower_ShrinkingLakeStateful(LeaderFollowerTestCase):
        """
        A stateful test class where the lake updates each instance.
        Fish deplete (<=0), we end early. The fish also doubles after each step.
        """

        def __init__(self, name="leader_follower_shrinking_lake_stateful"):
            # Define leader and follower prompt functions
            leader_prompt_funcs = [
                prompt_action_choose_amount_of_fish_to_catch,
                prompt_action_choose_amount_of_fish_to_catch_universalization,
                prompt_reflection_if_all_fisher_that_same_quantity,
                prompt_shrinking_limit,
                prompt_shrinking_limit_assumption,
                prompt_simple_reflection_if_all_fisher_that_same_quantity,
                prompt_simple_shrinking_limit,
                prompt_simple_shrinking_limit_assumption,
            ]

            follower_prompt_funcs = [
                prompt_action_choose_amount_of_fish_to_catch,
                prompt_action_choose_amount_of_fish_to_catch_universalization,
                prompt_reflection_if_all_fisher_that_same_quantity,
                prompt_shrinking_limit,
                prompt_shrinking_limit_assumption,
                prompt_simple_reflection_if_all_fisher_that_same_quantity,
                prompt_simple_shrinking_limit,
                prompt_simple_shrinking_limit_assumption,
            ]

            super().__init__(name, leader_prompt_funcs, follower_prompt_funcs)

        def prompt(self, **kwargs):
            """
            Extend the base prompt method to include lake doubling after catch.
            """
            answer, html_prompt = super().prompt(**kwargs)
            leader_catch, follower_catches, new_lake = answer

            # Double the remaining fish, capped at max_capacity
            doubled_lake = min(new_lake * 2, self.max_capacity)
            logger.debug(f"{self.name} - Doubling lake from {new_lake} to {doubled_lake}.")
            self.current_lake = doubled_lake

            # Update the answer with doubled lake
            updated_answer = (leader_catch, follower_catches, self.current_lake)
            return updated_answer, html_prompt

        def pass_condition(self, answer, **kwargs):
            """
            Override to use the old lake size before doubling.
            """
            leader_catch, follower_catches, new_lake = answer
            # Calculate old_lake_size by reversing the doubling
            old_lake_size = min(new_lake / 2, self.max_capacity)
            correct_answer = (int(old_lake_size) // 2) // 5

            def is_catch_ok(catch_val):
                if correct_answer == 0:
                    return catch_val == 0
                return 0 < catch_val <= correct_answer

            leader_ok = is_catch_ok(leader_catch)
            followers_ok_list = [is_catch_ok(fc) for fc in follower_catches]
            
            passed = leader_ok and all(followers_ok_list)
            logger.debug(f"{self.name} - Pass condition: leader_ok={leader_ok}, followers_ok_list={followers_ok_list}, passed={passed}")

            details = {
                "correct_answer": correct_answer,
                "leader_ok": leader_ok,
                "followers_ok_list": followers_ok_list,
            }
            return passed, details

    class DemocraticLeaderTestCase(LeaderFollowerTestCase):
        """
        Test case with a democratic leader and two followers.
        Uses specified prompt functions.
        """

        def __init__(self, name="democratic_leader_test_case"):
            leader_prompt_funcs = [
                prompt_leader_democratic_decision,  # Use the democratic prompt function
            ]

            follower_prompt_funcs = [
                prompt_follower_decision,  # Assuming this uses appropriate follower prompts
            ]

            super().__init__(name, leader_prompt_funcs, follower_prompt_funcs)

    class AutocraticLeaderTestCase(LeaderFollowerTestCase):
        """
        Test case with an autocratic leader and two followers.
        Uses specified prompt functions.
        """

        def __init__(self, name="autocratic_leader_test_case"):
            leader_prompt_funcs = [
                prompt_leader_autocratic_decision,  # Use the autocratic prompt function
            ]

            follower_prompt_funcs = [
                prompt_follower_decision,  # Assuming this uses appropriate follower prompts
            ]

            super().__init__(name, leader_prompt_funcs, follower_prompt_funcs)

    # -------------------------------------------------------------------
    # Assemble Test Cases
    # -------------------------------------------------------------------
    test_cases = [
        LeaderFollower_ShrinkingLakeStateful(),
        DemocraticLeaderTestCase(),
        AutocraticLeaderTestCase()
    ]

    logger.debug("All test cases initialized.")

    # -------------------------------------------------------------------
    # Run Tests
    # -------------------------------------------------------------------
    for test_case in tqdm.tqdm(test_cases, desc="Running Test Cases"):
        logger.info(f"Starting TestCase: {test_case.name}")
        test_case.run()
        logger.info(f"Completed TestCase: {test_case.name}")

# -------------------------------------------------------------------
# Define Context Managers for User and Assistant
# -------------------------------------------------------------------
# Note: Ensure that these context managers are defined or imported appropriately.
# They are used here as placeholders. Replace them with actual implementations.

from contextlib import contextmanager

@contextmanager
def user():
    """Context manager representing user input."""
    # Implement any necessary setup here
    yield
    # Implement any necessary teardown here

@contextmanager
def assistant():
    """Context manager representing assistant response."""
    # Implement any necessary setup here
    yield
    # Implement any necessary teardown here

if __name__ == "__main__":
    OmegaConf.register_resolver("uuid", lambda: f"run_{uuid.uuid4()}")
    main()
