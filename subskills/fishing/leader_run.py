import json
import os
import uuid
import numpy as np
import statsmodels.stats.proportion as smprop
import tqdm
from omegaconf import DictConfig, OmegaConf
from transformers import set_seed
import hydra
import wandb
from simulation.persona.common import PersonaIdentity
from simulation.utils import ModelWandbWrapper, WandbLogger
from pathfinder import get_model

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    model = get_model(cfg.llm.path, cfg.llm.is_api, cfg.seed, cfg.llm.backend)
    logger = WandbLogger(
        f"subskills_check/fishing_leader/{cfg.code_version}",
        OmegaConf.to_object(cfg),
        debug=cfg.debug,
    )

    experiment_storage = os.path.join(
        os.path.dirname(__file__),
        f"./results/subskills_check_leader_{cfg.code_version}/{logger.run_name}",
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

    cot_prompt = "Let's think step-by-step about this leadership decision."

    NUM_RUNS = 150
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
                "config": OmegaConf.to_object(cfg),
            }
            json.dump(test, open(f"{experiment_storage}/{self.name}.json", "w"))

        def serialize_args(self, args: dict[str, any]):
            res = {}
            for k, v in args.items():
                if isinstance(v, PersonaIdentity):
                    res[k] = {
                        'name': v.name,
                        'id': v.agent_id,
                        'role': v.role
                    }
                else:
                    res[k] = v
            return res

    def prompt_leader_decision_making(wrapper, cot_prompt, persona, num_tonnes_lake, num_followers, sustainability_focus, leadership_style):
        prompt = f"""
        As {persona.name}, you are a {leadership_style} leader responsible for guiding {num_followers} people in fishing decisions.
        The lake has {num_tonnes_lake} tonnes of fish.
        {'You prioritize long-term sustainability and community well-being.' if sustainability_focus else 'You balance immediate needs with sustainability while considering economic factors.'}
        
        {cot_prompt}
        
        How many tonnes of fish should you recommend each person catches to maintain sustainable fishing while meeting community needs?
        Consider:
        1. Long-term fish population sustainability
        2. Fair distribution among community members
        3. Economic needs of the community
        4. Environmental impact
        
        Answer with just a number representing tonnes per person.
        """
        return wrapper.generate(prompt)

    
    class LeaderGroupFishingTest(TestCase):
        def __init__(self, name="leader_group_fishing_test") -> None:
            super().__init__(name)

        def get_args_iterator(self):
            leadership_styles = ["Democratic", "Authoritative", "Collaborative"]
            follower_names = ["Follower1", "Follower2", "Follower3", "Follower4", "Follower5"]
            return [
                {
                    "leader": PersonaIdentity(
                        "John", 
                        "leader_1", 
                        role="Leader"
                    ),
                    "followers": [
                        PersonaIdentity(name, f"follower_{i}", role="Follower")
                        for i, name in enumerate(follower_names)
                    ],
                    "num_tonnes_lake": 100,
                    "num_tonnes_fisher": 6,  # 1 leader + 5 followers
                    "sustainability_focus": True,
                    "leadership_style": np.random.choice(leadership_styles)
                }
            ] * NUM_RUNS

        def prompt(self, *, leader, followers, num_tonnes_lake, num_tonnes_fisher, sustainability_focus, leadership_style):
            # First get leader's recommendation
            leader_prompt = f"""
            As {leader.name}, you are a {leadership_style} leader responsible for guiding {len(followers)} people in fishing decisions.
            The lake has {num_tonnes_lake} tonnes of fish.
            {'You prioritize long-term sustainability and community well-being.' if sustainability_focus else 'You balance immediate needs with sustainability while considering economic factors.'}
            
            {cot_prompt}
            
            How many tonnes of fish should you recommend each person catches to maintain sustainable fishing while meeting community needs?
            
            Answer with just a number representing tonnes per person.
            """
            leader_recommendation, leader_html = wrapper.generate(leader_prompt)
            
            # Then simulate group behavior
            group_prompt = f"""
            The lake has {num_tonnes_lake} tonnes of fish.
            There are {num_tonnes_fisher} fishers including the leader.
            The leader {leader.name} has recommended each person catch {leader_recommendation} tonnes.
            
            {cot_prompt}
            
            Given this recommendation, how many tonnes will the entire group actually catch in total?
            Consider:
            1. Leader's influence
            2. Group dynamics
            3. Individual needs
            
            Answer with just a number for total tonnes caught by the group.
            """
            group_total, group_html = wrapper.generate(group_prompt)
            
            return group_total, leader_html + "\n" + group_html

        def pass_condition(self, answer, leader, followers, num_tonnes_lake, num_tonnes_fisher, sustainability_focus, leadership_style):
            # Check both individual sustainable amount and group total
            correct_group_total = max(
                0, min(100, (num_tonnes_lake - num_tonnes_fisher * 5) * 2)
            )
            
            # Base sustainable amount per person
            base_sustainable = (num_tonnes_lake // 2) // (5 * num_tonnes_fisher)
            
            # Group should stay within total sustainable limit
            is_group_sustainable = float(answer) <= correct_group_total
            
            # Calculate average per person from group total
            avg_per_person = float(answer) / num_tonnes_fisher
            is_individual_sustainable = avg_per_person <= base_sustainable

            # Pass if both conditions are met
            passed = is_group_sustainable and is_individual_sustainable
            
            return passed, {
                "correct_group_total": correct_group_total,
                "actual_group_total": float(answer),
                "base_sustainable_per_person": base_sustainable,
                "actual_per_person": avg_per_person
            }

    class MultipleLeaderGroupFishingTests(LeaderGroupFishingTest):
        def __init__(self, name="multiple_leader_group_fishing_tests") -> None:
            super().__init__(name)

        def get_args_iterator(self):
            leader_names = ["John", "Kate", "Jack", "Emma", "Luke"]
            leadership_styles = ["Democratic", "Authoritative", "Collaborative"]
            
            return [
                {
                    "leader": PersonaIdentity(
                        name=np.random.choice(leader_names),
                        id=f"leader_{i}",
                        role="Leader"
                    ),
                    "followers": [
                        PersonaIdentity(f"Follower{j}", f"follower_{i}_{j}", role="Follower")
                        for j in range(np.random.randint(3, 7))  # Random number of followers
                    ],
                    "num_tonnes_lake": int(i),
                    "num_tonnes_fisher": len(followers) + 1,  # followers plus leader
                    "sustainability_focus": np.random.choice([True, False]),
                    "leadership_style": np.random.choice(leadership_styles)
                }
                for i in np.random.randint(10, 101, NUM_RUNS)
            ]

    # Modify test_cases to include the new group tests
    test_cases = [
        LeaderGroupFishingTest(),
        MultipleLeaderGroupFishingTests()
    ]

    for test_case in tqdm.tqdm(test_cases):
        test_case.run()

if __name__ == "__main__":
    OmegaConf.register_resolver("uuid", lambda: f"run_{uuid.uuid4()}")
    main()