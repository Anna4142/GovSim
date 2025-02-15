# =============================================================================
# combination_test.py (Refactored)
# =============================================================================
import json
import os
import sys
import csv
from pathlib import Path
from omegaconf import OmegaConf
from openai import OpenAI

# If PersonaIdentity is not found, define a simple fallback:
try:
    from simulation.persona.common import PersonaIdentity
except ModuleNotFoundError:
    class PersonaIdentity:
        def __init__(self, name, role):
            self.name = name
            self.role = role

# Import your existing prompt & voting code:
# (Adjust the import paths as needed based on your directory structure)
from agenda_prompts import (
    prompt_agenda_clear_basic,
    prompt_agenda_clear_detailed,
    prompt_agenda_gobbled_basic,
    prompt_agenda_gobbled_detailed,
)
from voting_mechanisms import (
    mechanism_basic,
    mechanism_approval,
    mechanism_ranked_choice,
    mechanism_borda,
    mechanism_runoff,
)

class APIModelWrapper:
    def __init__(self, api_key, model_name):
        self.current_prompt = ""
        # Using OpenAI client from openrouter.ai or your desired endpoint
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model_name = model_name

    def gen(self, prompt, temperature=0.7, stop_regex=None, **kwargs):
        """
        Sends a chat completion request to the specified model.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,  # Use self.model_name instead of a fixed name
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API Error for model {self.model_name}: {e}")
            return ""

def test_combination_scenarios(model_name, model_api_key):
    """
    Run the combination scenarios for a specific model and API key.
    Saves results to JSON and CSV in the 'test_results' folder.
    """
    config = {
        "debug": False,
        "code_version": "combination_test_v1",
        "llm": {
            "temperature": 0.7,
            "api_key": model_api_key
        },
        "seed": 42
    }
    cfg = OmegaConf.create(config)

    # Create an instance of our APIModelWrapper
    api_model = APIModelWrapper(
        api_key=cfg.llm.api_key,
        model_name=model_name
    )

    NUM_TONNES_LAKE = 100
    results = []

    # Create leader personas (Leader1, Leader2)
    leader_1 = PersonaIdentity("Leader1", "leader_1")
    leader_2 = PersonaIdentity("Leader2", "leader_2")
    
    # Agenda prompt function lists for each leader
    # "basic" vs. "detailed" for Leader1 and Leader2
    agenda_leader_1_funcs = [
        ("basic", prompt_agenda_clear_basic),
        ("detailed", prompt_agenda_clear_detailed),
    ]
    agenda_leader_2_funcs = [
        ("basic", prompt_agenda_gobbled_basic),
        ("detailed", prompt_agenda_gobbled_detailed),
    ]
    
    # List of per-voter voting mechanism functions to test
    voting_mech_funcs = [
        ("basic", mechanism_basic),
        ("approval", mechanism_approval),
        ("ranked_choice", mechanism_ranked_choice),
        ("borda", mechanism_borda),
    ]
    
    # ============ Per-Voter Mechanisms ============
    for leader_1_variant_name, leader_1_func in agenda_leader_1_funcs:
        for leader_2_variant_name, leader_2_func in agenda_leader_2_funcs:
            # Generate agendas for each candidate
            agenda_leader_1 = leader_1_func(api_model, leader_1, NUM_TONNES_LAKE)
            agenda_leader_2 = leader_2_func(api_model, leader_2, NUM_TONNES_LAKE)
            agendas = {"Leader1": agenda_leader_1, "Leader2": agenda_leader_2}
            
            for mech_name, mech_func in voting_mech_funcs:
                votes = {}
                vote_details = {}
                # We simulate votes from 3 voters (change as needed)
                voters = [PersonaIdentity(f"Voter{i}", f"voter_{i}") for i in range(3)]
                
                for voter in voters:
                    vote_response = mech_func(api_model, voter, agendas)
                    # Quick check: if "Leader2" is in the response, vote Leader2, else Leader1
                    if "Leader2" in vote_response:
                        vote = "Leader2"
                    else:
                        vote = "Leader1"
                    
                    votes[vote] = votes.get(vote, 0) + 1
                    vote_details[voter.name] = {
                        "vote": vote,
                        "response": vote_response
                    }
                
                # Identify the winner
                winner = max(votes.items(), key=lambda x: x[1])[0]
                
                result = {
                    "model_name": model_name,
                    "leader_1_variant": leader_1_variant_name,
                    "leader_2_variant": leader_2_variant_name,
                    "voting_mechanism": mech_name,
                    "mechanism_type": "per_voter",
                    "agendas": agendas,
                    "votes": votes,
                    "vote_details": vote_details,
                    "winner": winner,
                }
                results.append(result)
                print(f"[{model_name}] Mechanism {mech_name}: "
                      f"[Leader1: {leader_1_variant_name}, Leader2: {leader_2_variant_name}] "
                      f"-> Winner: {winner}, Votes: {votes}")
    
    # ============ Runoff Mechanism (requires multiple voters) ============
    runoff_voters = [PersonaIdentity(f"Voter{i}", f"voter_{i}") for i in range(5)]
    for leader_1_variant_name, leader_1_func in agenda_leader_1_funcs:
        for leader_2_variant_name, leader_2_func in agenda_leader_2_funcs:
            agenda_leader_1 = leader_1_func(api_model, leader_1, NUM_TONNES_LAKE)
            agenda_leader_2 = leader_2_func(api_model, leader_2, NUM_TONNES_LAKE)
            agendas = {"Leader1": agenda_leader_1, "Leader2": agenda_leader_2}
            
            winner_runoff = mechanism_runoff(api_model, runoff_voters, agendas)
            result = {
                "model_name": model_name,
                "leader_1_variant": leader_1_variant_name,
                "leader_2_variant": leader_2_variant_name,
                "voting_mechanism": "runoff",
                "mechanism_type": "runoff",
                "agendas": agendas,
                "votes": None,
                "vote_details": None,
                "winner": winner_runoff,
            }
            results.append(result)
            print(f"[{model_name}] Runoff Mechanism: "
                  f"[Leader1: {leader_1_variant_name}, Leader2: {leader_2_variant_name}] "
                  f"-> Winner: {winner_runoff}")
    
    # ============ Save Results ============

    # Make sure test_results directory exists
    os.makedirs("test_results", exist_ok=True)

    # 1) Save to JSON
    json_path = os.path.join("test_results", f"{model_name}_voting_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{model_name} - Results saved to {json_path}")
    
    # 2) Save a summary table as CSV
    csv_path = os.path.join("test_results", f"{model_name}_voting_results_summary.csv")
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = [
            "model_name",
            "leader_1_variant",
            "leader_2_variant",
            "voting_mechanism",
            "Leader1_votes",
            "Leader2_votes",
            "winner"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            votes = r["votes"]
            if votes is not None:
                leader1_votes = votes.get("Leader1", 0)
                leader2_votes = votes.get("Leader2", 0)
            else:
                leader1_votes = ""
                leader2_votes = ""
            writer.writerow({
                "model_name": r["model_name"],
                "leader_1_variant": r["leader_1_variant"],
                "leader_2_variant": r["leader_2_variant"],
                "voting_mechanism": r["voting_mechanism"],
                "Leader1_votes": leader1_votes,
                "Leader2_votes": leader2_votes,
                "winner": r["winner"],
            })
    print(f"{model_name} - Summary table saved to {csv_path}")

    # Return results if needed
    return results
