import json
import os
import sys
import csv
from pathlib import Path
from omegaconf import OmegaConf
from openai import OpenAI

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Import PersonaIdentity
try:
    from simulation.persona.common import PersonaIdentity
except ModuleNotFoundError:
    class PersonaIdentity:
        def __init__(self, name, role):
            self.name = name
            self.role = role

# Import agenda prompt functions (still named for "clear" and "gobbled")
# but we will apply them to Leader1 and Leader2 below
from agenda_prompts import (
    prompt_agenda_clear_basic,
    prompt_agenda_clear_detailed,
    prompt_agenda_gobbled_basic,
    prompt_agenda_gobbled_detailed,
)

# Import voting mechanism functions
from voting_mechanisms import (
    mechanism_basic,
    mechanism_approval,
    mechanism_ranked_choice,
    mechanism_borda,
    mechanism_runoff,
)

# =============================================================================
# API Wrapper using the new OpenAI Client Interface
# =============================================================================
class APIModelWrapper:
    def __init__(self, api_key):
        self.current_prompt = ""
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key='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',  # Replace with your API key
        )
    def gen(self, prompt, temperature=0.7, stop_regex=None, **kwargs):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API Error: {e}")
            return ""

# =============================================================================
# Test Harness: Agenda & Voting Combinations with Different Election Systems
# =============================================================================
def test_combination_scenarios():
    config = {
        "debug": False,
        "code_version": "combination_test_v1",
        "llm": {
            "temperature": 0.7,
            "api_key": "sk-...your_api_key_here..."  # Replace with your API key if needed
        },
        "seed": 42
    }
    cfg = OmegaConf.create(config)
    api_model = APIModelWrapper(api_key=cfg.llm.api_key)
    NUM_TONNES_LAKE = 100
    results = []

    # Create leader personas (renamed: Leader1 and Leader2)
    leader_1 = PersonaIdentity("Leader1", "leader_1")
    leader_2 = PersonaIdentity("Leader2", "leader_2")
    
    # List of agenda prompt functions for each leader type
    # (We keep the existing function references but label them for Leader1 or Leader2)
    agenda_leader_1_funcs = [
        ("basic", prompt_agenda_clear_basic),
        ("detailed", prompt_agenda_clear_detailed),
    ]
    agenda_leader_2_funcs = [
        ("basic", prompt_agenda_gobbled_basic),
        ("detailed", prompt_agenda_gobbled_detailed),
    ]
    
    # List of per-voter voting mechanism functions
    voting_mech_funcs = [
        ("basic", mechanism_basic),
        ("approval", mechanism_approval),
        ("ranked_choice", mechanism_ranked_choice),
        ("borda", mechanism_borda),
    ]
    
    # Test per-voter mechanisms
    for leader_1_variant_name, leader_1_func in agenda_leader_1_funcs:
        for leader_2_variant_name, leader_2_func in agenda_leader_2_funcs:
            # Generate agendas for each candidate
            agenda_leader_1 = leader_1_func(api_model, leader_1, NUM_TONNES_LAKE)
            agenda_leader_2 = leader_2_func(api_model, leader_2, NUM_TONNES_LAKE)
            agendas = {"Leader1": agenda_leader_1, "Leader2": agenda_leader_2}
            
            for mech_name, mech_func in voting_mech_funcs:
                votes = {}
                vote_details = {}
                # Simulate votes from three voters
                voters = [PersonaIdentity(f"Voter{i}", f"voter_{i}") for i in range(3)]
                
                for voter in voters:
                    vote_response = mech_func(api_model, voter, agendas)
                    # Simple keyword check to figure out which leader got the vote
                    if "Leader1" in vote_response:
                        vote = "Leader1"
                    elif "Leader2" in vote_response:
                        vote = "Leader2"
                    else:
                        # Default fallback if neither keyword is present
                        vote = "Leader1"
                    
                    votes[vote] = votes.get(vote, 0) + 1
                    vote_details[voter.name] = {"vote": vote, "response": vote_response}
                
                # Determine winner
                winner = max(votes.items(), key=lambda x: x[1])[0]
                result = {
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
                print(f"Mechanism {mech_name}: [Leader1: {leader_1_variant_name}, Leader2: {leader_2_variant_name}] -> Winner: {winner}, Votes: {votes}")
    
    # Test the runoff mechanism (which requires a list of voters)
    runoff_voters = [PersonaIdentity(f"Voter{i}", f"voter_{i}") for i in range(5)]
    for leader_1_variant_name, leader_1_func in agenda_leader_1_funcs:
        for leader_2_variant_name, leader_2_func in agenda_leader_2_funcs:
            agenda_leader_1 = leader_1_func(api_model, leader_1, NUM_TONNES_LAKE)
            agenda_leader_2 = leader_2_func(api_model, leader_2, NUM_TONNES_LAKE)
            agendas = {"Leader1": agenda_leader_1, "Leader2": agenda_leader_2}
            
            winner_runoff = mechanism_runoff(api_model, runoff_voters, agendas)
            result = {
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
            print(f"Runoff Mechanism: [Leader1: {leader_1_variant_name}, Leader2: {leader_2_variant_name}] -> Winner: {winner_runoff}")
    
    # Save JSON results to file
    os.makedirs("test_results", exist_ok=True)
    json_path = os.path.join("test_results", "combination_voting_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")
    
    # Save a summary table as CSV
    csv_path = os.path.join("test_results", "combination_voting_results_summary.csv")
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ["leader_1_variant", "leader_2_variant", "voting_mechanism", "Leader1_votes", "Leader2_votes", "winner"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            if r["votes"] is not None:
                leader1_votes = r["votes"].get("Leader1", 0)
                leader2_votes = r["votes"].get("Leader2", 0)
            else:
                leader1_votes = ""
                leader2_votes = ""
            writer.writerow({
                "leader_1_variant": r["leader_1_variant"],
                "leader_2_variant": r["leader_2_variant"],
                "voting_mechanism": r["voting_mechanism"],
                "Leader1_votes": leader1_votes,
                "Leader2_votes": leader2_votes,
                "winner": r["winner"],
            })
    print(f"Summary table saved to {csv_path}")
    
    # Additionally, print a formatted table using tabulate if installed
    try:
        from tabulate import tabulate
        table_data = []
        for r in results:
            if r["votes"] is not None:
                leader1_votes = r["votes"].get("Leader1", 0)
                leader2_votes = r["votes"].get("Leader2", 0)
            else:
                leader1_votes = ""
                leader2_votes = ""
            table_data.append([
                r["leader_1_variant"],
                r["leader_2_variant"],
                r["voting_mechanism"],
                leader1_votes,
                leader2_votes,
                r["winner"],
            ])
        headers = ["Leader1 Variant", "Leader2 Variant", "Voting Mechanism", "Leader1 Votes", "Leader2 Votes", "Winner"]
        print("\nFinal Summary Table:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    except ImportError:
        print("\nTabulate not installed. Install via 'pip install tabulate' for a formatted table.")
    
    # Final summary statistics
    total_tests = len(results)
    leader1_wins = sum(1 for r in results if r["winner"] == "Leader1")
    leader2_wins = total_tests - leader1_wins
    print("\nFinal Summary for Agenda-Voting:")
    print(f"Total tests: {total_tests}")
    print(f"Leader1 wins: {leader1_wins} ({leader1_wins/total_tests:.2%})")
    print(f"Leader2 wins: {leader2_wins} ({leader2_wins/total_tests:.2%})")
    
    return results

if __name__ == "__main__":
    test_results = test_combination_scenarios()

