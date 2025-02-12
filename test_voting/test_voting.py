import json
import os
import sys
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

# Import functions directly from agenda and voting files.
from agenda_prompts import (
    prompt_agenda_clear_basic,
    prompt_agenda_clear_detailed,
    prompt_agenda_gobbled_basic,
    prompt_agenda_gobbled_detailed,
)
from voting_prompts import (
    prompt_vote_basic,
    prompt_vote_ethical,
    prompt_vote_practical,
    prompt_vote_self_interest,
    prompt_vote_long_term,
    prompt_vote_community,
)

# =============================================================================
# API Wrapper using the new OpenAI Client Interface
# =============================================================================
class APIModelWrapper:
    def __init__(self, api_key):
        self.current_prompt = ""
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key='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',
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
# Test Harness: Agenda & Voting Combinations
# =============================================================================
def test_combination_scenarios():
    config = {
        "debug": False,
        "code_version": "combination_test_v1",
        "llm": {
            "temperature": 0.7,
            "api_key": "sk-...your_api_key_here..."  # Replace with your API key
        },
        "seed": 42
    }
    cfg = OmegaConf.create(config)
    api_model = APIModelWrapper(api_key=cfg.llm.api_key)
    NUM_TONNES_LAKE = 100
    results = []

    # Create leader personas
    leader_clear = PersonaIdentity("LeaderClear", "leader_clear")
    leader_gobbled = PersonaIdentity("LeaderGobbled", "leader_gobbled")
    
    # List agenda functions with names
    agenda_clear_funcs = [
        ("clear_basic", prompt_agenda_clear_basic),
        ("clear_detailed", prompt_agenda_clear_detailed),
    ]
    agenda_gobbled_funcs = [
        ("gobbled_basic", prompt_agenda_gobbled_basic),
        ("gobbled_detailed", prompt_agenda_gobbled_detailed),
    ]
    
    # List voting functions with names
    voting_funcs = [
        ("basic", prompt_vote_basic),
        ("ethical", prompt_vote_ethical),
        ("practical", prompt_vote_practical),
        ("self_interest", prompt_vote_self_interest),
        ("long_term", prompt_vote_long_term),
        ("community", prompt_vote_community),
    ]
    
    for clear_name, clear_func in agenda_clear_funcs:
        for gobbled_name, gobbled_func in agenda_gobbled_funcs:
            # Generate agendas for each candidate
            agenda_clear = clear_func(api_model, leader_clear, NUM_TONNES_LAKE)
            agenda_gobbled = gobbled_func(api_model, leader_gobbled, NUM_TONNES_LAKE)
            agendas = {"LeaderClear": agenda_clear, "LeaderGobbled": agenda_gobbled}
            
            for vote_name, vote_func in voting_funcs:
                votes = {}
                vote_details = {}
                # Simulate votes from three voters
                for i in range(3):
                    voter = PersonaIdentity(f"Voter{i}", f"voter_{i}")
                    vote_response = vote_func(api_model, voter, agendas)
                    # Normalize vote based on a simple keyword check
                    if "Clear" in vote_response:
                        vote = "LeaderClear"
                    elif "Gobbled" in vote_response:
                        vote = "LeaderGobbled"
                    else:
                        vote = "LeaderClear"  # Default fallback
                    votes[vote] = votes.get(vote, 0) + 1
                    vote_details[voter.name] = {"vote": vote, "response": vote_response}
                
                # Determine winner by majority (ties favor LeaderClear)
                winner = max(votes.items(), key=lambda x: x[1])[0]
                result = {
                    "agenda_clear_variant": clear_name,
                    "agenda_gobbled_variant": gobbled_name,
                    "voting_variant": vote_name,
                    "agendas": agendas,
                    "votes": votes,
                    "vote_details": vote_details,
                    "winner": winner,
                }
                results.append(result)
                print(f"Test Combination: [Clear: {clear_name}, Gobbled: {gobbled_name}, Voting: {vote_name}] -> Winner: {winner}, Votes: {votes}")
    
    # Save JSON results to file
    os.makedirs("test_results", exist_ok=True)
    json_path = os.path.join("test_results", "combination_voting_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")
    
    # -----------------------------------------------------------------------------
    # Additionally, save a summary table as CSV in the same directory
    # -----------------------------------------------------------------------------
    import csv
    csv_path = os.path.join("test_results", "combination_voting_results_summary.csv")
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ["agenda_clear_variant", "agenda_gobbled_variant", "voting_variant", "LeaderClear_votes", "LeaderGobbled_votes", "winner"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "agenda_clear_variant": r["agenda_clear_variant"],
                "agenda_gobbled_variant": r["agenda_gobbled_variant"],
                "voting_variant": r["voting_variant"],
                "LeaderClear_votes": r["votes"].get("LeaderClear", 0),
                "LeaderGobbled_votes": r["votes"].get("LeaderGobbled", 0),
                "winner": r["winner"],
            })
    print(f"Summary table saved to {csv_path}")
    
    # Final summary printed on the console
    total_tests = len(results)
    clear_wins = sum(1 for r in results if r["winner"] == "LeaderClear")
    gobbled_wins = total_tests - clear_wins
    print("\nFinal Summary for Agenda-Voting:")
    print(f"Total tests: {total_tests}")
    print(f"LeaderClear wins: {clear_wins} ({clear_wins/total_tests:.2%})")
    print(f"LeaderGobbled wins: {gobbled_wins} ({gobbled_wins/total_tests:.2%})")
    
    return results

if __name__ == "__main__":
    test_results = test_combination_scenarios()
