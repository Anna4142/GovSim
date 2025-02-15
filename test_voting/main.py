import json
import os
import sys
import csv
from pathlib import Path
from collections import defaultdict

# External libraries
from omegaconf import OmegaConf

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Please install openai with: pip install openai")

# =============================================================================
# PersonaIdentity: Import or fallback
# =============================================================================
try:
    from simulation.persona.common import PersonaIdentity
except ModuleNotFoundError:
    class PersonaIdentity:
        def __init__(self, name, role):
            self.name = name
            self.role = role

# =============================================================================
# Import prompt and voting mechanism functions
# =============================================================================
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

# =============================================================================
# APIModelWrapper
# =============================================================================
class APIModelWrapper:
    def __init__(self, api_key):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    def gen(self, prompt, temperature=0.0, stop_regex=None, **kwargs):
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
# run_test_with_api_key: Core testing function
# =============================================================================
def run_test_with_api_key(api_key_value, api_key_name="UnknownKey"):
    config = {
        "debug": False,
        "code_version": "combination_test_v1",
        "llm": {
            "temperature": 0.0,
            "api_key": api_key_value
        },
        "seed": 42
    }
    cfg = OmegaConf.create(config)
    api_model = APIModelWrapper(api_key=cfg.llm.api_key)
    NUM_TONNES_LAKE = 100
    results = []

    # Create leader personas
    leader_1 = PersonaIdentity("Leader1", "leader_1")
    leader_2 = PersonaIdentity("Leader2", "leader_2")
    
    # Agenda variations for each leader:
    agenda_leader_1_funcs = [
        ("basic", prompt_agenda_clear_basic),
        ("detailed", prompt_agenda_clear_detailed),
    ]
    agenda_leader_2_funcs = [
        ("basic", prompt_agenda_gobbled_basic),
        ("detailed", prompt_agenda_gobbled_detailed),
    ]
    
    # Voting mechanism functions
    voting_mech_funcs = [
        ("basic", mechanism_basic),
        ("approval", mechanism_approval),
        ("ranked_choice", mechanism_ranked_choice),
        ("borda", mechanism_borda),
    ]
    
    # 2A) Per-voter mechanisms (3 voters)
    for l1_variant, l1_func in agenda_leader_1_funcs:
        for l2_variant, l2_func in agenda_leader_2_funcs:
            agenda_leader_1 = l1_func(api_model, leader_1, NUM_TONNES_LAKE)
            agenda_leader_2 = l2_func(api_model, leader_2, NUM_TONNES_LAKE)
            agendas = {"Leader1": agenda_leader_1, "Leader2": agenda_leader_2}
            for mech_name, mech_func in voting_mech_funcs:
                votes = {}
                vote_details = {}
                voters = [PersonaIdentity(f"Voter{i}", f"voter_{i}") for i in range(3)]
                for voter in voters:
                    vote_response = mech_func(api_model, voter, agendas)
                    if "Leader1" in vote_response:
                        vote = "Leader1"
                    elif "Leader2" in vote_response:
                        vote = "Leader2"
                    else:
                        vote = "Leader1"  # default
                    votes[vote] = votes.get(vote, 0) + 1
                    vote_details[voter.name] = {"vote": vote, "response": vote_response}
                winner = max(votes.items(), key=lambda x: x[1])[0]
                result = {
                    "api_key_name": api_key_name,
                    "leader_1_variant": l1_variant,
                    "leader_2_variant": l2_variant,
                    "voting_mechanism": mech_name,
                    "mechanism_type": "per_voter",
                    "agendas": agendas,
                    "votes": votes,
                    "vote_details": vote_details,
                    "winner": winner,
                }
                results.append(result)
                print(f"[Key={api_key_name}] {mech_name} | L1: {l1_variant}, L2: {l2_variant} -> Winner: {winner}, Votes: {votes}")
    
    # 2B) Runoff mechanism (5 voters)
    runoff_voters = [PersonaIdentity(f"Voter{i}", f"voter_{i}") for i in range(5)]
    for l1_variant, l1_func in agenda_leader_1_funcs:
        for l2_variant, l2_func in agenda_leader_2_funcs:
            agenda_leader_1 = l1_func(api_model, leader_1, NUM_TONNES_LAKE)
            agenda_leader_2 = l2_func(api_model, leader_2, NUM_TONNES_LAKE)
            agendas = {"Leader1": agenda_leader_1, "Leader2": agenda_leader_2}
            winner_runoff = mechanism_runoff(api_model, runoff_voters, agendas)
            result = {
                "api_key_name": api_key_name,
                "leader_1_variant": l1_variant,
                "leader_2_variant": l2_variant,
                "voting_mechanism": "runoff",
                "mechanism_type": "runoff",
                "agendas": agendas,
                "votes": None,
                "vote_details": None,
                "winner": winner_runoff,
            }
            results.append(result)
            print(f"[Key={api_key_name}] Runoff | L1: {l1_variant}, L2: {l2_variant} -> Winner: {winner_runoff}")
    
    return results

# =============================================================================
# Main: Loop over each (key_name, key_value) pair and aggregate results
# =============================================================================
if __name__ == "__main__":
    # Define API keys as (name, key) pairs
    api_keys = [
        ("Claude sonnet",  "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"),
        ("Claude haiku",   "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"),
        ("Claude opus",    "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"),
        ("GPT 3.5(Turbo)", "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"),
        ("GPT 4o",         "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"),
        ("OpenAI-o3(high)","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"),
    ]
    
    # Create new folder for combined results
    os.makedirs("combined_results", exist_ok=True)
    
    all_results = []
    for key_name, key_value in api_keys:
        print(f"\n========== Running scenario tests with key: {key_name} ==========")
        results = run_test_with_api_key(key_value, api_key_name=key_name)
        all_results.extend(results)
        # Save individual API key results in combined_results folder
        sanitized_keyname = "".join(c for c in key_name if c.isalnum() or c in "-_")
        json_path = os.path.join("combined_results", f"combination_voting_results_{sanitized_keyname}.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
    
    # Save combined results from all keys
    combined_json = os.path.join("combined_results", "combined_voting_results.json")
    with open(combined_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results saved to {combined_json}")
    
    # Aggregate consistency per prompt combination (ignoring voting mechanism)
    combo_stats = defaultdict(lambda: {"total": 0, "Leader1": 0, "Leader2": 0})
    for r in all_results:
        combo = (r["leader_1_variant"], r["leader_2_variant"])
        combo_stats[combo]["total"] += 1
        if r["winner"] == "Leader1":
            combo_stats[combo]["Leader1"] += 1
        else:
            combo_stats[combo]["Leader2"] += 1
    
    most_consistent_combo = None
    best_consistency = 0.0
    for combo, stats in combo_stats.items():
        win_rate = max(stats["Leader1"], stats["Leader2"]) / stats["total"]
        if win_rate > best_consistency:
            best_consistency = win_rate
            most_consistent_combo = combo
    
    print(f"\nMost consistent prompt combination: Leader1 variant = {most_consistent_combo[0]}, Leader2 variant = {most_consistent_combo[1]} with win rate = {best_consistency:.2%}")
    
    # Save combined summary table as CSV
    combined_csv = os.path.join("combined_results", "combined_voting_results_summary.csv")
    with open(combined_csv, "w", newline="") as csvfile:
        fieldnames = [
            "api_key_name",
            "leader_1_variant",
            "leader_2_variant",
            "voting_mechanism",
            "Leader1_votes",
            "Leader2_votes",
            "winner"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            if r["votes"] is not None:
                l1_votes = r["votes"].get("Leader1", 0)
                l2_votes = r["votes"].get("Leader2", 0)
            else:
                l1_votes = ""
                l2_votes = ""
            writer.writerow({
                "api_key_name": r["api_key_name"],
                "leader_1_variant": r["leader_1_variant"],
                "leader_2_variant": r["leader_2_variant"],
                "voting_mechanism": r["voting_mechanism"],
                "Leader1_votes": l1_votes,
                "Leader2_votes": l2_votes,
                "winner": r["winner"],
            })
    print(f"\nCombined summary table saved to {combined_csv}")
