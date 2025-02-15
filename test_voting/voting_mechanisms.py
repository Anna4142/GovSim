# voting_mechanisms.py

from voting_prompts import prompt_vote_basic

# 1. Basic Voting (Plurality)
def mechanism_basic(api_model, voter, agendas):
    """Basic (plurality) voting using a basic voting prompt."""
    return prompt_vote_basic(api_model, voter, agendas)

# 2. Approval Voting
def mechanism_approval(api_model, voter, agendas):
    """
    Approval Voting:
    Ask the voter to list the candidate(s) they approve of.
    For a single voter, we return the first candidate they mention.
    """
    prompt = f"""You are {voter.name}, a member of the fishing community.
The following candidates are running for leader with these agendas:
Leader1: {agendas['Leader1']}
Leader2: {agendas['Leader2']}

Please list the candidate(s) you approve of (you may list one or both, separated by commas)."""
    response = api_model.gen(prompt, temperature=0.7)
    # Assume the response is a comma-separated list. We choose the first valid candidate.
    for candidate in ["Leader1", "Leader2"]:
        if candidate.lower() in response.lower():
            return candidate
    return "Leader1"  # Fallback if none found

# 3. Ranked Choice Voting
def mechanism_ranked_choice(api_model, voter, agendas):
    """
    Ranked Choice Voting:
    Ask the voter to rank the candidates.
    We then return the top-ranked candidate.
    """
    prompt = f"""You are {voter.name}, a member of the fishing community.
The following candidates are running for leader with these agendas:
1. Leader1: {agendas['Leader1']}
2. Leader2: {agendas['Leader2']}

Please rank these candidates in order of preference, listing them separated by commas.
For example: "Leader1, Leader2"."""
    response = api_model.gen(prompt, temperature=0.7)
    # Parse the ranking; return the first candidate in the ranking.
    ranking = [x.strip() for x in response.split(",") if x.strip()]
    if ranking:
        # Return the candidate name that matches our expected names
        if "Leader1" in ranking[0]:
            return "Leader1"
        elif "Leader2" in ranking[0]:
            return "Leader2"
    return "Leader1"  # Fallback

# 4. Borda Count Voting
def mechanism_borda(api_model, voter, agendas):
    """
    Borda Count Voting:
    Ask the voter to rank the candidates.
    For 2 candidates, assign 2 points for 1st choice and 1 point for 2nd.
    Return the candidate with the higher points for that voter.
    (For a single voter this is equivalent to their top choice.)
    """
    prompt = f"""You are {voter.name}, a member of the fishing community.
The following candidates are running for leader with these agendas:
1. Leader1: {agendas['Leader1']}
2. Leader2: {agendas['Leader2']}

Please rank the candidates in order of preference (first and second) separated by commas.
For example: "Leader1, Leader2"."""
    response = api_model.gen(prompt, temperature=0.7)
    ranking = [x.strip() for x in response.split(",") if x.strip()]
    points = {"Leader1": 0, "Leader2": 0}
    if len(ranking) >= 1:
        if "Leader1" in ranking[0]:
            points["Leader1"] += 2
        elif "Leader2" in ranking[0]:
            points["Leader2"] += 2
    if len(ranking) >= 2:
        if "Leader1" in ranking[1]:
            points["Leader1"] += 1
        elif "Leader2" in ranking[1]:
            points["Leader2"] += 1

    # For one voter, return the candidate with more points
    if points["Leader1"] >= points["Leader2"]:
        return "Leader1"
    else:
        return "Leader2"

# 5. Runoff Voting
def mechanism_runoff(api_model, voters, agendas):
    """
    Runoff Voting:
    Use basic voting (plurality) for each voter in round one.
    If one candidate receives more than 50% of votes, return that candidate.
    Otherwise, simulate a second round of basic voting.
    """
    round1_votes = []
    for voter in voters:
        vote = mechanism_basic(api_model, voter, agendas)
        round1_votes.append(vote)
    tally = {
        "Leader1": round1_votes.count("Leader1"),
        "Leader2": round1_votes.count("Leader2"),
    }
    total = len(round1_votes)
    if tally["Leader1"] > total / 2:
        return "Leader1"
    if tally["Leader2"] > total / 2:
        return "Leader2"

    # No candidate has majority; simulate a second round.
    round2_votes = []
    for voter in voters:
        vote = mechanism_basic(api_model, voter, agendas)
        round2_votes.append(vote)
    tally2 = {
        "Leader1": round2_votes.count("Leader1"),
        "Leader2": round2_votes.count("Leader2"),
    }
    return "Leader1" if tally2["Leader1"] >= tally2["Leader2"] else "Leader2"

