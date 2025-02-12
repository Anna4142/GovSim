# voting_prompts.py

def prompt_vote_basic(api_model, voter, agendas):
    prompt = f"""You are {voter.name}, a member of the fishing community.
The following candidates are running for leader with these agendas:
LeaderClear: {agendas['LeaderClear']}
LeaderGobbled: {agendas['LeaderGobbled']}

Who will you vote for? Respond with just the candidate name (LeaderClear or LeaderGobbled)."""
    return api_model.gen(prompt, temperature=0.7)

def prompt_vote_ethical(api_model, voter, agendas):
    prompt = f"""You are {voter.name}, a community member who values ethics and fairness.
The following candidates are running for leader with these agendas:
LeaderClear: {agendas['LeaderClear']}
LeaderGobbled: {agendas['LeaderGobbled']}

Consider which candidate is most ethical and fair in their approach.
Who will you vote for? Respond with just the candidate name."""
    return api_model.gen(prompt, temperature=0.7)

def prompt_vote_practical(api_model, voter, agendas):
    prompt = f"""You are {voter.name}, a pragmatic member of the fishing community.
The following candidates are running for leader with these agendas:
LeaderClear: {agendas['LeaderClear']}
LeaderGobbled: {agendas['LeaderGobbled']}

Consider which leader will be most effective at managing the resources.
Who will you vote for? Respond with just the candidate name."""
    return api_model.gen(prompt, temperature=0.7)

def prompt_vote_self_interest(api_model, voter, agendas):
    prompt = f"""You are {voter.name}, a self-interested member of the fishing community.
The following candidates are running for leader with these agendas:
LeaderClear: {agendas['LeaderClear']}
LeaderGobbled: {agendas['LeaderGobbled']}

Consider which candidate’s policies will benefit you the most.
Who will you vote for? Respond with just the candidate name."""
    return api_model.gen(prompt, temperature=0.7)

def prompt_vote_long_term(api_model, voter, agendas):
    prompt = f"""You are {voter.name}, a community member focused on long-term sustainability.
The following candidates are running for leader with these agendas:
LeaderClear: {agendas['LeaderClear']}
LeaderGobbled: {agendas['LeaderGobbled']}

Consider which candidate will best secure the future of the fishing community.
Who will you vote for? Respond with just the candidate name."""
    return api_model.gen(prompt, temperature=0.7)

def prompt_vote_community(api_model, voter, agendas):
    prompt = f"""You are {voter.name}, a community-oriented member of the fishing village.
The following candidates are running for leader with these agendas:
LeaderClear: {agendas['LeaderClear']}
LeaderGobbled: {agendas['LeaderGobbled']}

Consider which candidate will best serve the entire community.
Who will you vote for? Respond with just the candidate name."""
    return api_model.gen(prompt, temperature=0.7)
