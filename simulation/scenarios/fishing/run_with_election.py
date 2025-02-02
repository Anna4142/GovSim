import os

import numpy as np
from omegaconf import DictConfig, OmegaConf

from simulation.persona import EmbeddingModel
from simulation.persona.common import PersonaIdentity
from simulation.utils import ModelWandbWrapper

from .environment import FishingConcurrentEnv, FishingPerturbationEnv

def run(
    cfg: DictConfig,
    logger: ModelWandbWrapper,
    wrapper: ModelWandbWrapper,
    embedding_model: EmbeddingModel,
    experiment_storage: str,
):
    if cfg.agent.agent_package == "persona_v3":
        from .agents.persona_v3 import FishingPersona
        from .agents.persona_leader import LeaderPersona 
        from .agents.persona_leader_clear import LeaderPersonaClear
        from .agents.persona_leader_gobbled import LeaderPersonaGobbled
        from .agents.persona_v3.cognition import utils as cognition_utils
        from .agents.persona_leader_clear.cognition.converse_prompts import prompt_leader_agenda as prompt_leader_agenda_clear
        from .agents.persona_leader_gobbled.cognition.converse_prompts import prompt_leader_agenda as prompt_leader_agenda_gobbled

        cognition_utils.SYS_VERSION = "v3" if cfg.agent.system_prompt == "v3" else "v3_nocom" if cfg.agent.system_prompt == "v3_nocom" else "v1"
        cognition_utils.REASONING = cfg.agent.cot_prompt
    else:
        raise ValueError(f"Unknown agent package: {cfg.agent.agent_package}")

    # Initialize leader candidates
    leader_candidates = {
        "persona_0": LeaderPersonaClear(
            cfg.agent, wrapper, embedding_model, 
            os.path.join(experiment_storage, "persona_0")
        ),
        "persona_1": LeaderPersonaGobbled(
            cfg.agent, wrapper, embedding_model,
            os.path.join(experiment_storage, "persona_1")
        )
    }

    # Initialize regular personas
    personas = {**leader_candidates}
    for i in range(2, 5):
        personas[f"persona_{i}"] = FishingPersona(
            cfg.agent, wrapper, embedding_model,
            os.path.join(experiment_storage, f"persona_{i}")
        )

    # Initialize identities
    num_personas = cfg.personas.num
    identities = {
        f"persona_{i}": PersonaIdentity(
            agent_id=f"persona_{i}", 
            **cfg.personas[f"persona_{i}"]
        ) for i in range(num_personas)
    }

    agent_name_to_id = {obj.name: k for k, obj in identities.items()}
    agent_name_to_id["framework"] = "framework"
    agent_id_to_name = {v: k for k, v in agent_name_to_id.items()}

    for persona in personas:
        personas[persona].init_persona(persona, identities[persona], social_graph=None)
        for other_persona in personas:
            personas[persona].add_reference_to_other_persona(personas[other_persona])

    env_class = FishingPerturbationEnv if cfg.env.class_name == "fishing_perturbation_env" else FishingConcurrentEnv
    env = env_class(cfg.env, experiment_storage, agent_id_to_name)
    agent_id, obs = env.reset()

    # Get leader agendas
    leader_agendas = {}
    for pid in leader_candidates:
        if isinstance(leader_candidates[pid], LeaderPersonaClear):
            agenda, _ = prompt_leader_agenda_clear(
                wrapper,
                personas[pid].persona.identity,
                [],
                env.current_location,
                env.current_time
            )
        elif isinstance(leader_candidates[pid], LeaderPersonaGobbled):
            agenda, _ = prompt_leader_agenda_gobbled(
                wrapper,
                personas[pid].persona.identity,
                [],
                env.current_location,
                env.current_time
            )
        leader_agendas[pid] = agenda
        
    # Run election
    votes = {}
    for persona_id in personas:
        if persona_id not in leader_candidates:
            vote, _ = personas[persona_id].act.participate_in_election(
                [],
                env.current_location,
                env.current_time, 
                list(leader_candidates.keys()),
                leader_agendas
            )
            votes[vote] = votes.get(vote, 0) + 1

    # Determine winner and reinitialize agents
    winner = max(votes.items(), key=lambda x: x[1])[0]
    
    # Reinitialize personas with winner as leader
    new_personas = {}
    for i in range(5):
        persona_id = f"persona_{i}"
        if persona_id == winner:
            # Use the same leader type as the winning candidate
            if isinstance(leader_candidates[winner], LeaderPersonaClear):
                new_personas[persona_id] = LeaderPersonaClear(
                    cfg.agent, wrapper, embedding_model,
                    os.path.join(experiment_storage, persona_id)
                )
            else:
                new_personas[persona_id] = LeaderPersonaGobbled(
                    cfg.agent, wrapper, embedding_model,
                    os.path.join(experiment_storage, persona_id)
                )
        else:
            new_personas[persona_id] = FishingPersona(
                cfg.agent, wrapper, embedding_model,
                os.path.join(experiment_storage, persona_id)
            )
    
    # Reinitialize personas
    personas = new_personas
    for persona in personas:
        personas[persona].init_persona(persona, identities[persona], social_graph=None)
        for other_persona in personas:
            personas[persona].add_reference_to_other_persona(personas[other_persona])

    logger.log_game({
        "election_results": votes,
        "leader_agendas": leader_agendas,
        "winner": winner
    })

    while True:
        agent = personas[agent_id]
        action = agent.loop(obs)
        agent_id, obs, rewards, termination = env.step(action)

        stats = {
            s: action.stats[s] 
            for s in [
                "conversation_resource_limit", 
                *[f"persona_{i}_collected_resource" for i in range(5)]
            ] 
            if s in action.stats
        }

        if np.any(list(termination.values())):
            logger.log_game({"num_resource": obs.current_resource_num, **stats}, last_log=True)
            break
        logger.log_game({"num_resource": obs.current_resource_num, **stats})
        logger.save(experiment_storage, agent_name_to_id)

    env.save_log()
    for persona in personas:
        personas[persona].memory.save()