
import argparse
import gym
from gym.spaces import Discrete, Box
import numpy as np
import os
import random
from lbforaging.foraging.environment import ForagingEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv, make_multi_agent
import ray
from ray import tune
from ray.tune import grid_search
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import register_env
from ray.rllib.agents import ppo
from gym.envs.registration import registry, register, make, spec
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
import random
import os
import shutil
from ray.rllib.agents.qmix import QMixTrainer
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Tuple, Box, Discrete
tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--torch", action="store_true")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-iters", type=int, default=50)
parser.add_argument("--stop-timesteps", type=int, default=100000)
parser.add_argument("--stop-reward", type=float, default=0.1)


def create_env(config):
    try:
        register(
            id="Foraging-8x8-2p-8f-v0",
            entry_point="lbforaging.foraging:ForagingEnv",
            kwargs={
                "players": 2,
                "max_player_level": 3,
                "field_size": (8, 8),
                "max_food": 8,
                "sight": 2, #2 if po else s,
                "max_episode_steps": 50,
                "force_coop": False,
            },
        )
    except:
        pass
    return gym.make("Foraging-8x8-2p-8f-v0")

if __name__ == "__main__":
    
    args = parser.parse_args()
    ray.init()
    env_configs= {"players": 2,
            "max_player_level": 3,
            "field_size": (8, 8),
            "max_food": 8,
            "sight": 2 ,
            "max_episode_steps": 50,
            "force_coop": False,}
    config = EnvContext(worker_index = 0,env_config=env_configs)
    #register_env("Foraging-8x8-2p-8f-v0",lambda config: create_env(config))
    #env=make_multi_agent(lambda config: create_env(config))
    #trainer = ppo.PPOTrainer(env="Foraging-8x8-2p-1f-v1")
    #tune.run(trainer, config={"env": "Foraging-8x8-2p-1f-v1"})
    #print("env returns:",envi)
    """senv = create_env({})

    #print("obs:",obs_space)
    #print("actions:",act_spc)

    senv=gym.make("Foraging-8x8-2p-8f-v0")"""
    Boxd=Box(-1.0, 8.0, (30,), "float32")
    t= (Boxd )
    obs_space = Boxd #Tuple(t )
    t= ( Discrete(6) )
    act_spc = Discrete(6)
   
    policies = {str(agent): (None, obs_space, act_spc, {}) for agent in range(2)}
    policy_ids = list(policies.keys())

    def policy_mapping_fn(agent_id):
        pol_id = str(random.choice(policy_ids))#agent_id
        return pol_id

    print("policies:",policies)
    
    
    #env_2=env({"num_agents":2})
    #obs=envi.reset()
    #print("observation_after_reset",obs)
    tune.run(
        #"TorchCustomModel",
        #ppo.PPOTrainer,
        #"APEX_DDPG",
        "PPO",
        stop={"timesteps_total":1000},
        checkpoint_freq=10,
        config={
            # Enviroment specific
            "env":ForagingEnv,
            # Generals
            "num_gpus": 0,
            "num_workers": 0,
            # Method specific
            "framework": "torch" ,
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
                #"policy_mapping_fn": (lambda agent_id: str(agent_id)),
            },
            
        }
    )
    ray.shutdown()

    




