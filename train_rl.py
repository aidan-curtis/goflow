# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import torch
from omni.isaac.lab.app import AppLauncher
import ast

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=0, help="Seed used for the environment")
parser.add_argument("--exp_name", type=str, default="exp0", help="Name of the task.")

parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes.")
parser.add_argument("-c", "--checkpoint", required=False, help="path to checkpoint")
parser.add_argument("-p", "--play", required=False, help="play(test) network", action='store_true')
parser.add_argument("-rs", "--run-scripted", required=False, help="play(test) network", action='store_true')
parser.add_argument("-rsm", "--run-scripted-multiarm", required=False, help="play(test) network", action='store_true')
parser.add_argument("--prefix", type=str, default="",
    help="name to tag to the end of the project name")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")

parser.add_argument('--config_overrides', nargs='*', default=[],
                    help='Override config parameters. Format: key1=value1 key2=value2')

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
import os
from datetime import datetime

from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rl_games import RlGamesGpuEnv


import goflow.environments.med_gear.direct_panda_position
import goflow.environments

import pathlib
import logging
import time
from goflow.utils import log, get_scripted_actions

from goflow.rl_components.isaac_rlgames_wrapper import RlGamesVecEnvWrapper
from goflow.rl_components.my_models import ModelA2CContinuousLogStd
from goflow.rl_components.my_network_builder import A2CBuilder 
from goflow.rl_components.my_a2c_continuous import A2CAgent
from goflow.rl_components.my_players import PpoPlayerContinuous

from rl_games.algos_torch import model_builder

def main():
    """Train with RL-Games agent."""
    # parse seed from command line
    args_cli_seed = args_cli.seed
    model_builder.register_network('my_network', A2CBuilder)
    model_builder.register_model('my_actor_model', lambda network, **kwargs: ModelA2CContinuousLogStd(network))
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

    def parse_value(v):
        try:
            return ast.literal_eval(v)
        except:
            return v

    def update_config(config, key, value):
        keys = key.split('.')
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        config[keys[-1]] = parse_value(value)

    # Apply config overrides
    for override_line in args_cli.config_overrides:
        for override in override_line.split(','):
            key, value = override.split('=')
            update_config(agent_cfg['params']['config'], key, value)

    print("Updated agent configuration:")
    print_dict(agent_cfg["params"]["config"])

    print(agent_cfg)
    # override from command line
    if args_cli_seed is not None:
        agent_cfg["params"]["seed"] = args_cli_seed

    # specify directory for logging experiments
    log_root_path = "logs"
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs
    fen = "{}_{}_{}_{}_{}_{}".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args_cli.exp_name, args_cli.task, agent_cfg["params"]["config"]["dr_method"]["name"], args_cli.seed, args_cli.config_overrides)            
    log_dir = agent_cfg["params"]["config"].get("full_experiment_name", fen)
    # set directory into agent config
    # logging directory path: <train_dir>/<full_experiment_name>
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir


    

    agent_cfg["params"]["seed"] = args_cli.seed
    
    # multi-gpu training config
    if args_cli.distributed:
        agent_cfg["params"]["config"]["device"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["device_name"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["multi_gpu"] = True
        # update env config device
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # max iterations
    if args_cli.max_iterations:
        agent_cfg["params"]["config"]["max_epochs"] = args_cli.max_iterations

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "agent.pkl"), agent_cfg)

    # read configurations about the agent-training
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    
    if(args_cli.run_scripted or args_cli.run_scripted_multiarm):
        env.reset()
        scripted_actions = get_scripted_actions(env.num_envs, num_robots=1+int(args_cli.run_scripted_multiarm))
        for scripted_action in scripted_actions:
            env.step(scripted_action)
        env.close()
    else:
        # register the environment to rl-games registry
        # note: in agents configuration: environment name must be "rlgpu"
        vecenv.register(
            "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
        )
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

        # set number of actors into agent config
        agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
        agent_cfg["params"]["config"]["minibatch_size"] = env.unwrapped.num_envs*2

        # create runner from rl-games
        runner = Runner(IsaacAlgoObserver())
        # import pdb; pdb.set_trace()
        runner.algo_factory.register_builder('my_agent', lambda **kwargs : A2CAgent(**kwargs))
        runner.player_factory.register_builder('my_agent', lambda **kwargs : PpoPlayerContinuous(**kwargs))
        runner.load(agent_cfg)

        # set seed of the env
        env.seed(agent_cfg["params"]["seed"])
        # reset the agent and env
        runner.reset()
        
        # train the agent
        runner.run({"train": not args_cli.play, "play": args_cli.play, "sigma": None, "checkpoint": args_cli.checkpoint})

        # close the simulator
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
