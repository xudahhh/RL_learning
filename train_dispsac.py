import os
import pickle
import argparse
import numpy as np

import torch
import torch.onnx
import torch.nn as nn

from dispsac_run import *

from datetime import datetime


def get_args():
    
    parser = argparse.ArgumentParser(description="sac")
    
    # 创建REINFORCE对象
    env = gym.make('CartPole-v0',render_mode = "human")
    parser.add_argument("--env", type=float, default=env)    
    parser.add_argument("--actor-lr", type=float, default=3e-4)                         # learning rate of actor
    parser.add_argument("--critic-lr", type=float, default=3e-4)                        # learning rate of critic
    
    # running parameters
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=True)
    parser.add_argument("--n-steps", type=int, default=int(2e6))
    parser.add_argument("--buffer-size", type=int, default=int(2e6))
    parser.add_argument("--start-learning", type=int, default=int(5000))
    parser.add_argument("--update-interval", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)                           # mini-batch size
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--eval-n-episodes", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=int(1e4))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
 
    args = parser.parse_args()
    return args


def train_psac(args):
    str_t = datetime.now()
    record_time = str_t.strftime("%m%d%H%M%S")
    args.record_time = record_time + "gym"
    trainer = PSACTrainer(args)
    trainer.run()
def test_psac(args):

    
    # 10ms opt: load_time = "0302104938"
    # 50ms opt: load_time = "0307155506"
    load_time = "0718102735gym"
    agent = SAC(
        state_dim= args.env.observation_space.shape[0],
        action_dim= args.env.action_space.n,
    )
    agent.load_model(f"result/psac/{load_time}/model/model_time-{load_time}_reward-200.0.pth")
    with open(f"result/psac/{load_time}/model/agent_time-{load_time}.pkl", "wb") as f:
        pickle.dump(agent, f)

    episode_rewards = []
    for _ in range(args.eval_n_episodes):
        done = False
        episode_rewards.append(0)
        obs = args.env.reset()
        while not done:
            obs = to_device(obs)               
            action = agent.select_action(obs, deterministic=True)
            action = to_multi_discrete(action)
            obs, reward, done, info = args.env.step(action)
            episode_rewards[-1] += reward
    print(episode_rewards)

if __name__ == "__main__":
    args = get_args()
    if args.train: train_psac(args)
    else: test_psac(args)

