import os
import json
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from disp_sac import *

def to_device(data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.tensor(data, dtype = torch.float32).to(device)
    return data

def graph_action():
    values = [0, 1]
    # 初始化一个空列表，用于存储所有情况的1维向量
    all_combinations = []
    # 嵌套循环遍历所有可能的取值组合
    for a in values:
        all_combinations.append(np.array([a]))
    # 将结果列表转换为 NumPy 数组
    result = np.array(all_combinations)
    return result

def to_multi_discrete(action):
    result = graph_action()
    result = result[action]
    return result[0]

class PSACTrainer():
    """ train soft actor-critic """
    def __init__(self, args):

        self.env = args.env
        # init agent
        self.agent = SAC(
            state_dim= self.env.observation_space.shape[0],
            action_dim= self.env.action_space.n,
            auto_alpha=False
        )
        self.agent.train()

        
        # running parameters
        self.n_steps = args.n_steps
        self.start_learning = args.start_learning # 存储记忆 5000
        self.update_interval = args.update_interval
        self.batch_size = args.batch_size
        self.eval_interval = args.eval_interval
        self.eval_n_episodes = args.eval_n_episodes
        self.save_interval = args.save_interval
        self.device = args.device
        self.args = args

        self.model_dir = "result/{}/{}/model".format("psac", args.record_time)
        self.record_dir = "result/{}/{}/record".format("psac", args.record_time)
        self.log_dir = "result/{}/{}/log".format("psac", args.record_time)
        if not os.path.exists(self.model_dir): 
            os.makedirs(self.model_dir)
        if not os.path.exists(self.record_dir):
            os.makedirs(self.record_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.logger = SummaryWriter(self.log_dir)
    def _reset(self):
        obs = self.env.reset()
        return obs
    def _eval_policy(self):
        """ evaluate policy """
        episode_rewards = []
        for _ in range(self.eval_n_episodes):
            done = False
            episode_rewards.append(0)
            obs = self.env.reset()
            while not done:
                obs = to_device(obs)               
                action = self.agent.select_action(obs, deterministic=True)
                action = to_multi_discrete(action)
                obs, reward, done, info = self.env.step(action)
                episode_rewards[-1] += reward
        return np.mean(episode_rewards)
    
    def _warm_up(self):
        """ randomly sample a lot of transitions into buffer before starting learning """
        obs = self._reset()
        # step for {self.start_learning} time-steps
        pbar = tqdm(range(self.start_learning), desc="Warming up")
        for _ in pbar:
            action = self.env.action_space.sample()
            next_obs, reward, done, info = self.env.step(action)
            # 添加样本到经验回放缓冲区
            self.agent.add_to_replay_buffer(obs, action, reward, next_obs, done)
            obs = next_obs
            if done: obs = self._reset()
        return obs

    def run(self):
        """ train {args.algo} on {args.env} for {args.n_steps} steps"""
        # init
        records = {"step": [], "loss": {"actor": [], "critic1": [], "critic2": []}, "alpha": [], "reward": []}
        _ = self._warm_up()
        obs = self._reset()

        actor_loss, critic1_loss, critic2_loss, alpha = [None]*4
        eval_reward= [None]
        pbar = tqdm(range(self.n_steps), desc="Training {} on {}".format("psac", "MotorEnv"))
        # 先存储记忆
        for it in pbar:
            # step in env
            action = self.agent.select_action(to_device(obs)) # 索引
            next_obs, reward, done, info = self.env.step(to_multi_discrete(action))
            self.agent.add_to_replay_buffer(obs, action, reward, next_obs, done)
            obs = next_obs
            if done: obs = self._reset()
            # update policy
            if it%self.update_interval == 0:
                learning_info = self.agent.update(self.batch_size)
                actor_loss = learning_info["loss"]["actor"]
                critic1_loss = learning_info["loss"]["critic1"]
                critic2_loss = learning_info["loss"]["critic2"]
                alpha = learning_info["alpha"]
                
                records["step"].append(it)
                records["loss"]["actor"].append(actor_loss)
                records["loss"]["critic1"].append(critic1_loss)
                records["loss"]["critic2"].append(critic2_loss)
                records["alpha"].append(alpha)
                
                self.logger.add_scalar("loss/actor", actor_loss, it)
                self.logger.add_scalar("loss/critic1", critic1_loss, it)
                self.logger.add_scalar("loss/critic2", critic2_loss, it)
                self.logger.add_scalar("alpha", alpha, it)
                
                if eval_reward:
                    records["reward"].append(eval_reward)

            if it%self.eval_interval == 0:
                eval_reward= self._eval_policy()
                if eval_reward > -1000:
                    self.agent.save_model(os.path.join(self.model_dir, f"model_time-{self.args.record_time}_reward-{eval_reward}.pth"))
                self.logger.add_scalar("eval/reward", eval_reward, it)
                # self.logger.add_scalar("eval/reward_yaw", yaw_reward, it)
                # self.logger.add_scalar("eval/reward_car_speed", car_speed_reward, it)
                # self.logger.add_scalar("eval/reward_slip_ratio", slip_ratio_reward, it)

            pbar.set_postfix(
                alpha=alpha,
                actor_loss=actor_loss, 
                critic1_loss=critic1_loss, 
                critic2_loss=critic2_loss, 
                eval_reward=eval_reward
            )

            # save
            if it%self.save_interval == 0: self._save(records)

        self._save(records)
        self.logger.close()
    
    def _save(self, records):
        """ save model and record """
        self.agent.save_model(os.path.join(self.model_dir, "model_time-{}.pth".format(self.args.record_time)))
        with open(os.path.join(self.record_dir, "record_time-{}.txt".format(self.args.record_time)), "w") as f:
            json.dump(records, f)
