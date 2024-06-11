import os
import sys
import csv
import copy
import random
import json
import numpy as np
import torch
from matplotlib import pyplot as plt
import wandb
from grid2op.Episode import EpisodeData
from grid2op.Runner.basic_logger import DoNothingLog, ConsoleLog
from grid2op.dtypes import dt_int, dt_float, dt_bool
import time
from tensorboardX import SummaryWriter

class TrainAgent(object):
    def __init__(self, agent, env, test_env, device,dn_json_path, serialized_agent_dir, dn_ffw, ep_infos, episode = None, epsilon):
        self.device = device
        self.agent = agent
        self.env = env
        self.test_env = test_env
        self.dn_json_path = dn_json_path
        self.serialized_agent_dir = serialized_agent_dir
        self.dn_ffw = dn_ffw
        self.ep_infos = ep_infos
        self.episode = episode
        self.writer = SummaryWriter()
        self.global_timestep_count = 0
        self.global_episode_count = 0
        self.epsilon = epsilon


    def save_model(self, path, name):
        self.agent.save_model(path, name)
    
    def store_chronic_prio(self, chronic_prio):
        chronic_prio_path = os.path.join(self.agent.cversion_path, f'chronic_priority.npy')
        np.save(chronic_prio_path , np.array(chronic_prio))

    def load_chronic_prio(self):
        chronic_prio_path = os.path.join(self.agent.cversion_path, f'chronic_priority.npy')
        a = np.load(chronic_prio_path)
        return a

    # following competition evaluation script
    def compute_episode_score(self, chronic_id, agent_step, agent_reward, ffw=None):
        min_losses_ratio = 0.7
        ep_marginal_cost = self.env.gen_cost_per_MW.max()
        print("chronic id in episode score:", chronic_id)
        if ffw is None:
            ep_do_nothing_reward = self.ep_infos[chronic_id]["donothing_reward"]
            ep_do_nothing_nodisc_reward = self.ep_infos[chronic_id]["donothing_nodisc_reward"]
            ep_dn_played = self.ep_infos[chronic_id]['dn_played']
            ep_loads = np.array(self.ep_infos[chronic_id]["sum_loads"])
            ep_losses = np.array(self.ep_infos[chronic_id]["losses"])
        else:
            start_idx = 0 if ffw == 0 else ffw * 288 - 2
            end_idx = start_idx + 864
            ep_dn_played, ep_do_nothing_reward, ep_do_nothing_nodisc_reward = self.dn_ffw[(chronic_id, ffw)]
            ep_loads = np.array(self.ep_infos[chronic_id]["sum_loads"])[start_idx:end_idx]
            ep_losses = np.array(self.ep_infos[chronic_id]["losses"])[start_idx:end_idx]

        # Add cost of non delivered loads for blackout steps
        blackout_loads = ep_loads[agent_step:]
        if len(blackout_loads) > 0:
            blackout_reward = np.sum(blackout_loads) * ep_marginal_cost
            agent_reward += blackout_reward

        # Compute ranges
        worst_reward = np.sum(ep_loads) * ep_marginal_cost
        best_reward = np.sum(ep_losses) * min_losses_ratio
        zero_reward = ep_do_nothing_reward
        zero_blackout = ep_loads[ep_dn_played:]
        zero_reward += np.sum(zero_blackout) * ep_marginal_cost
        nodisc_reward = ep_do_nothing_nodisc_reward

        # Linear interp episode reward to codalab score
        if zero_reward != nodisc_reward:
            # DoNothing agent doesnt complete the scenario
            reward_range = [best_reward, nodisc_reward, zero_reward, worst_reward]
            score_range = [100.0, 80.0, 0.0, -100.0]
        else:
            # DoNothing agent can complete the scenario
            reward_range = [best_reward, zero_reward, worst_reward]
            score_range = [100.0, 0.0, -100.0]
            
        ep_score = np.interp(agent_reward, reward_range, score_range)
        return ep_score

    def interaction(self, obs, prev_act, cid, ffw, start_step):
        state = self.agent.get_current_state()
        adj = self.agent.adj.clone()
        action = self.agent.goal.clone()
        order = None if self.agent.order is None else self.agent.order.clone()
        reward, train_reward, step = 0, 0, 0
        time_act = 0.0
        while True:
            # prev_act is executed at first anyway
            if prev_act:
                beg__ = time.perf_counter()
                act = prev_act
                prev_act = None
                end__ = time.perf_counter()
            else:
                beg__ = time.perf_counter()
                act = self.agent.act(obs, None, None)
                end__ = time.perf_counter()
                if self.agent.save:
                    # pass this act to the next step.
                    prev_act = act
                    break
            time_act += end__ - beg__
            # just step if action is okay or failed to find other action
            obs, rew, done, info = self.env.step(act)
            reward += rew
            self.episode.incr_store(efficient_storing = True, time_step = step, 
                               time_step_duration = end__ - beg__, reward = rew, env_act = self.env._env_modification,
                               act = act, obs = obs, opp_attack = None, info = info)
            new_reward = info['rewards']['loss']
            train_reward += new_reward
            step += 1
            self.global_timestep_count += 1

            self.writer.add_scalar('Loss Reward per time step', train_reward, global_step=self.global_timestep_count)
            self.writer.add_scalar('Reward per time step', reward, global_step=self.global_timestep_count)

            if start_step + step == 864:
                done = True

            if done:
                break
        train_reward = np.clip(train_reward, -2, 10)

        next_state = self.agent.get_current_state()
        next_adj = self.agent.adj.clone()
        die = bool(done and info['exception'])
        transition = (state, adj, action, train_reward, next_state, next_adj, die, order)
        etcs = (step + start_step, prev_act, info)
        infos = (transition, etcs, time_act)
        return obs, reward, done, infos

    def multi_step_transition(self, temp_memory):
        transitions = []
        running_reward = 0
        final_state, final_adj, final_die = temp_memory[-1][4:7]
        for tran in reversed(temp_memory):
            (state, adj, action, train_reward, _,_,_, order) = tran
            running_reward += train_reward
            new_tran = (state, adj, action, running_reward, final_state, final_adj, final_die, order)
            transitions.append(new_tran)

        return transitions
    
    # compute weight for chronic sampling
    def chronic_priority(self, cid, ffw, step):
        m = 864
        scale = 2.
        diff_coef = 0.05
        d = self.dn_ffw[(cid, ffw)][0]
        progress = 1 - np.sqrt(step/m)
        difficulty = 1 - np.sqrt(d/m)
        score = (progress + diff_coef * difficulty) * scale
        return score
    
    def initialize_episodeData(self, efficient_storing, nb_timestep_max):
        disc_lines_templ = np.full((1, self.env.backend.n_line), fill_value=False, dtype=dt_bool)

        attack_templ = np.full(
        (1, self.env._oppSpace.action_space.size()), fill_value=0.0, dtype=dt_float
        )
        if efficient_storing:
            times = np.full(nb_timestep_max, fill_value=np.NaN, dtype=dt_float)
            rewards_episode = np.full(nb_timestep_max, fill_value=np.NaN, dtype=dt_float)
            actions = np.full(
                (nb_timestep_max, self.env.action_space.n), fill_value=np.NaN, dtype=dt_float
            )
            env_actions = np.full(
                (nb_timestep_max, self.env._helper_action_env.n),
                fill_value=np.NaN,
                dtype=dt_float,
            )
            observations = np.full(
                (nb_timestep_max + 1, self.env.observation_space.n),
                fill_value=np.NaN,
                dtype=dt_float,
            )
            disc_lines = np.full(
                (nb_timestep_max, self.env.backend.n_line), fill_value=np.NaN, dtype=dt_bool
            )
            attack = np.full(
                (nb_timestep_max, self.env._opponent_action_space.n),
                fill_value=0.0,
                dtype=dt_float,
            )
            legal = np.full(nb_timestep_max, fill_value=True, dtype=dt_bool)
            ambiguous = np.full(nb_timestep_max, fill_value=False, dtype=dt_bool)
        else:
            times = np.full(0, fill_value=np.NaN, dtype=dt_float)
            rewards_episode = np.full(0, fill_value=np.NaN, dtype=dt_float)
            actions = np.full((0, self.env.action_space.n), fill_value=np.NaN, dtype=dt_float)
            env_actions = np.full(
                (0, self.env._helper_action_env.n), fill_value=np.NaN, dtype=dt_float
            )
            observations = np.full(
                (0, self.env.observation_space.n), fill_value=np.NaN, dtype=dt_float
            )
            disc_lines = np.full((0, self.env.backend.n_line), fill_value=np.NaN, dtype=dt_bool)
            attack = np.full(
                (0, self.env._opponent_action_space.n), fill_value=0.0, dtype=dt_float
            )
            legal = np.full(0, fill_value=True, dtype=dt_bool)
            ambiguous = np.full(0, fill_value=False, dtype=dt_bool)
        
        return disc_lines_templ, attack_templ, times, rewards_episode, actions, env_actions, observations, disc_lines, attack, legal, ambiguous

    def train(self, seed, nb_frame, test_step, train_chronics, valid_chronics, output_dir, model_path, max_ffw):
                
        best_score = -100
        time_step = int(0)
        verbose  = False
        logger = ConsoleLog(DoNothingLog.INFO if verbose else DoNothingLog.ERROR)
        if verbose:
            logger.setLevel("debug")
        else:
            logger.disabled = True

        tensorboard_logs = os.path.join(output_dir, 'tensorboard_summaries')
        self.writer = SummaryWriter(log_dir=tensorboard_logs)
        # initialize training chronic sampling weights
        train_chronics_ffw = [(cid, fw) for cid in train_chronics for fw in range(max_ffw)]
        total_chronic_num = len(train_chronics_ffw)
        chronic_records = [0] * total_chronic_num
        chronic_step_records = [0] * total_chronic_num
        
        if self.agent.run_num > 1:
            chronic_records = self.load_chronic_prio()
        else:
            for i in chronic_records:
                cid, fw = train_chronics_ffw[i]
                chronic_records[i] = self.chronic_priority(cid, fw, 1)

        # training loop
        while self.agent.update_step < nb_frame:

            # sample training chronic
            dist = torch.distributions.categorical.Categorical(logits=torch.Tensor(chronic_records))
            record_idx = dist.sample().item()
            chronic_id, ffw = train_chronics_ffw[record_idx]
            self.env.set_id(chronic_id)
            self.env.seed(seed)
            obs = self.env.reset()

            if ffw > 0:
                self.env.fast_forward_chronics(ffw * 288 - 3)
                obs, *_ = self.env.step(self.env.action_space())
            done = False
            alive_frame = 0
            total_reward = 0
            train_reward = 0
            
            need_store_first_act = self.serialized_agent_dir is not None
            nb_timestep_max = self.env.chronics_handler.max_timestep()
            efficient_storing = nb_timestep_max > 0
            nb_timestep_max = max(nb_timestep_max, 0)
            
            disc_lines_templ, attack_templ, times, rewards_episode, actions, env_actions, observations, disc_lines, attack, legal, ambiguous = self.initialize_episodeData(efficient_storing, nb_timestep_max) 
            
            if efficient_storing:
                observations[time_step, :] = obs.to_vect()
            else:
                observations = np.concatenate((observations, obs.to_vect().reshape(1, -1)))
            
            
            episode = EpisodeData(
            actions=actions,
            env_actions=env_actions,
            observations=observations,
            rewards=rewards_episode,
            disc_lines=disc_lines,
            times=times,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            helper_action_env=self.env._helper_action_env,
            path_save=self.serialized_agent_dir,
            disc_lines_templ=disc_lines_templ,
            attack_templ=attack_templ,
            attack=attack,
            attack_space=self.env._opponent_action_space,
            logger=logger,
            name=self.env.chronics_handler.get_name(),
            other_rewards=[],
            legal=legal,
            ambiguous=ambiguous,
            has_legal_ambiguous=True,
            )
            episode.serialize = True
            self.episode = episode

            if need_store_first_act:
                # I need to manually force in the first observation (otherwise it's not computed)
                self.episode.observations.objects[0] = self.episode.observations.helper.from_vect(
                observations[time_step, :]
                )
            self.episode.set_parameters(self.env)

            self.agent.reset(obs)
            prev_act = self.agent.act(obs, None, None)
            temp_memory = []
            # interaction consists of multiple steps, the agent gets killed in one of these steps or the max possible steps has been finished.
            while not done:
                beg_ = time.perf_counter()
                obs, reward, done, info = self.interaction(obs, prev_act, chronic_id, ffw, alive_frame)
                end_ = time.perf_counter()
                alive_frame, prev_act = info[1][:2]
                # episode.set_game_over(game_over_step = alive_frame) # Does not work
                total_reward += reward
                train_reward += info[0][3] 

                self.episode.set_meta(self.env, alive_frame, float(total_reward), seed, None)
                temp_memory.append(list(map(lambda x: x.cpu() if torch.is_tensor(x) else x, info[0])))
                if len(temp_memory) == self.agent.k_step or done:
                    for transition in self.multi_step_transition(temp_memory):
                        self.agent.append_sample(*transition)
                    temp_memory.clear()
                
                if len(self.agent.memory) > self.agent.update_start:
                    self.agent.update(self.writer)
                    self.global_episode_count = self.agent.update_step

                    if self.agent.update_step % test_step == 0:
                        eval_iter = self.agent.update_step // test_step
                        cache = self.agent.cache_stat()
                        result, stats, scores, steps = self.test(valid_chronics, max_ffw)
                        self.writer.add_scalar('Evaluation/real reward', stats['reward'], global_step=eval_iter)
                        self.writer.add_scalar('Evaluation/episode L2RPN score', stats['score'], global_step=eval_iter)
                        self.writer.add_scalar('Evaluation/episode steps', stats['step'], global_step=eval_iter)


                        self.agent.load_cache_stat(cache)
                        print(f"[{eval_iter:4d}] Valid: score {stats['score']} | step {stats['step']}")
                        
                        # log and save model
                        with open(os.path.join(model_path, 'score.csv'), 'a', newline='') as cf:
                            csv.writer(cf).writerow(scores)
                        with open(os.path.join(model_path, 'step.csv'), 'a', newline='') as cf:
                            csv.writer(cf).writerow(steps)
                        if best_score < stats['score']:
                            best_score = stats['score']
                            self.agent.save_model(model_path, 'best')

                    if self.agent.update_step % self.agent.update_current_version == 0:
                        self.agent.save_most_recent()
                        self.store_chronic_prio(chronic_records)

                if self.agent.update_step > nb_frame :
                    break

            time_act = info[2]
            li_text = [
            "Env: {:.2f}s",
            "\t - apply act {:.2f}s",
            "\t - run pf: {:.2f}s",
            "\t - env update + observation: {:.2f}s",
            "Agent: {:.2f}s",
            "Total time: {:.2f}s",
            "Cumulative reward: {:1f}",
            ]
            msg_ = "\n".join(li_text)
            logger.info(
                msg_.format(
                    self.env._time_apply_act + self.env._time_powerflow + self.env._time_extract_obs,
                    self.env._time_apply_act,
                    self.env._time_powerflow,
                    self.env._time_extract_obs,
                    time_act,
                    end_ - beg_,
                    total_reward,
                )
            )
            t_time = end_ - beg_
            self.writer.add_scalar('Cumulative reward', total_reward, global_step=self.global_episode_count)
            self.writer.add_scalar('Cumulative Loss reward', train_reward, global_step=self.global_episode_count)
            self.writer.add_scalar('Agent Time to Act', time_act, global_step=self.global_episode_count)
            self.writer.add_scalar('Total Episode Time', t_time, global_step=self.global_episode_count)
            self.writer.add_scalar('Episode Alive Frame', alive_frame, global_step=self.global_episode_count)

            self.episode.set_episode_times(self.env, time_act, beg_, end_)
            
            self.episode.to_disk()
            # update chronic sampling weight
            chronic_records[record_idx] = self.chronic_priority(chronic_id, ffw, alive_frame)
            chronic_step_records[record_idx] = alive_frame
            
        epsil = self.epsilon
        #No ensor file for epsilon = 0
        file_name = os.path.join(output_dir, f'ep_by_normjacobians_{epsil}_rn.pt')
        torch.save(self.agent.ep_by_normjacobians, file_name)
        if (self.agent.welford_state_n > 1):
            torch.save(self.agent.welford_state_mean, os.path.join(output_dir, f'mean.pt'))
            torch.save(self.agent.welford_state_mean, os.path.join(output_dir, f'std.pt'))

    def test(self, chronics, max_ffw, f=None, verbose=False):
        result = {}
        steps, scores = [], []

        if max_ffw == 5:
            chronics = chronics * 5
        for idx, i in enumerate(chronics):
            if max_ffw == 5:
                ffw = idx
            else:
                ffw = int(np.argmin([self.dn_ffw[(i, fw)][0] for fw in range(max_ffw) if (i, fw) in self.dn_ffw and self.dn_ffw[(i, fw)][0] >= 10]))

            dn_step = self.dn_ffw[(i, ffw)][0]
            self.test_env.seed(59)
            self.test_env.set_id(i)
            obs = self.test_env.reset()            
            self.agent.reset(obs)
            
            if ffw > 0:
                self.test_env.fast_forward_chronics(ffw * 288 - 3)
                obs, *_ = self.test_env.step(self.test_env.action_space())

            total_reward = 0
            alive_frame = 0
            done = False
            result[(i, ffw)] = {}
            while not done:
                act = self.agent.act(obs, 0, 0)
                obs, reward, done, info = self.test_env.step(act)
                total_reward += reward
                alive_frame += 1
                if alive_frame == 864:
                    done = True
            
            l2rpn_score = float(self.compute_episode_score(i, alive_frame, total_reward, ffw))
            print(f'[Test Ch{i:4d}({ffw:2d})] {alive_frame:3d}/864 ({dn_step:3d}) Score: {l2rpn_score:9.4f} ')
            scores.append(l2rpn_score)
            steps.append(alive_frame)

            result[(i, ffw)]["real_reward"] = total_reward
            result[(i, ffw)]["reward"] = l2rpn_score
            result[(i, ffw)]["step"] = alive_frame

        val_step = val_score = val_rew = 0
        for key in result:
            val_step += result[key]['step']
            val_score += result[key]['reward']
            val_rew += result[key]['real_reward']
        stats = {
            'step': val_step / len(chronics),
            'score': val_score / len(chronics),
            'reward': val_rew / len(chronics),
            'alpha': self.agent.log_alpha.exp().item()
        }
        return result, stats, scores, steps

    def evaluate(self, model_name, chronics, max_ffw, fig_path, mode='best', plot_topo=False):
        print("in evaluate")
        tensorboard_logs = os.path.join(fig_path, 'tensorboard_summaries')
        self.writer = SummaryWriter(log_dir=tensorboard_logs)
        
        if plot_topo:
            from grid2op.PlotGrid import PlotMatplot
            plot_helper = PlotMatplot(self.test_env.observation_space, width=1280, height=1280,
                                    sub_radius=7.5, gen_radius=2.5, load_radius=2.5)
            self.test_env.attach_renderer()
        result = {}
        steps, scores = [], []
        
        sub_ids = [str(i) for i in range(self.test_env.action_space().n_sub)]
        act_substation_dict = dict.fromkeys(sub_ids, 0)
        print(act_substation_dict)

        if max_ffw == 5:
            chronics = chronics * 5
        test_global_step = 0
        for idx, i in enumerate(chronics):
            print("chronic_id", i)
            print("ffw",idx)
            if max_ffw == 5:
                ffw = idx
            else:
                ffw = int(np.argmin([self.dn_ffw[(i, fw)][0] for fw in range(max_ffw) if (i, fw) in self.dn_ffw and self.dn_ffw[(i, fw)][0] >= 10]))

            dn_step = self.dn_ffw[(i, ffw)][0]
            self.test_env.seed(59)
            self.test_env.set_id(i)
            obs = self.test_env.reset()
            self.agent.reset(obs)
            
            if ffw > 0:
                self.test_env.fast_forward_chronics(ffw * 288 - 3)
                obs, *_ = self.test_env.step(self.test_env.action_space())

            total_reward = 0
            alive_frame = 0
            done = False
            topo_dist = []

            result[(i, ffw)] = {}
            bus_goal = None
            while not done:
                if plot_topo:
                    danger = not self.agent.is_safe(obs)
                    if self.agent.save and danger:
                        temp_acts = []
                        temp_obs = [obs]
                        bus_goal = self.agent.bus_goal.numpy() + 1
                        prev_topo = obs.topo_vect[self.agent.converter.sub_mask]
                        prev_step = alive_frame
                    topo_dist.append(float((obs.topo_vect==2).sum()))
                act = self.agent.act(obs, 0, 0)
                obs, reward, done, info = self.test_env.step(act)
                total_reward += reward
                alive_frame += 1

                act_dict = act.as_dict()
                if ('change_bus_vect' in act_dict) or ('set_bus_vect' in act_dict): 
                    if ('change_bus_vect' in act_dict):
                        changed_bus_dict = act_dict['change_bus_vect']
                        subid_impacted = changed_bus_dict['modif_subs_id']
                        total_action_subid = [len(changed_bus_dict[each]) for each in subid_impacted]
                        for impacted_id in range(len(subid_impacted)):
                            key = subid_impacted[impacted_id]
                            act_substation_dict[key] += total_action_subid[impacted_id]     
                    elif ('set_bus_vect' in act_dict):
                        set_bus_dict = act_dict['set_bus_vect']
                        subid_impacted = set_bus_dict['modif_subs_id']
                        total_action_subid = [len(set_bus_dict[each]) for each in subid_impacted]
                        for impacted_id in range(len(subid_impacted)):
                            key = subid_impacted[impacted_id]
                            act_substation_dict[key] += total_action_subid[impacted_id]

                if plot_topo:
                    if bus_goal is not None:
                        temp_acts.append(act)
                        temp_obs.append(obs)
                        if self.agent.is_safe(obs) and len(self.agent.low_actions)==0:
                            if (np.sum([a == self.test_env.action_space() for a in temp_acts]) < len(temp_acts) -1) and alive_frame - prev_step > 1:
                                temp_topo = obs.topo_vect[self.agent.converter.sub_mask]
                                print('Prev:', prev_topo)
                                print('Goal:', bus_goal)
                                print('Topo:', temp_topo)
                                for j in range(3):
                                    fig = plot_helper.plot_obs(temp_obs[j], line_info="rho", load_info=None, gen_info=None)
                                    fig.savefig(f'{i}_{idx}_{alive_frame}_obs{j}_{model_name}.pdf')
                                print(prev_step, alive_frame - prev_step, (prev_topo != temp_topo).sum())
                            bus_goal = None
                            temp_acts = []

                if alive_frame == 864:
                    done = True
           
            print("chronic_id",i) 
            l2rpn_score = float(self.compute_episode_score(i, alive_frame, total_reward, ffw))

            print(f'[Test Ch{i:4d}({ffw:2d})] {alive_frame:3d}/864 ({dn_step:3d}) Score: {l2rpn_score:9.4f}')
            scores.append(l2rpn_score)
            steps.append(alive_frame)

            result[(i, ffw)]["real_reward"] = total_reward
            result[(i, ffw)]["reward"] = l2rpn_score
            result[(i, ffw)]["step"] = alive_frame
            

            # plot topo dist
            if plot_topo:
                plt.figure(figsize=(8, 6))
                plt.plot(np.arange(len(topo_dist)), topo_dist)
                plt.savefig(os.path.join(fig_path, f'{mode}_{idx}_topo.png'))
                np.save(os.path.join(fig_path, f'{mode}_{idx}_topo.npy'), np.array(topo_dist))

        print(act_substation_dict)
        with open(os.path.join(fig_path, 'act_substation_dict.json'), 'w', encoding='utf-8') as f:
            json.dump(act_substation_dict, f)
        val_step = val_score = val_rew = 0
        for key in result:
            val_step += result[key]['step']
            val_score += result[key]['reward']
            val_rew += result[key]['real_reward']
            
        stats = {
            'step': val_step / len(chronics),
            'score': val_score / len(chronics),
            'reward': val_rew / len(chronics)
        }
        if plot_topo:
            with open(os.path.join(fig_path, f"{mode}_{stats['score']:.3f}.txt"), 'w') as f:
                f.write(str(stats))
                f.write(str(result))
        return stats, scores, steps

    def evaluate_neurips(self, model_name, chronics, fig_path, mode='best', plot_topo=False):
        if plot_topo:
            from grid2op.PlotGrid import PlotMatplot
            plot_helper = PlotMatplot(self.test_env.observation_space, width=1280, height=1280,
                                    sub_radius=7.5, gen_radius=2.5, load_radius=2.5)
            self.test_env.attach_renderer()
        result = {}
        steps, scores = [], []
        test_global_step = 0
        sub_ids = [str(i) for i in range(self.test_env.action_space().n_sub)]
        act_substation_dict = dict.fromkeys(sub_ids, 0)
        print(act_substation_dict)

        for idx, i in enumerate(chronics):
            test_global_step += 1
            print("Chronic",i) 
            self.test_env.seed(59)
            self.test_env.set_id(i)
            obs = self.test_env.reset()
            self.agent.reset(obs)
            
            total_reward = 0
            alive_frame = 0
            done = False
            topo_dist = []

            result[i] = {}
            bus_goal = None
            while not done:
                if plot_topo:
                    danger = not self.agent.is_safe(obs)
                    if self.agent.save and danger:
                        temp_acts = []
                        temp_obs = [obs]
                        bus_goal = self.agent.bus_goal.numpy() + 1
                        prev_topo = obs.topo_vect[self.agent.converter.sub_mask]
                        prev_step = alive_frame
                    topo_dist.append(float((obs.topo_vect==2).sum()))
                act = self.agent.act(obs, 0, 0)
                obs, reward, done, info = self.test_env.step(act)
                total_reward += reward
                alive_frame += 1

                act_dict = act.as_dict()
                if ('change_bus_vect' in act_dict) or ('set_bus_vect' in act_dict): 
                    if ('change_bus_vect' in act_dict):
                        changed_bus_dict = act_dict['change_bus_vect']
                        subid_impacted = changed_bus_dict['modif_subs_id']
                        total_action_subid = [len(changed_bus_dict[each]) for each in subid_impacted]
                        for impacted_id in range(len(subid_impacted)):
                            key = subid_impacted[impacted_id]
                            act_substation_dict[key] += total_action_subid[impacted_id]     
                    elif ('set_bus_vect' in act_dict):
                        set_bus_dict = act_dict['set_bus_vect']
                        subid_impacted = set_bus_dict['modif_subs_id']
                        total_action_subid = [len(set_bus_dict[each]) for each in subid_impacted]
                        for impacted_id in range(len(subid_impacted)):
                            key = subid_impacted[impacted_id]
                            act_substation_dict[key] += total_action_subid[impacted_id]


                if plot_topo:
                    if bus_goal is not None:
                        temp_acts.append(act)
                        temp_obs.append(obs)
                        if self.agent.is_safe(obs) and len(self.agent.low_actions)==0:
                            if (np.sum([a == self.test_env.action_space() for a in temp_acts]) < len(temp_acts) -1) and alive_frame - prev_step > 1:
                                temp_topo = obs.topo_vect[self.agent.converter.sub_mask]
                                print('Prev:', prev_topo)
                                print('Goal:', bus_goal)
                                print('Topo:', temp_topo)
                                for j in range(3):
                                    fig = plot_helper.plot_obs(temp_obs[j], line_info="rho", load_info=None, gen_info=None)
                                    fig.savefig(f'{i}_{alive_frame}_obs{j}_{model_name}.pdf')
                                print(prev_step, alive_frame - prev_step, (prev_topo != temp_topo).sum())
                            bus_goal = None
                            temp_acts = []

                if alive_frame == 864:
                    done = True
            

            print(f'[Test Ch{i:4d}] {alive_frame:3d}/864  Reward: {total_reward:9.4f}')
            steps.append(alive_frame)

            result[i]["real_reward"] = total_reward
            result[i]["step"] = alive_frame

            # plot topo dist
            if plot_topo:
                plt.figure(figsize=(8, 6))
                plt.plot(np.arange(len(topo_dist)), topo_dist)
                plt.savefig(os.path.join(fig_path, f'{mode}_{i}_topo.png'))
                np.save(os.path.join(fig_path, f'{mode}_{i}_topo.npy'), np.array(topo_dist))

        val_step = val_score = val_rew = 0
        print(act_substation_dict)
        with open(os.path.join(fig_path, 'act_substation_dict_neurips.json'), 'w', encoding='utf-8') as f:
            json.dump(act_substation_dict, f)

        for key in result:
            val_step += result[key]['step']
            val_rew += result[key]['real_reward']
            
        stats = {
            'step': val_step / len(chronics),
            'reward': val_rew / len(chronics)
        }
        if plot_topo:
            with open(os.path.join(fig_path, f"{mode}_{stats['reward']:.3f}.txt"), 'w') as f:
                f.write(str(stats))
                f.write(str(result))
        return stats, scores, steps

