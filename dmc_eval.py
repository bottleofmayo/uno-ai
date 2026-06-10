"""Periodic evaluation hooks for RLCard DMC training."""

import io
import os
import pprint
import threading
import time
import timeit

import torch
from torch import multiprocessing as mp

import rlcard
from rlcard.utils import tournament

from dmc_device import resolve_dmc_device


def clone_agents_for_eval(learner_model):
    agents = []
    for agent in learner_model.get_agents():
        buffer = io.BytesIO()
        torch.save(agent, buffer)
        buffer.seek(0)
        eval_agent = torch.load(buffer, map_location="cpu", weights_only=False)
        if hasattr(eval_agent, "net"):
            eval_agent.device = "cpu"
            eval_agent.net.to("cpu")
        eval_agent.eval()
        agents.append(eval_agent)
    return agents


def evaluate_learner(learner_model, eval_config):
    eval_env = rlcard.make(
        eval_config["env_name"],
        config={"seed": eval_config["seed"]},
    )
    agents = clone_agents_for_eval(learner_model)
    eval_env.set_agents(agents)
    reward = tournament(eval_env, eval_config["num_eval_games"])[0]
    eval_config["logger"].log_performance(eval_config["_next_eval_episode"], reward)
    return reward


def _start_with_periodic_eval(self):
    from rlcard.agents.dmc_agent.trainer import (
        act,
        act_pettingzoo,
        create_buffers,
        create_buffers_pettingzoo,
        create_optimizers,
        get_batch,
        learn,
        log,
    )

    eval_config = self._eval_config
    eval_lock = threading.Lock()

    models = {}
    for device in self.device_iterator:
        model = self.model_func(device)
        model.share_memory()
        model.eval()
        models[device] = model

    if not self.is_pettingzoo_env:
        buffers = create_buffers(
            self.T,
            self.num_buffers,
            self.env.state_shape,
            self.action_shape,
            self.device_iterator,
        )
    else:
        buffers = create_buffers_pettingzoo(
            self.T,
            self.num_buffers,
            self.env,
            self.device_iterator,
        )

    actor_processes = []
    ctx = mp.get_context("spawn")
    free_queue = {}
    full_queue = {}
    for device in self.device_iterator:
        _free_queue = [ctx.SimpleQueue() for _ in range(self.num_players)]
        _full_queue = [ctx.SimpleQueue() for _ in range(self.num_players)]
        free_queue[device] = _free_queue
        full_queue[device] = _full_queue

    learner_model = self.model_func(self.training_device)
    optimizers = create_optimizers(
        self.num_players,
        self.learning_rate,
        self.momentum,
        self.epsilon,
        self.alpha,
        learner_model,
    )

    stat_keys = []
    for p in range(self.num_players):
        stat_keys.append("mean_episode_return_" + str(p))
        stat_keys.append("loss_" + str(p))
    frames, stats = 0, {k: 0 for k in stat_keys}

    map_location = resolve_dmc_device(self.training_device)
    if self.load_model and os.path.exists(self.checkpointpath):
        checkpoint_states = torch.load(
            self.checkpointpath,
            map_location=map_location,
        )
        for p in range(self.num_players):
            learner_model.get_agent(p).load_state_dict(
                checkpoint_states["model_state_dict"][p]
            )
            optimizers[p].load_state_dict(checkpoint_states["optimizer_state_dict"][p])
            for device in self.device_iterator:
                models[device].get_agent(p).load_state_dict(
                    learner_model.get_agent(p).state_dict()
                )
        stats = checkpoint_states["stats"]
        frames = checkpoint_states["frames"]
        log.info(f"Resuming preempted job, current stats:\n{stats}")

    for device in self.device_iterator:
        for i in range(self.num_actors):
            actor = ctx.Process(
                target=act_pettingzoo if self.is_pettingzoo_env else act,
                args=(
                    i,
                    device,
                    self.T,
                    free_queue[device],
                    full_queue[device],
                    models[device],
                    buffers[device],
                    self.env,
                ),
            )
            actor.start()
            actor_processes.append(actor)

    def maybe_evaluate():
        nonlocal frames
        episode = frames // eval_config["frames_per_episode"]
        while episode >= eval_config["_next_eval_episode"]:
            with eval_lock:
                if episode < eval_config["_next_eval_episode"]:
                    break
                evaluate_learner(learner_model, eval_config)
                eval_config["_next_eval_episode"] += eval_config["evaluate_every"]

    def batch_and_learn(i, device, position, local_lock, position_lock, lock=threading.Lock()):
        nonlocal frames, stats
        while frames < self.total_frames:
            batch = get_batch(
                free_queue[device][position],
                full_queue[device][position],
                buffers[device][position],
                self.B,
                local_lock,
            )
            _stats = learn(
                position,
                models,
                learner_model.get_agent(position),
                batch,
                optimizers[position],
                self.training_device,
                self.max_grad_norm,
                self.mean_episode_return_buf,
                position_lock,
            )

            with lock:
                for k in _stats:
                    stats[k] = _stats[k]
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stat_keys})
                self.plogger.log(to_log)
                frames += self.T * self.B

    for device in self.device_iterator:
        for m in range(self.num_buffers):
            for p in range(self.num_players):
                free_queue[device][p].put(m)

    threads = []
    locks = {
        device: [threading.Lock() for _ in range(self.num_players)]
        for device in self.device_iterator
    }
    position_locks = [threading.Lock() for _ in range(self.num_players)]

    for device in self.device_iterator:
        for i in range(self.num_threads):
            for position in range(self.num_players):
                thread = threading.Thread(
                    target=batch_and_learn,
                    name="batch-and-learn-%d" % i,
                    args=(
                        i,
                        device,
                        position,
                        locks[device][position],
                        position_locks[position],
                    ),
                )
                thread.start()
                threads.append(thread)

    def checkpoint(frame_count):
        log.info("Saving checkpoint to %s", self.checkpointpath)
        _agents = learner_model.get_agents()
        torch.save(
            {
                "model_state_dict": [_agent.state_dict() for _agent in _agents],
                "optimizer_state_dict": [
                    optimizer.state_dict() for optimizer in optimizers
                ],
                "stats": stats,
                "frames": frame_count,
            },
            self.checkpointpath,
        )

        for position in range(self.num_players):
            model_weights_dir = os.path.expandvars(
                os.path.expanduser(
                    "%s/%s/%s"
                    % (
                        self.savedir,
                        self.xpid,
                        str(position) + "_" + str(frame_count) + ".pth",
                    )
                )
            )
            torch.save(learner_model.get_agent(position), model_weights_dir)

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()
        while frames < self.total_frames:
            start_frames = frames
            start_time = timer()
            time.sleep(5)

            maybe_evaluate()

            if timer() - last_checkpoint_time > self.save_interval * 60:
                checkpoint(frames)
                last_checkpoint_time = timer()

            end_time = timer()
            fps = (frames - start_frames) / (end_time - start_time)
            log.info(
                "After %i frames: @ %.1f fps Stats:\n%s",
                frames,
                fps,
                pprint.pformat(stats),
            )
    except KeyboardInterrupt:
        return learner_model
    else:
        for thread in threads:
            thread.join()
        log.info("Learning finished after %d frames.", frames)

    maybe_evaluate()
    checkpoint(frames)
    self.plogger.close()
    return learner_model


def apply_dmc_eval_patch():
    from rlcard.agents.dmc_agent import trainer

    original_start = trainer.DMCTrainer.start

    def patched_start(self):
        if getattr(self, "_eval_config", None):
            return _start_with_periodic_eval(self)
        return original_start(self)

    trainer.DMCTrainer.start = patched_start
