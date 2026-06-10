"""Device helpers for RLCard DMC training on CPU, CUDA, or Apple MPS."""

import torch


def resolve_torch_device(cuda=""):
    """Pick a torch device string for inference or training."""
    value = str(cuda)
    if value in ("", "auto"):
        if torch.backends.mps.is_available():
            return "mps:0"
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"
    return resolve_dmc_device(value)


def resolve_dmc_device(device):
    """Map DMC device identifiers to torch device strings."""
    value = str(device)
    if value == "cpu":
        return "cpu"
    if value == "mps" or value.startswith("mps:"):
        return "mps:0" if value == "mps" else value
    if value.startswith("cuda:"):
        return value
    if value == "":
        return resolve_torch_device("")
    return f"cuda:{value}"


def is_mps_training_device(training_device):
    return str(training_device).startswith("mps")


def resolve_training_device(cuda, training_device):
    """Pick the learner device when not using CUDA_VISIBLE_DEVICES."""
    if cuda:
        return training_device

    if training_device == "cpu":
        return "cpu"
    if is_mps_training_device(training_device):
        return "mps"
    if training_device != "0":
        return training_device

    if torch.backends.mps.is_available():
        print("--> Training on MPS (actors stay on CPU)")
        return "mps"
    return "cpu"


def apply_dmc_device_patches():
    """Patch RLCard DMC to understand MPS device strings."""
    from rlcard.agents.dmc_agent import model, trainer

    original_agent_init = model.DMCAgent.__init__
    original_learn = trainer.learn
    original_trainer_init = trainer.DMCTrainer.__init__

    def patched_agent_init(
        self,
        state_shape,
        action_shape,
        mlp_layers=None,
        exp_epsilon=0.01,
        device="0",
    ):
        if mlp_layers is None:
            mlp_layers = [512, 512, 512, 512, 512]
        self.use_raw = False
        self.device = resolve_dmc_device(device)
        self.net = model.DMCNet(state_shape, action_shape, mlp_layers).to(self.device)
        self.exp_epsilon = exp_epsilon
        self.action_shape = action_shape

    def patched_learn(
        position,
        actor_models,
        agent,
        batch,
        optimizer,
        training_device,
        max_grad_norm,
        mean_episode_return_buf,
        lock,
    ):
        device = resolve_dmc_device(training_device)
        state = torch.flatten(batch["state"].to(device), 0, 1).float()
        action = torch.flatten(batch["action"].to(device), 0, 1).float()
        target = torch.flatten(batch["target"].to(device), 0, 1)
        episode_returns = batch["episode_return"][batch["done"]]
        mean_episode_return_buf[position].append(torch.mean(episode_returns).to(device))

        with lock:
            values = agent.forward(state, action)
            loss = trainer.compute_loss(values, target)
            stats = {
                "mean_episode_return_" + str(position): torch.mean(
                    torch.stack([_r for _r in mean_episode_return_buf[position]])
                ).item(),
                "loss_" + str(position): loss.item(),
            }

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

            for actor_model in actor_models.values():
                actor_model.get_agent(position).load_state_dict(agent.state_dict())
            return stats

    def patched_trainer_init(self, env, cuda="", training_device="0", **kwargs):
        original_trainer_init(self, env, cuda=cuda, training_device=training_device, **kwargs)
        if cuda == "" and is_mps_training_device(training_device):
            self.device_iterator = ["cpu"]
            self.training_device = "mps"

    model.DMCAgent.__init__ = patched_agent_init
    trainer.learn = patched_learn
    trainer.DMCTrainer.__init__ = patched_trainer_init

    original_start = trainer.DMCTrainer.start

    def patched_start(self):
        original_load = torch.load

        def load_with_device(path, **load_kwargs):
            if path == self.checkpointpath and "map_location" not in load_kwargs:
                load_kwargs["map_location"] = resolve_dmc_device(self.training_device)
            return original_load(path, **load_kwargs)

        torch.load = load_with_device
        try:
            return original_start(self)
        finally:
            torch.load = original_load

    trainer.DMCTrainer.start = patched_start
