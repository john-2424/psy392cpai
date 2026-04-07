from __future__ import annotations

from tensordict.nn import (
    TensorDictModule,
    TensorDictSequential,
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
)
from torch.distributions import Categorical
from torchrl.modules import ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from src.models.cnn_encoder import GridCNNEncoder
from src.models.policy_heads import DiscretePolicyHead
from src.models.value_heads import ValueHead


def make_common_module(feature_dim: int = 128) -> TensorDictModule:
    encoder = GridCNNEncoder(feature_dim=feature_dim)
    return TensorDictModule(
        module=encoder,
        in_keys=["observation"],
        out_keys=["features"],
    )


def make_policy(action_spec, feature_dim: int = 128, hidden_dim: int = 64):
    common_module = make_common_module(feature_dim=feature_dim)

    policy_head = DiscretePolicyHead(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        n_actions=4,
    )

    logits_module = TensorDictModule(
        module=policy_head,
        in_keys=["features"],
        out_keys=["logits"],
    )

    dist_module = ProbabilisticTensorDictModule(
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=Categorical,
        return_log_prob=True,
    )

    # Important: the final policy object itself must be probabilistic
    policy = ProbabilisticTensorDictSequential(
        common_module,
        logits_module,
        dist_module,
    )
    return policy


def make_value_model(feature_dim: int = 128, hidden_dim: int = 64):
    common_module = make_common_module(feature_dim=feature_dim)

    value_head = ValueHead(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
    )

    critic = ValueOperator(
        module=value_head,
        in_keys=["features"],
        out_keys=["state_value"],
    )

    value_model = TensorDictSequential(common_module, critic)
    return value_model


def make_ppo_models(action_spec, feature_dim: int = 128, hidden_dim: int = 64):
    policy = make_policy(
        action_spec=action_spec,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
    )
    value_model = make_value_model(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
    )
    return policy, value_model


def make_ppo_loss(
    policy,
    value_model,
    gamma: float = 0.99,
    lmbda: float = 0.95,
    clip_epsilon: float = 0.2,
    entropy_coeff: float = 0.01,
    critic_coeff: float = 1.0,
):
    advantage_module = GAE(
        gamma=gamma,
        lmbda=lmbda,
        value_network=value_model,
        average_gae=True,
    )

    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=value_model,
        clip_epsilon=clip_epsilon,
        entropy_coeff=entropy_coeff,
        critic_coeff=critic_coeff,
    )

    return advantage_module, loss_module


def build_ppo_components(
    env,
    feature_dim: int = 128,
    hidden_dim: int = 64,
    gamma: float = 0.99,
    lmbda: float = 0.95,
    clip_epsilon: float = 0.2,
    entropy_coeff: float = 0.01,
    critic_coeff: float = 1.0,
):
    policy, value_model = make_ppo_models(
        action_spec=env.action_spec,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
    )

    advantage_module, loss_module = make_ppo_loss(
        policy=policy,
        value_model=value_model,
        gamma=gamma,
        lmbda=lmbda,
        clip_epsilon=clip_epsilon,
        entropy_coeff=entropy_coeff,
        critic_coeff=critic_coeff,
    )

    return {
        "policy": policy,
        "value_model": value_model,
        "advantage_module": advantage_module,
        "loss_module": loss_module,
    }