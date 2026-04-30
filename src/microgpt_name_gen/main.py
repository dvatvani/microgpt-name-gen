"""
The most atomic way to train and run inference for a GPT in pure, dependency-free Python.
This file is the complete algorithm.
Everything else is just efficiency.

@karpathy
"""

from __future__ import annotations

import math
import random
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MicroGPTConfig:
    """Hyperparameters and data paths for one training + generation run."""

    seed: int = 42
    data_path: Path | str = Path("input.txt")
    names_url: str = (
        "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
    )
    download_if_missing: bool = True
    num_steps: int = 1000
    num_samples: int = 20
    temperature: float = 0.5
    learning_rate: float = 0.01
    beta1: float = 0.85
    beta2: float = 0.99
    eps_adam: float = 1e-8
    n_layer: int = 1
    n_embd: int = 16
    block_size: int = 16
    n_head: int = 4


def _ensure_corpus(config: MicroGPTConfig) -> list[str]:
    path = Path(config.data_path)
    if not path.exists():
        if not config.download_if_missing:
            msg = f"Corpus not found at {path} and download_if_missing=False"
            raise FileNotFoundError(msg)
        urllib.request.urlretrieve(config.names_url, path)
    with path.open(encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


class Value:
    __slots__ = (
        "data",
        "grad",
        "_children",
        "_local_grads",
    )

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return Value(self.data**other, (self,), (other * self.data ** (other - 1),))

    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def _make_matrix(nout, nin, std=0.08):
    return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]


def _build_state_dict(
    *,
    vocab_size: int,
    block_size: int,
    n_layer: int,
    n_embd: int,
    n_head: int,
):
    head_dim = n_embd // n_head
    state_dict = {
        "wte": _make_matrix(vocab_size, n_embd),
        "wpe": _make_matrix(block_size, n_embd),
        "lm_head": _make_matrix(vocab_size, n_embd),
    }
    for i in range(n_layer):
        state_dict[f"layer{i}.attn_wq"] = _make_matrix(n_embd, n_embd)
        state_dict[f"layer{i}.attn_wk"] = _make_matrix(n_embd, n_embd)
        state_dict[f"layer{i}.attn_wv"] = _make_matrix(n_embd, n_embd)
        state_dict[f"layer{i}.attn_wo"] = _make_matrix(n_embd, n_embd)
        state_dict[f"layer{i}.mlp_fc1"] = _make_matrix(4 * n_embd, n_embd)
        state_dict[f"layer{i}.mlp_fc2"] = _make_matrix(n_embd, 4 * n_embd)
    params = [p for mat in state_dict.values() for row in mat for p in row]
    return state_dict, params, head_dim


def gpt(
    token_id,
    pos_id,
    keys,
    values,
    *,
    state_dict,
    n_layer: int,
    n_head: int,
    head_dim: int,
):
    tok_emb = state_dict["wte"][token_id]
    pos_emb = state_dict["wpe"][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(n_layer):
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f"layer{li}.attn_wq"])
        k = linear(x, state_dict[f"layer{li}.attn_wk"])
        v = linear(x, state_dict[f"layer{li}.attn_wv"])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs : hs + head_dim]
            k_h = [ki[hs : hs + head_dim] for ki in keys[li]]
            v_h = [vi[hs : hs + head_dim] for vi in values[li]]
            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                for t in range(len(k_h))
            ]
            attn_weights = softmax(attn_logits)
            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(head_dim)
            ]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f"layer{li}.attn_wo"])
        x = [a + b for a, b in zip(x, x_residual)]
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f"layer{li}.mlp_fc1"])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f"layer{li}.mlp_fc2"])
        x = [a + b for a, b in zip(x, x_residual)]

    return linear(x, state_dict["lm_head"])


def run_microgpt(
    config: MicroGPTConfig,
    *,
    on_step: Callable[[int, int, float], None] | None = None,
) -> list[str]:
    """Train a tiny character-level GPT on names, then sample new strings.

    ``on_step`` is called as ``(step_index_1based, num_steps, loss)`` after each step.
    """
    random.seed(config.seed)
    docs = _ensure_corpus(config)
    random.shuffle(docs)

    uchars = sorted(set("".join(docs)))
    bos = len(uchars)
    vocab_size = len(uchars) + 1

    state_dict, params, head_dim = _build_state_dict(
        vocab_size=vocab_size,
        block_size=config.block_size,
        n_layer=config.n_layer,
        n_embd=config.n_embd,
        n_head=config.n_head,
    )

    m_buf = [0.0] * len(params)
    v_buf = [0.0] * len(params)

    cfg = config
    for step in range(cfg.num_steps):
        doc = docs[step % len(docs)]
        tokens = [bos] + [uchars.index(ch) for ch in doc] + [bos]
        n = min(cfg.block_size, len(tokens) - 1)

        keys, values = (
            [[] for _ in range(cfg.n_layer)],
            [[] for _ in range(cfg.n_layer)],
        )
        losses = []
        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = gpt(
                token_id,
                pos_id,
                keys,
                values,
                state_dict=state_dict,
                n_layer=cfg.n_layer,
                n_head=cfg.n_head,
                head_dim=head_dim,
            )
            probs = softmax(logits)
            loss_t = -probs[target_id].log()
            losses.append(loss_t)
        loss = (1 / n) * sum(losses)
        loss.backward()

        lr_t = cfg.learning_rate * (1 - step / cfg.num_steps)
        for i, p in enumerate(params):
            m_buf[i] = cfg.beta1 * m_buf[i] + (1 - cfg.beta1) * p.grad
            v_buf[i] = cfg.beta2 * v_buf[i] + (1 - cfg.beta2) * p.grad**2
            m_hat = m_buf[i] / (1 - cfg.beta1 ** (step + 1))
            v_hat = v_buf[i] / (1 - cfg.beta2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat**0.5 + cfg.eps_adam)
            p.grad = 0

        if on_step is not None:
            on_step(step + 1, cfg.num_steps, loss.data)

    samples: list[str] = []
    for _ in range(cfg.num_samples):
        keys, values = (
            [[] for _ in range(cfg.n_layer)],
            [[] for _ in range(cfg.n_layer)],
        )
        token_id = bos
        sample: list[str] = []
        for pos_id in range(cfg.block_size):
            logits = gpt(
                token_id,
                pos_id,
                keys,
                values,
                state_dict=state_dict,
                n_layer=cfg.n_layer,
                n_head=cfg.n_head,
                head_dim=head_dim,
            )
            probs = softmax([logit / cfg.temperature for logit in logits])
            token_id = random.choices(
                range(vocab_size), weights=[p.data for p in probs]
            )[0]
            if token_id == bos:
                break
            sample.append(uchars[token_id])
        samples.append("".join(sample))
    return samples
