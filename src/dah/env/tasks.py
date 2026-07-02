"""Synthetic multi-session task generation.

Design goals for the benchmark:

* **Cross-session dependencies.** A query in session *s* frequently needs facts
  introduced in an *earlier* session. Since working context resets between
  sessions, these queries are unanswerable without persistent memory. This is
  the whole point: it is where episodic/semantic memory can pay off.
* **Topic-clustered distractors.** Facts embed near a topic centroid, so
  similarity-based retrieval will surface plausible-but-wrong neighbours. This
  makes retrieval *precision* (not just recall) matter, which is exactly the
  regime where naive retrieval can backfire.
* **Multi-fact queries.** Some queries need several facts at once (logical AND),
  so recall errors compound - a realistic stressor for memory systems.
* **Tunable context fill.** Filler tokens per turn control how fast the context
  window fills, letting us sweep the "long-horizon degradation" axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..types import Fact, Task, Turn


@dataclass
class TaskConfig:
    """All knobs for task generation. Defaults produce a moderately hard suite."""

    n_sessions: int = 6
    turns_per_session: int = 14
    query_fraction: float = 0.35  # fraction of turns that are queries

    n_topics: int = 12
    embed_dim: int = 32
    embed_noise: float = 0.35  # higher => retrieval is noisier (key ablation knob)

    facts_per_info_turn: int = 2
    fact_tokens: int = 20
    filler_tokens_per_turn: int = 55  # "noise" tokens that fill context
    query_tokens: int = 15

    max_required_facts: int = 2  # queries need 1..this many facts (AND)
    p_cross_session: float = 0.55  # prob a query depends on an earlier session

    def validate(self) -> None:
        if not (0.0 <= self.query_fraction <= 1.0):
            raise ValueError("query_fraction must be in [0, 1]")
        if self.n_topics < 2:
            raise ValueError("need >= 2 topics")
        if self.max_required_facts < 1:
            raise ValueError("max_required_facts must be >= 1")


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _make_topic_centroids(cfg: TaskConfig, rng: np.random.Generator) -> np.ndarray:
    """Random unit-norm topic centroids in R^embed_dim.

    Random (rather than orthogonal) centroids give a realistic similarity
    structure where some topics are closer than others.
    """
    c = rng.standard_normal((cfg.n_topics, cfg.embed_dim))
    return np.stack([_unit(row) for row in c])


def generate_task(cfg: TaskConfig, seed: int, task_id: str | None = None) -> Task:
    """Generate one multi-session task, fully determined by ``cfg`` and ``seed``."""
    cfg.validate()
    rng = np.random.default_rng(seed)
    centroids = _make_topic_centroids(cfg, rng)

    facts: dict[str, Fact] = {}
    facts_by_topic: dict[int, list[str]] = {t: [] for t in range(cfg.n_topics)}
    introduced: list[str] = []  # fact ids introduced so far (in order)
    introduced_by_session: list[list[str]] = []

    sessions: list[list[Turn]] = []
    global_turn = 0
    fact_counter = 0

    for s in range(cfg.n_sessions):
        introduced_by_session.append([])
        turns: list[Turn] = []
        for _ in range(cfg.turns_per_session):
            make_query = (
                rng.random() < cfg.query_fraction
                and len(introduced) >= 1
                # ensure at least one info turn precedes the first query
                and any(f for f in introduced)
            )
            if not make_query:
                # info turn: introduce new facts
                new_ids: list[str] = []
                for _k in range(cfg.facts_per_info_turn):
                    topic = int(rng.integers(cfg.n_topics))
                    emb = _unit(centroids[topic] + cfg.embed_noise * rng.standard_normal(cfg.embed_dim))
                    fid = f"f{fact_counter}"
                    fact_counter += 1
                    fact = Fact(
                        id=fid,
                        topic=topic,
                        content=f"[topic {topic}] fact {fid}",
                        embedding=emb,
                        tokens=cfg.fact_tokens,
                        salience=float(rng.uniform(0.5, 1.0)),
                        session_introduced=s,
                        turn_introduced=global_turn,
                    )
                    facts[fid] = fact
                    facts_by_topic[topic].append(fid)
                    introduced.append(fid)
                    introduced_by_session[s].append(fid)
                    new_ids.append(fid)
                tokens = cfg.filler_tokens_per_turn + sum(facts[i].tokens for i in new_ids)
                turns.append(
                    Turn(
                        index=global_turn,
                        session=s,
                        kind="info",
                        text=f"info turn introducing {new_ids}",
                        tokens=tokens,
                        fact_ids=new_ids,
                    )
                )
            else:
                required = _choose_required(cfg, rng, introduced, introduced_by_session, s)
                qid = f"q{global_turn}"
                turns.append(
                    Turn(
                        index=global_turn,
                        session=s,
                        kind="query",
                        text=f"query {qid} needs {required}",
                        tokens=cfg.query_tokens,
                        required_fact_ids=required,
                        query_id=qid,
                    )
                )
            global_turn += 1
        sessions.append(turns)

    task = Task(
        id=task_id or f"task-{seed}",
        sessions=sessions,
        facts=facts,
        meta={"config": cfg.__dict__.copy(), "seed": seed},
    )
    return task


def _choose_required(
    cfg: TaskConfig,
    rng: np.random.Generator,
    introduced: list[str],
    introduced_by_session: list[list[str]],
    current_session: int,
) -> list[str]:
    """Pick the fact(s) a query needs, biasing toward cross-session dependencies."""
    k = 1 + int(rng.integers(cfg.max_required_facts))  # 1..max
    k = min(k, len(introduced))
    chosen: list[str] = []

    earlier_pool = [
        f for si in range(current_session) for f in introduced_by_session[si]
    ]
    want_cross = rng.random() < cfg.p_cross_session and len(earlier_pool) > 0
    if want_cross:
        chosen.append(str(rng.choice(earlier_pool)))

    while len(chosen) < k:
        cand = str(rng.choice(introduced))
        if cand not in chosen:
            chosen.append(cand)
    return chosen


def generate_suite(cfg: TaskConfig, n_tasks: int, base_seed: int = 0) -> list[Task]:
    """Generate ``n_tasks`` independent tasks with deterministic per-task seeds."""
    return [generate_task(cfg, seed=base_seed + i, task_id=f"task-{base_seed + i}") for i in range(n_tasks)]
