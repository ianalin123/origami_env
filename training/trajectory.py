from dataclasses import dataclass, field


@dataclass
class Step:
    prompt: str
    completion: str
    reward: float
    done: bool
    state_hash: int
    reward_breakdown: dict = field(default_factory=dict)
    log_prob: float = 0.0


@dataclass
class Trajectory:
    task: str
    steps: list[Step] = field(default_factory=list)

    def add_step(self, **kwargs) -> None:
        self.steps.append(Step(**kwargs))

    @property
    def total_reward(self) -> float:
        return sum(s.reward for s in self.steps)

    @property
    def length(self) -> int:
        return len(self.steps)
