import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from agent_dynamic_system.config import SimulationConfig


@dataclass(frozen=True)
class Observation:
    step: int
    grass_biomass: float
    rabbits: int
    foxes: int
    grass_capacity_total: float


@dataclass(frozen=True)
class ControlAction:
    """Single intervention choice with independent non-negative channels."""

    cut_fraction: float = 0.0
    fertilizer_fraction: float = 0.0


class Controller:
    name = "controller"

    def reset(self, config: SimulationConfig, seed: int) -> None:
        return None

    def act(self, observation: Observation) -> ControlAction:
        raise NotImplementedError

    def act_with_state(
        self,
        observation: Observation,
        grass: Any,
        rabbits: Sequence[Any],
        foxes: Sequence[Any],
        config: SimulationConfig,
    ) -> ControlAction:
        return self.act(observation)


class NoControl(Controller):
    name = "baseline"

    def act(self, observation: Observation) -> ControlAction:
        return ControlAction()


class RuleBasedStabilityController(Controller):
    """Threshold policy focused on minimizing movement from initial targets."""

    name = "rule_based"

    def __init__(
        self,
        inner_band: float = 0.12,
        outer_band: float = 0.28,
        low_grass_fraction: float = 0.12,
        high_grass_fraction: float = 0.30,
        min_action_fraction: float = 0.03,
        moderate_action_fraction: float = 0.12,
        strong_action_fraction: float = 0.22,
    ) -> None:
        self.inner_band = inner_band
        self.outer_band = outer_band
        self.low_grass_fraction = low_grass_fraction
        self.high_grass_fraction = high_grass_fraction
        self.min_action_fraction = min_action_fraction
        self.moderate_action_fraction = moderate_action_fraction
        self.strong_action_fraction = strong_action_fraction

        self._target_rabbits = 1.0
        self._target_foxes = 1.0
        self._rabbit_safety = 0.0
        self._fox_safety = 0.0
        self._max_cut_fraction = 0.0
        self._max_fertilizer_fraction = 0.0

    def reset(self, config: SimulationConfig, seed: int) -> None:
        self._target_rabbits = float(config.initial_rabbits)
        self._target_foxes = float(config.initial_foxes)
        self._rabbit_safety = float(config.rabbit_safety_threshold)
        self._fox_safety = float(config.fox_safety_threshold)
        self._max_cut_fraction = config.max_cut_fraction
        self._max_fertilizer_fraction = config.max_fertilizer_fraction

    def act(self, observation: Observation) -> ControlAction:
        rabbit_ratio = observation.rabbits / max(self._target_rabbits, 1.0)
        fox_ratio = observation.foxes / max(self._target_foxes, 1.0)
        grass_fraction = observation.grass_biomass / max(
            observation.grass_capacity_total,
            1.0,
        )

        if (
            observation.rabbits <= self._rabbit_safety
            or observation.foxes <= self._fox_safety
        ):
            return self._fertilize_if_resource_limited(grass_fraction, strong=True)

        if self._inside_ideal_band(rabbit_ratio, fox_ratio):
            return ControlAction()

        rabbit_surplus = rabbit_ratio - 1.0
        rabbit_deficit = 1.0 - rabbit_ratio
        fox_surplus = fox_ratio - 1.0
        fox_deficit = 1.0 - fox_ratio

        if rabbit_surplus > self.outer_band:
            return self._cut(self.strong_action_fraction)
        if rabbit_surplus > self.inner_band and grass_fraction > self.low_grass_fraction:
            return self._cut(self.moderate_action_fraction)

        if fox_surplus > self.outer_band and observation.rabbits > 1.5 * self._rabbit_safety:
            return self._cut(self.strong_action_fraction)
        if fox_surplus > self.inner_band and observation.rabbits > 2.0 * self._rabbit_safety:
            return self._cut(self.moderate_action_fraction)

        if rabbit_deficit > self.outer_band or fox_deficit > self.outer_band:
            return self._fertilize_if_resource_limited(grass_fraction, strong=True)

        if (
            (rabbit_deficit > self.inner_band or fox_deficit > self.inner_band)
            and grass_fraction < self.high_grass_fraction
        ):
            return self._fertilize_if_resource_limited(grass_fraction, strong=False)

        if rabbit_surplus > self.inner_band or fox_surplus > self.outer_band:
            return self._cut(self.min_action_fraction)

        return ControlAction()

    def _inside_ideal_band(self, rabbit_ratio: float, fox_ratio: float) -> bool:
        lower = 1.0 - self.inner_band
        upper = 1.0 + self.inner_band
        return lower <= rabbit_ratio <= upper and lower <= fox_ratio <= upper

    def _cut(self, amount: float) -> ControlAction:
        return ControlAction(
            cut_fraction=min(max(amount, 0.0), self._max_cut_fraction),
        )

    def _fertilize_if_resource_limited(
        self,
        grass_fraction: float,
        strong: bool,
    ) -> ControlAction:
        if grass_fraction >= self.high_grass_fraction:
            return ControlAction()
        amount = self.strong_action_fraction if strong else self.moderate_action_fraction
        if grass_fraction >= self.low_grass_fraction:
            amount = min(amount, self.moderate_action_fraction)
        return ControlAction(
            fertilizer_fraction=min(max(amount, 0.0), self._max_fertilizer_fraction),
        )


class LookAheadMiniSimulationController(Controller):
    """Random shooting controller using short internal ecosystem rollouts."""

    name = "look_ahead"

    def __init__(
        self,
        planning_horizon: int = 10,
        random_rollouts: int = 24,
        minimum_improvement: float = 0.002,
    ) -> None:
        self.planning_horizon = planning_horizon
        self.random_rollouts = random_rollouts
        self.minimum_improvement = minimum_improvement
        self._target_rabbits = 1.0
        self._target_foxes = 1.0
        self._rabbit_safety = 0.0
        self._fox_safety = 0.0
        self._max_cut_fraction = 0.0
        self._max_fertilizer_fraction = 0.0
        self._rng = None

    def reset(self, config: SimulationConfig, seed: int) -> None:
        import numpy as np

        self._target_rabbits = float(config.initial_rabbits)
        self._target_foxes = float(config.initial_foxes)
        self._rabbit_safety = float(config.rabbit_safety_threshold)
        self._fox_safety = float(config.fox_safety_threshold)
        self._max_cut_fraction = config.max_cut_fraction
        self._max_fertilizer_fraction = config.max_fertilizer_fraction
        self._rng = np.random.default_rng(seed + 910_013)

    def act(self, observation: Observation) -> ControlAction:
        return ControlAction()

    def act_with_state(
        self,
        observation: Observation,
        grass: Any,
        rabbits: Sequence[Any],
        foxes: Sequence[Any],
        config: SimulationConfig,
    ) -> ControlAction:
        if self._rng is None:
            self.reset(config, 0)

        candidate_plans = self._candidate_plans()
        scored = [
            (
                self._score_plan(
                    plan,
                    observation,
                    config,
                ),
                plan[0],
            )
            for plan in candidate_plans
        ]
        scored.sort(key=lambda item: item[0])

        do_nothing_score = self._score_plan(
            [ControlAction()] * self.planning_horizon,
            observation,
            config,
        )
        best_score, best_action = scored[0]
        if do_nothing_score - best_score < self.minimum_improvement:
            return ControlAction()
        return best_action

    def _candidate_plans(self) -> List[List[ControlAction]]:
        plans = [
            [ControlAction()] * self.planning_horizon,
            [self._cut(0.08)] * self.planning_horizon,
            [self._cut(0.18)] * self.planning_horizon,
            [self._fertilize(0.08)] * self.planning_horizon,
            [self._fertilize(0.16)] * self.planning_horizon,
        ]
        for _ in range(self.random_rollouts):
            plans.append(self._random_plan())
        return plans

    def _random_plan(self) -> List[ControlAction]:
        plan = []
        for _ in range(self.planning_horizon):
            draw = float(self._rng.random())
            if draw < 0.30:
                plan.append(ControlAction())
            elif draw < 0.68:
                plan.append(self._cut(float(self._rng.uniform(0.02, 0.24))))
            else:
                plan.append(self._fertilize(float(self._rng.uniform(0.02, 0.18))))
        return plan

    def _score_plan(
        self,
        plan: Sequence[ControlAction],
        observation: Observation,
        config: SimulationConfig,
    ) -> float:
        import numpy as np

        grass_biomass = float(observation.grass_biomass)
        rabbits = float(observation.rabbits)
        foxes = float(observation.foxes)
        previous_rabbits = rabbits
        previous_foxes = foxes
        scores = []

        for action in plan:
            grass_biomass, rabbits, foxes = self._aggregate_step(
                grass_biomass,
                rabbits,
                foxes,
                action,
                config,
            )
            scores.append(
                self._state_instability(
                    rabbits=rabbits,
                    foxes=foxes,
                    previous_rabbits=previous_rabbits,
                    previous_foxes=previous_foxes,
                )
                + 0.03 * (action.cut_fraction + action.fertilizer_fraction)
            )
            previous_rabbits = rabbits
            previous_foxes = foxes

        return float(np.mean(scores))

    def _aggregate_step(
        self,
        grass_biomass: float,
        rabbits: float,
        foxes: float,
        action: ControlAction,
        config: SimulationConfig,
    ) -> tuple:
        grass_biomass = self._apply_aggregate_action(grass_biomass, action, config)
        grass_biomass += config.grass_regrowth_rate * (
            config.grass_capacity_total - grass_biomass
        )
        grass_biomass = min(max(grass_biomass, 0.0), config.grass_capacity_total)

        grass_fraction = grass_biomass / max(config.grass_capacity_total, 1.0)
        rabbit_resource_growth = 0.22 * rabbits * (grass_fraction - 0.10)
        rabbit_crowding = 0.10 * rabbits * max(0.0, rabbits / config.max_rabbits)
        predation = 0.018 * foxes * rabbits / max(self._target_rabbits, 1.0)
        next_rabbits = rabbits + rabbit_resource_growth - rabbit_crowding - predation

        prey_pressure = rabbits / max(self._target_rabbits, 1.0) - 0.65
        fox_growth = 0.12 * foxes * prey_pressure
        fox_crowding = 0.04 * foxes * max(0.0, foxes / config.max_foxes)
        next_foxes = foxes + fox_growth - fox_crowding

        return (
            grass_biomass,
            max(0.0, min(float(config.max_rabbits), next_rabbits)),
            max(0.0, min(float(config.max_foxes), next_foxes)),
        )

    def _apply_aggregate_action(
        self,
        grass_biomass: float,
        action: ControlAction,
        config: SimulationConfig,
    ) -> float:
        if action.cut_fraction > 0.0:
            return grass_biomass * (1.0 - min(action.cut_fraction, config.max_cut_fraction))
        if action.fertilizer_fraction > 0.0:
            fraction = min(action.fertilizer_fraction, config.max_fertilizer_fraction)
            return grass_biomass + fraction * (
                config.grass_capacity_total - grass_biomass
            )
        return grass_biomass

    def _state_instability(
        self,
        rabbits: float,
        foxes: float,
        previous_rabbits: float,
        previous_foxes: float,
    ) -> float:
        rabbit_deviation = abs(rabbits - self._target_rabbits) / self._target_rabbits
        fox_deviation = abs(foxes - self._target_foxes) / self._target_foxes
        rabbit_change = abs(rabbits - previous_rabbits) / self._target_rabbits
        fox_change = abs(foxes - previous_foxes) / self._target_foxes
        rabbit_safety = max(0.0, (self._rabbit_safety - rabbits) / self._rabbit_safety)
        fox_safety = max(0.0, (self._fox_safety - foxes) / self._fox_safety)

        return (
            0.30 * rabbit_deviation
            + 0.30 * fox_deviation
            + 0.15 * rabbit_change
            + 0.15 * fox_change
            + 0.05 * rabbit_safety
            + 0.05 * fox_safety
        )

    def _cut(self, amount: float) -> ControlAction:
        return ControlAction(
            cut_fraction=min(max(amount, 0.0), self._max_cut_fraction),
        )

    def _fertilize(self, amount: float) -> ControlAction:
        return ControlAction(
            fertilizer_fraction=min(max(amount, 0.0), self._max_fertilizer_fraction),
        )


class CodexAgentInLoopController(Controller):
    """Consult a local Codex CLI process at fixed simulation intervals."""

    name = "agent_in_loop"

    def __init__(
        self,
        decision_interval: int = 10,
        timeout_seconds: int = 120,
        codex_command: str = "codex",
    ) -> None:
        self.decision_interval = max(1, int(decision_interval))
        self.timeout_seconds = max(1, int(timeout_seconds))
        self.codex_command = codex_command
        self._target_rabbits = 1.0
        self._target_foxes = 1.0
        self._rabbit_safety = 0.0
        self._fox_safety = 0.0
        self._max_cut_fraction = 0.0
        self._max_fertilizer_fraction = 0.0
        self._history = []
        self._fallback = RuleBasedStabilityController()
        self._session_id: Optional[str] = None

    def reset(self, config: SimulationConfig, seed: int) -> None:
        self._target_rabbits = float(config.initial_rabbits)
        self._target_foxes = float(config.initial_foxes)
        self._rabbit_safety = float(config.rabbit_safety_threshold)
        self._fox_safety = float(config.fox_safety_threshold)
        self._max_cut_fraction = config.max_cut_fraction
        self._max_fertilizer_fraction = config.max_fertilizer_fraction
        self._history = []
        self._session_id = None
        self._fallback.reset(config, seed)

    def act(self, observation: Observation) -> ControlAction:
        return ControlAction()

    def act_with_state(
        self,
        observation: Observation,
        grass: Any,
        rabbits: Sequence[Any],
        foxes: Sequence[Any],
        config: SimulationConfig,
    ) -> ControlAction:
        self._record_history(observation)
        if observation.step % self.decision_interval != 0:
            return ControlAction()

        prompt = self._build_prompt(
            observation,
            config,
            continuing=self._session_id is not None,
        )
        try:
            decision = self._ask_codex(prompt)
        except Exception:
            return self._fallback.act(observation)
        return self._parse_decision(decision, observation)

    def _record_history(self, observation: Observation) -> None:
        self._history.append(
            {
                "step": observation.step,
                "grass_fraction": observation.grass_biomass
                / max(observation.grass_capacity_total, 1.0),
                "rabbits": observation.rabbits,
                "foxes": observation.foxes,
            }
        )
        self._history = self._history[-12:]

    def _build_prompt(
        self,
        observation: Observation,
        config: SimulationConfig,
        continuing: bool,
    ) -> str:
        state = {
            "current_step": observation.step,
            "consultation_interval_steps": self.decision_interval,
            "current_state": {
                "grass_biomass": observation.grass_biomass,
                "grass_capacity_total": observation.grass_capacity_total,
                "grass_fraction_of_capacity": observation.grass_biomass
                / max(observation.grass_capacity_total, 1.0),
                "rabbits": observation.rabbits,
                "foxes": observation.foxes,
            },
            "targets": {
                "rabbits": self._target_rabbits,
                "foxes": self._target_foxes,
            },
            "safety_floors": {
                "rabbits": self._rabbit_safety,
                "foxes": self._fox_safety,
            },
            "action_limits": {
                "max_cut_fraction": config.max_cut_fraction,
                "max_fertilizer_fraction": config.max_fertilizer_fraction,
            },
            "recent_history": self._history,
            "stability_metric": {
                "lower_is_better": True,
                "ideal": "keep rabbits and foxes close to initial targets",
                "penalizes": [
                    "rabbit and fox variability",
                    "mean absolute percent change from initial populations",
                    "RMSE deviation from initial populations",
                    "safety-floor breaches",
                    "extinction",
                ],
            },
        }
        session_context = (
            "This is a continuing update in the same active controller session. "
            "Use the new state below as the current truth.\n\n"
            if continuing
            else ""
        )
        prompt = (
            "You are the agent-in-the-loop controller for a grass/rabbit/fox "
            "agent-based simulation. Choose exactly one grass intervention for "
            "the current consultation step. The action will be applied once; "
            f"the simulation will ask again after {self.decision_interval} "
            "steps. Optimize for the provided lower-is-better instability "
            "metric.\n\n"
            "Allowed actions:\n"
            "- none: amount must be 0\n"
            "- cut: amount is fraction of standing grass removed, 0 to max_cut_fraction\n"
            "- fertilize: amount is fraction of grass capacity gap filled, 0 to max_fertilizer_fraction\n\n"
            "Return only JSON with this shape:\n"
            "{\"action\":\"none|cut|fertilize\",\"amount\":0.0,\"reason\":\"short reason\"}\n\n"
            f"Simulation state:\n{json.dumps(state, indent=2, sort_keys=True)}"
        )
        return session_context + prompt

    def _ask_codex(self, prompt: str) -> str:
        with tempfile.NamedTemporaryFile(
            mode="r",
            suffix=".txt",
            delete=False,
        ) as output_file:
            output_path = Path(output_file.name)

        command = self._codex_command(output_path)
        try:
            completed = subprocess.run(
                command,
                input=prompt,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_seconds,
                check=True,
            )
            self._remember_session_id(completed.stdout)
            return output_path.read_text().strip()
        finally:
            try:
                output_path.unlink()
            except OSError:
                pass

    def _codex_command(self, output_path: Path) -> List[str]:
        if self._session_id is None:
            return [
                self.codex_command,
                "exec",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--json",
                "--output-last-message",
                str(output_path),
                "--color",
                "never",
                "-",
            ]
        return [
            self.codex_command,
            "exec",
            "resume",
            "--skip-git-repo-check",
            "--json",
            "--output-last-message",
            str(output_path),
            self._session_id,
            "-",
        ]

    def _remember_session_id(self, stdout: str) -> None:
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            session_id = self._extract_session_id(event)
            if session_id is not None:
                self._session_id = session_id

    def _extract_session_id(self, value: Any) -> Optional[str]:
        if isinstance(value, dict):
            for key, item in value.items():
                key_lower = str(key).lower()
                if isinstance(item, str) and self._looks_like_session_field(key_lower):
                    match = re.fullmatch(self._uuid_pattern(), item.strip())
                    if match:
                        return item.strip()
                nested = self._extract_session_id(item)
                if nested is not None:
                    return nested
        if isinstance(value, list):
            for item in value:
                nested = self._extract_session_id(item)
                if nested is not None:
                    return nested
        return None

    @staticmethod
    def _looks_like_session_field(key: str) -> bool:
        return "session" in key or "conversation" in key or "thread" in key

    @staticmethod
    def _uuid_pattern() -> str:
        return (
            r"[0-9a-fA-F]{8}-"
            r"[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{12}"
        )

    def _parse_decision(
        self,
        response: str,
        observation: Observation,
    ) -> ControlAction:
        match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if not match:
            return self._fallback.act(observation)
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return self._fallback.act(observation)

        action = str(payload.get("action", "none")).strip().lower()
        try:
            amount = float(payload.get("amount", 0.0))
        except (TypeError, ValueError):
            amount = 0.0

        if action == "cut":
            return ControlAction(
                cut_fraction=min(max(amount, 0.0), self._max_cut_fraction),
            )
        if action == "fertilize":
            return ControlAction(
                fertilizer_fraction=min(
                    max(amount, 0.0),
                    self._max_fertilizer_fraction,
                ),
            )
        return ControlAction()


class CodexSteadyGrassController(CodexAgentInLoopController):
    name = "codex_steady"

    def __init__(
        self,
        decision_interval: int = 20,
        timeout_seconds: int = 120,
        codex_command: str = "codex",
    ) -> None:
        super().__init__(decision_interval, timeout_seconds, codex_command)


class CodexGuardianGrassController(CodexAgentInLoopController):
    name = "codex_guardian"

    def __init__(
        self,
        decision_interval: int = 20,
        timeout_seconds: int = 120,
        codex_command: str = "codex",
        guardian_threshold: float = 0.06,
    ) -> None:
        super().__init__(decision_interval, timeout_seconds, codex_command)
        self.guardian_threshold = guardian_threshold

    def act_with_state(
        self,
        observation: Observation,
        grass: Any,
        rabbits: Sequence[Any],
        foxes: Sequence[Any],
        config: SimulationConfig,
    ) -> ControlAction:
        self._record_history(observation)
        check_interval = max(1, self.decision_interval // 4)
        if observation.step % check_interval != 0 or not self._guardian_triggered():
            return ControlAction()

        prompt = self._build_prompt(
            observation,
            config,
            continuing=self._session_id is not None,
        )
        prompt = (
            "You were called because recent grass/rabbit/fox instability is "
            "worsening or a safety threshold is being approached.\n\n"
            + prompt
        )
        try:
            decision = self._ask_codex(prompt)
        except Exception:
            return self._fallback.act(observation)
        return self._parse_decision(decision, observation)

    def _guardian_triggered(self) -> bool:
        if len(self._history) < 4:
            return False
        scores = [self._history_instability(item) for item in self._history[-6:]]
        recent = float(sum(scores[-2:]) / 2.0)
        previous = float(sum(scores[:-2]) / max(len(scores[:-2]), 1))
        latest = self._history[-1]
        return (
            recent - previous > self.guardian_threshold
            or latest["rabbits"] < 1.35 * self._rabbit_safety
            or latest["foxes"] < 1.35 * self._fox_safety
        )

    def _history_instability(self, item: Dict[str, float]) -> float:
        rabbit_error = abs(item["rabbits"] - self._target_rabbits) / self._target_rabbits
        fox_error = abs(item["foxes"] - self._target_foxes) / self._target_foxes
        grass_error = abs(item["grass_fraction"] - 0.45) / 0.45
        return 0.40 * rabbit_error + 0.40 * fox_error + 0.20 * grass_error


class CodexControlAdvisedGrassController(CodexAgentInLoopController):
    name = "codex_control_advised"

    def __init__(
        self,
        decision_interval: int = 20,
        timeout_seconds: int = 120,
        codex_command: str = "codex",
    ) -> None:
        super().__init__(decision_interval, timeout_seconds, codex_command)

    def _build_prompt(
        self,
        observation: Observation,
        config: SimulationConfig,
        continuing: bool,
    ) -> str:
        prompt = super()._build_prompt(observation, config, continuing)
        advisory = self._control_advisory(observation)
        return (
            "A local control-theory advisory has been computed before this "
            "Codex decision. Use it as structured feedback, but you may override "
            "it if the state history suggests a better intervention.\n\n"
            f"Control advisory:\n{json.dumps(advisory, indent=2, sort_keys=True)}\n\n"
            + prompt
        )

    def _control_advisory(self, observation: Observation) -> Dict[str, Any]:
        rabbit_error = (self._target_rabbits - observation.rabbits) / self._target_rabbits
        fox_error = (self._target_foxes - observation.foxes) / self._target_foxes
        grass_fraction = observation.grass_biomass / max(observation.grass_capacity_total, 1.0)
        if observation.rabbits > self._target_rabbits * 1.15:
            action = "cut"
            amount = min(self._max_cut_fraction, 0.08 + 0.35 * abs(rabbit_error))
        elif observation.rabbits < self._target_rabbits * 0.85 or observation.foxes < self._target_foxes * 0.80:
            action = "fertilize"
            amount = min(self._max_fertilizer_fraction, 0.08 + 0.35 * max(rabbit_error, fox_error))
        elif grass_fraction < 0.20:
            action = "fertilize"
            amount = min(self._max_fertilizer_fraction, 0.10)
        else:
            action = "none"
            amount = 0.0
        return {
            "method": "proportional feedback around initial animal targets",
            "rabbit_error_target_minus_current": rabbit_error,
            "fox_error_target_minus_current": fox_error,
            "grass_fraction": grass_fraction,
            "recommended_action": action,
            "recommended_amount": amount,
        }



class PIGrassController(Controller):
    """PI-style feedback using grass as the manipulated ecosystem resource.

    The controller scores two independent actuators:
    - cutting removes a fraction of available grass biomass;
    - fertilizer increases grass toward carrying capacity.

    The simulation executes only one choice per step: do nothing, cut, or
    fertilize. Both action amounts are non-negative. They are not represented
    as positive and negative signs of a single control value.
    """

    name = "control_theory"

    def __init__(
        self,
        target_rabbits: Optional[float] = None,
        target_foxes: Optional[float] = None,
        support_kp: float = 0.14,
        support_ki: float = 0.006,
        suppression_kp: float = 0.22,
        suppression_ki: float = 0.014,
        derivative_gain: float = 0.08,
        deadband: float = 0.035,
        integral_limit: float = 3.0,
        grass_floor_fraction: float = 0.10,
        resource_floor_weight: float = 0.24,
    ) -> None:
        self.target_rabbits = target_rabbits
        self.target_foxes = target_foxes
        self.support_kp = support_kp
        self.support_ki = support_ki
        self.suppression_kp = suppression_kp
        self.suppression_ki = suppression_ki
        self.derivative_gain = derivative_gain
        self.deadband = deadband
        self.integral_limit = integral_limit
        self.grass_floor_fraction = grass_floor_fraction
        self.resource_floor_weight = resource_floor_weight

        self._max_cut_fraction = 0.0
        self._max_fertilizer_fraction = 0.0
        self._support_integral = 0.0
        self._suppression_integral = 0.0
        self._previous_rabbit_error = 0.0
        self._previous_fox_error = 0.0

    def reset(self, config: SimulationConfig, seed: int) -> None:
        if self.target_rabbits is None:
            self.target_rabbits = float(config.initial_rabbits)
        if self.target_foxes is None:
            self.target_foxes = float(config.initial_foxes)

        self._max_cut_fraction = config.max_cut_fraction
        self._max_fertilizer_fraction = config.max_fertilizer_fraction
        self._support_integral = 0.0
        self._suppression_integral = 0.0
        self._previous_rabbit_error = 0.0
        self._previous_fox_error = 0.0

    def act(self, observation: Observation) -> ControlAction:
        target_rabbits = max(float(self.target_rabbits or 1.0), 1.0)
        target_foxes = max(float(self.target_foxes or 1.0), 1.0)

        rabbit_error = (target_rabbits - observation.rabbits) / target_rabbits
        fox_error = (target_foxes - observation.foxes) / target_foxes
        rabbit_derivative = rabbit_error - self._previous_rabbit_error
        fox_derivative = fox_error - self._previous_fox_error

        rabbit_deficit = max(rabbit_error, 0.0)
        fox_deficit = max(fox_error, 0.0)
        rabbit_surplus = max(-rabbit_error, 0.0)
        fox_surplus = max(-fox_error, 0.0)
        grass_fraction = observation.grass_biomass / max(
            observation.grass_capacity_total,
            1.0,
        )
        grass_shortfall = max(
            0.0,
            (self.grass_floor_fraction - grass_fraction) / self.grass_floor_fraction,
        )

        support_signal = (
            self.resource_floor_weight * grass_shortfall
            + 0.25 * rabbit_deficit
            + 0.06 * fox_deficit
            + 0.40
            * self.derivative_gain
            * (max(rabbit_derivative, 0.0) + max(fox_derivative, 0.0))
        )
        suppression_signal = (
            0.96 * rabbit_surplus
            + 0.24 * fox_surplus
            + self.derivative_gain
            * (max(-rabbit_derivative, 0.0) + 0.35 * max(-fox_derivative, 0.0))
        )

        support_signal = max(0.0, support_signal - self.deadband)
        suppression_signal = max(0.0, suppression_signal - self.deadband)

        self._support_integral = self._bounded_integral(
            0.90 * self._support_integral + support_signal
        )
        self._suppression_integral = self._bounded_integral(
            0.92 * self._suppression_integral + suppression_signal
        )

        fertilizer = (
            self.support_kp * support_signal
            + self.support_ki * self._support_integral
        )
        cut = (
            self.suppression_kp * suppression_signal
            + self.suppression_ki * self._suppression_integral
        )

        self._previous_rabbit_error = rabbit_error
        self._previous_fox_error = fox_error

        return self._single_action_choice(
            cut=self._clip(cut, self._max_cut_fraction),
            fertilizer=self._clip(fertilizer, self._max_fertilizer_fraction),
        )

    def _bounded_integral(self, value: float) -> float:
        return self._clip(value, self.integral_limit)

    @staticmethod
    def _clip(value: float, upper: float) -> float:
        return min(max(value, 0.0), upper)

    @staticmethod
    def _single_action_choice(cut: float, fertilizer: float) -> ControlAction:
        if cut <= 0.0 and fertilizer <= 0.0:
            return ControlAction()
        if cut >= fertilizer:
            return ControlAction(cut_fraction=cut)
        return ControlAction(fertilizer_fraction=fertilizer)
