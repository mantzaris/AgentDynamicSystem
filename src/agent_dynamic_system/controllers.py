from dataclasses import dataclass
from typing import Optional

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
