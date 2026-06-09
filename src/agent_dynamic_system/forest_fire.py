from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from agent_dynamic_system.benchmark import GenericRun, GenericScenario, MetricSpec


@dataclass(frozen=True)
class ForestFireConfig:
    width: int = 55
    height: int = 55
    steps: int = 500
    initial_ignitions: int = 8
    lightning_probability: float = 0.0012
    base_spread_probability: float = 0.23
    wind_east_bias: float = 0.055
    fuel_regrowth: float = 0.0015
    initial_fuel_min: float = 0.55
    initial_fuel_max: float = 1.0
    max_water_fraction: float = 0.32
    max_firebreak_fraction: float = 0.02
    max_controlled_burn_fraction: float = 0.018
    target_burning_fraction: float = 0.012
    target_burned_fraction: float = 0.16
    burned_safety_max: float = 0.42
    burning_safety_max: float = 0.08

    @property
    def cell_count(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class FireAction:
    water_fraction: float = 0.0
    firebreak_fraction: float = 0.0
    controlled_burn_fraction: float = 0.0


@dataclass(frozen=True)
class FireObservation:
    step: int
    burning_fraction: float
    burned_fraction: float
    fuel_fraction: float


FirePolicy = Callable[[FireObservation, ForestFireConfig], FireAction]


def build_forest_fire_scenarios(config: ForestFireConfig) -> List[GenericScenario]:
    return [
        GenericScenario("baseline", lambda seed: simulate_forest_fire(config, seed, no_fire_control)),
        GenericScenario("control_theory", lambda seed: simulate_forest_fire(config, seed, fire_feedback_control())),
        GenericScenario("rule_based", lambda seed: simulate_forest_fire(config, seed, fire_rule_based)),
        GenericScenario("look_ahead", lambda seed: simulate_forest_fire(config, seed, fire_look_ahead(config))),
    ]


def forest_fire_metric_spec(config: ForestFireConfig) -> MetricSpec:
    return MetricSpec(
        target={
            "burning_fraction": config.target_burning_fraction,
            "burned_fraction": config.target_burned_fraction,
            "fuel_fraction": 0.62,
        },
        safety_min={"fuel_fraction": 0.20},
        safety_max={
            "burning_fraction": config.burning_safety_max,
            "burned_fraction": config.burned_safety_max,
        },
        cost_weight=0.04,
    )


def forest_fire_labels() -> Tuple[Dict[str, str], Dict[str, str]]:
    variables = {
        "burning_fraction": "Burning area fraction",
        "burned_fraction": "Cumulative burned fraction",
        "fuel_fraction": "Available fuel fraction",
    }
    actions = {
        "water_fraction": "Water-drop fraction",
        "firebreak_fraction": "Firebreak fraction",
        "controlled_burn_fraction": "Controlled-burn fraction",
    }
    return variables, actions


def simulate_forest_fire(
    config: ForestFireConfig,
    seed: int,
    policy: FirePolicy,
) -> GenericRun:
    rng = np.random.default_rng(seed)
    fuel = rng.uniform(
        config.initial_fuel_min,
        config.initial_fuel_max,
        size=(config.height, config.width),
    )
    burning = np.zeros((config.height, config.width), dtype=bool)
    burned = np.zeros((config.height, config.width), dtype=bool)
    for _ in range(config.initial_ignitions):
        burning[int(rng.integers(config.height)), int(rng.integers(config.width))] = True

    time = np.arange(config.steps + 1)
    series = {
        "burning_fraction": np.zeros(config.steps + 1),
        "burned_fraction": np.zeros(config.steps + 1),
        "fuel_fraction": np.zeros(config.steps + 1),
    }
    actions = {
        "water_fraction": np.zeros(config.steps + 1),
        "firebreak_fraction": np.zeros(config.steps + 1),
        "controlled_burn_fraction": np.zeros(config.steps + 1),
    }
    _record(series, 0, burning, burned, fuel, config)

    for step in range(1, config.steps + 1):
        observation = FireObservation(
            step=step - 1,
            burning_fraction=series["burning_fraction"][step - 1],
            burned_fraction=series["burned_fraction"][step - 1],
            fuel_fraction=series["fuel_fraction"][step - 1],
        )
        action = _single_fire_action(policy(observation, config), config)
        _apply_fire_action(action, burning, burned, fuel, config, rng)
        _spread_fire(burning, burned, fuel, config, rng)
        _regrow_fuel(fuel, burning, burned, config)
        actions["water_fraction"][step] = action.water_fraction
        actions["firebreak_fraction"][step] = action.firebreak_fraction
        actions["controlled_burn_fraction"][step] = action.controlled_burn_fraction
        _record(series, step, burning, burned, fuel, config)

    return GenericRun(time=time, variables=series, actions=actions)


def no_fire_control(observation: FireObservation, config: ForestFireConfig) -> FireAction:
    return FireAction()


def fire_feedback_control() -> FirePolicy:
    integral = 0.0

    def policy(observation: FireObservation, config: ForestFireConfig) -> FireAction:
        nonlocal integral
        burn_error = observation.burning_fraction - config.target_burning_fraction
        damage_error = observation.burned_fraction - config.target_burned_fraction
        integral = float(np.clip(0.94 * integral + burn_error, -0.25, 0.25))
        response = 2.7 * max(0.0, burn_error) + 0.9 * max(0.0, damage_error) + 0.7 * max(0.0, integral)
        if observation.burning_fraction > config.burning_safety_max * 0.55:
            return FireAction(water_fraction=min(config.max_water_fraction, response))
        if observation.fuel_fraction > 0.70 and observation.burned_fraction < config.target_burned_fraction:
            return FireAction(controlled_burn_fraction=min(config.max_controlled_burn_fraction, 0.006 + 0.02 * response))
        if response > 0.03:
            return FireAction(firebreak_fraction=min(config.max_firebreak_fraction, 0.004 + 0.025 * response))
        return FireAction()

    return policy


def fire_rule_based(observation: FireObservation, config: ForestFireConfig) -> FireAction:
    if observation.burning_fraction > config.burning_safety_max * 0.65:
        return FireAction(water_fraction=config.max_water_fraction)
    if observation.burning_fraction > config.target_burning_fraction * 1.8:
        return FireAction(firebreak_fraction=config.max_firebreak_fraction)
    if observation.fuel_fraction > 0.74 and observation.burned_fraction < config.target_burned_fraction * 0.8:
        return FireAction(controlled_burn_fraction=config.max_controlled_burn_fraction * 0.65)
    return FireAction()


def fire_look_ahead(config: ForestFireConfig) -> FirePolicy:
    candidate_actions = [
        FireAction(),
        FireAction(water_fraction=0.14),
        FireAction(water_fraction=0.28),
        FireAction(firebreak_fraction=0.010),
        FireAction(firebreak_fraction=0.018),
        FireAction(controlled_burn_fraction=0.010),
    ]

    def policy(observation: FireObservation, config: ForestFireConfig) -> FireAction:
        scored = [
            (_score_fire_action(observation, action, config), action)
            for action in candidate_actions
        ]
        scored.sort(key=lambda item: item[0])
        return scored[0][1]

    return policy


def _score_fire_action(
    observation: FireObservation,
    action: FireAction,
    config: ForestFireConfig,
) -> float:
    burning = observation.burning_fraction
    burned = observation.burned_fraction
    fuel = observation.fuel_fraction
    scores = []
    for _ in range(10):
        burning, burned, fuel = _aggregate_fire_step(burning, burned, fuel, action, config)
        scores.append(
            2.0 * max(0.0, burning - config.target_burning_fraction)
            + 1.4 * max(0.0, burned - config.target_burned_fraction)
            + 0.5 * abs(fuel - 0.62)
            + 0.04 * (action.water_fraction + action.firebreak_fraction + action.controlled_burn_fraction)
        )
    return float(np.mean(scores))


def _aggregate_fire_step(
    burning: float,
    burned: float,
    fuel: float,
    action: FireAction,
    config: ForestFireConfig,
) -> Tuple[float, float, float]:
    burning *= 1.0 - min(action.water_fraction, config.max_water_fraction)
    fuel *= 1.0 - min(action.controlled_burn_fraction, config.max_controlled_burn_fraction)
    spread = config.base_spread_probability * fuel * burning * (1.0 - 8.0 * action.firebreak_fraction)
    new_burning = max(0.0, min(1.0 - burned, 0.58 * burning + spread + config.lightning_probability * fuel))
    new_burned = min(1.0, burned + 0.52 * burning + 0.7 * action.controlled_burn_fraction)
    new_fuel = min(1.0, max(0.0, fuel - 0.48 * burning - action.controlled_burn_fraction + config.fuel_regrowth * (1.0 - fuel)))
    return new_burning, new_burned, new_fuel


def _single_fire_action(action: FireAction, config: ForestFireConfig) -> FireAction:
    amounts = {
        "water": min(max(action.water_fraction, 0.0), config.max_water_fraction),
        "firebreak": min(max(action.firebreak_fraction, 0.0), config.max_firebreak_fraction),
        "controlled_burn": min(max(action.controlled_burn_fraction, 0.0), config.max_controlled_burn_fraction),
    }
    choice = max(amounts, key=amounts.get)
    if amounts[choice] <= 0.0:
        return FireAction()
    if choice == "water":
        return FireAction(water_fraction=amounts[choice])
    if choice == "firebreak":
        return FireAction(firebreak_fraction=amounts[choice])
    return FireAction(controlled_burn_fraction=amounts[choice])


def _apply_fire_action(
    action: FireAction,
    burning: np.ndarray,
    burned: np.ndarray,
    fuel: np.ndarray,
    config: ForestFireConfig,
    rng: np.random.Generator,
) -> None:
    if action.water_fraction > 0.0 and np.any(burning):
        indices = np.argwhere(burning)
        count = max(1, int(action.water_fraction * len(indices)))
        chosen = indices[rng.choice(len(indices), size=min(count, len(indices)), replace=False)]
        burning[chosen[:, 0], chosen[:, 1]] = False
    elif action.firebreak_fraction > 0.0:
        candidates = np.argwhere((fuel > 0.0) & ~burning & ~burned)
        count = min(len(candidates), max(1, int(action.firebreak_fraction * config.cell_count)))
        if count:
            chosen = candidates[rng.choice(len(candidates), size=count, replace=False)]
            fuel[chosen[:, 0], chosen[:, 1]] = 0.0
    elif action.controlled_burn_fraction > 0.0:
        candidates = np.argwhere((fuel > 0.35) & ~burning & ~burned)
        count = min(len(candidates), max(1, int(action.controlled_burn_fraction * config.cell_count)))
        if count:
            chosen = candidates[rng.choice(len(candidates), size=count, replace=False)]
            fuel[chosen[:, 0], chosen[:, 1]] *= 0.25
            burned[chosen[:, 0], chosen[:, 1]] = True


def _spread_fire(
    burning: np.ndarray,
    burned: np.ndarray,
    fuel: np.ndarray,
    config: ForestFireConfig,
    rng: np.random.Generator,
) -> None:
    neighbor_pressure = (
        np.roll(burning, 1, axis=0)
        + np.roll(burning, -1, axis=0)
        + np.roll(burning, 1, axis=1)
        + np.roll(burning, -1, axis=1)
    )
    east_pressure = np.roll(burning, 1, axis=1)
    spread_probability = (
        config.base_spread_probability * neighbor_pressure
        + config.wind_east_bias * east_pressure
        + config.lightning_probability
    ) * fuel
    new_burning = (rng.random(burning.shape) < spread_probability) & ~burned & (fuel > 0.08)
    burned |= burning
    fuel[burning] = 0.0
    burning[:] = new_burning


def _regrow_fuel(
    fuel: np.ndarray,
    burning: np.ndarray,
    burned: np.ndarray,
    config: ForestFireConfig,
) -> None:
    fuel += config.fuel_regrowth * (1.0 - fuel)
    fuel[burning | burned] *= 0.995
    np.clip(fuel, 0.0, 1.0, out=fuel)


def _record(
    series: Dict[str, np.ndarray],
    step: int,
    burning: np.ndarray,
    burned: np.ndarray,
    fuel: np.ndarray,
    config: ForestFireConfig,
) -> None:
    series["burning_fraction"][step] = float(np.mean(burning))
    series["burned_fraction"][step] = float(np.mean(burned))
    series["fuel_fraction"][step] = float(np.mean(fuel))
