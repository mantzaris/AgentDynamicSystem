from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from agent_dynamic_system.benchmark import GenericRun, GenericScenario, MetricSpec


@dataclass(frozen=True)
class SmartGridConfig:
    steps: int = 500
    base_load: float = 1000.0
    renewable_capacity: float = 620.0
    dispatchable_capacity: float = 620.0
    battery_capacity: float = 420.0
    initial_battery_charge: float = 0.55
    load_noise: float = 45.0
    weather_shock_probability: float = 0.035
    max_demand_response: float = 0.18
    max_battery_dispatch: float = 0.34
    max_backup_generation: float = 0.30
    max_renewable_curtailment: float = 0.25
    target_imbalance: float = 0.015
    target_battery_charge: float = 0.55
    target_price: float = 1.0
    outage_safety_max: float = 0.08


@dataclass(frozen=True)
class GridAction:
    demand_response: float = 0.0
    battery_dispatch: float = 0.0
    backup_generation: float = 0.0
    renewable_curtailment: float = 0.0


@dataclass(frozen=True)
class GridObservation:
    step: int
    imbalance_fraction: float
    battery_charge: float
    price_index: float
    outage_fraction: float
    renewable_fraction: float


GridPolicy = Callable[[GridObservation, SmartGridConfig], GridAction]


def build_smart_grid_scenarios(config: SmartGridConfig) -> List[GenericScenario]:
    return [
        GenericScenario("baseline", lambda seed: simulate_smart_grid(config, seed, no_grid_control)),
        GenericScenario("control_theory", lambda seed: simulate_smart_grid(config, seed, grid_feedback_control())),
        GenericScenario("rule_based", lambda seed: simulate_smart_grid(config, seed, grid_rule_based)),
        GenericScenario("look_ahead", lambda seed: simulate_smart_grid(config, seed, grid_look_ahead(config))),
    ]


def smart_grid_metric_spec(config: SmartGridConfig) -> MetricSpec:
    return MetricSpec(
        target={
            "imbalance_fraction": config.target_imbalance,
            "battery_charge": config.target_battery_charge,
            "price_index": config.target_price,
            "outage_fraction": 0.01,
        },
        safety_min={"battery_charge": 0.12},
        safety_max={
            "imbalance_fraction": 0.14,
            "price_index": 1.75,
            "outage_fraction": config.outage_safety_max,
        },
        cost_weight=0.045,
    )


def smart_grid_labels() -> Tuple[Dict[str, str], Dict[str, str]]:
    variables = {
        "imbalance_fraction": "Supply-demand imbalance",
        "battery_charge": "Battery charge",
        "price_index": "Price index",
        "outage_fraction": "Outage fraction",
    }
    actions = {
        "demand_response": "Demand response",
        "battery_dispatch": "Battery dispatch",
        "backup_generation": "Backup generation",
        "renewable_curtailment": "Renewable curtailment",
    }
    return variables, actions


def simulate_smart_grid(
    config: SmartGridConfig,
    seed: int,
    policy: GridPolicy,
) -> GenericRun:
    rng = np.random.default_rng(seed)
    battery = config.initial_battery_charge
    price = config.target_price
    outage = 0.0

    time = np.arange(config.steps + 1)
    variables = {name: np.zeros(config.steps + 1) for name in smart_grid_labels()[0]}
    actions = {name: np.zeros(config.steps + 1) for name in smart_grid_labels()[1]}
    load, renewable = _exogenous_power(0, config, rng)
    imbalance = abs(load - renewable - 0.78 * config.dispatchable_capacity) / config.base_load
    _record(variables, 0, imbalance, battery, price, outage)

    for step in range(1, config.steps + 1):
        observation = GridObservation(
            step=step - 1,
            imbalance_fraction=variables["imbalance_fraction"][step - 1],
            battery_charge=battery,
            price_index=price,
            outage_fraction=outage,
            renewable_fraction=renewable / max(load, 1.0),
        )
        action = _single_grid_action(policy(observation, config), config)
        load, renewable = _exogenous_power(step, config, rng)
        load *= 1.0 - action.demand_response
        renewable *= 1.0 - action.renewable_curtailment
        dispatchable = 0.78 * config.dispatchable_capacity + action.backup_generation * config.dispatchable_capacity
        battery_power = action.battery_dispatch * config.battery_capacity
        available_battery_power = min(battery_power, battery * config.battery_capacity)
        supply = renewable + dispatchable + available_battery_power
        shortage = max(0.0, load - supply)
        surplus = max(0.0, supply - load)
        battery = battery - available_battery_power / config.battery_capacity
        battery += min(surplus, config.battery_capacity * 0.08) / config.battery_capacity
        battery += 0.008 * max(0.0, 0.55 - battery)
        battery = float(np.clip(battery, 0.0, 1.0))
        imbalance = abs(load - supply) / max(load, 1.0)
        outage = 0.86 * outage + 0.14 * min(1.0, shortage / max(load, 1.0))
        price = _update_price(price, imbalance, outage, action, config)

        actions["demand_response"][step] = action.demand_response
        actions["battery_dispatch"][step] = action.battery_dispatch
        actions["backup_generation"][step] = action.backup_generation
        actions["renewable_curtailment"][step] = action.renewable_curtailment
        _record(variables, step, imbalance, battery, price, outage)

    return GenericRun(time=time, variables=variables, actions=actions)


def no_grid_control(observation: GridObservation, config: SmartGridConfig) -> GridAction:
    return GridAction()


def grid_feedback_control() -> GridPolicy:
    integral = 0.0

    def policy(observation: GridObservation, config: SmartGridConfig) -> GridAction:
        nonlocal integral
        error = observation.imbalance_fraction - config.target_imbalance
        integral = float(np.clip(0.90 * integral + error, -0.7, 0.7))
        if observation.outage_fraction > 0.035:
            return GridAction(backup_generation=min(config.max_backup_generation, 0.08 + 1.4 * observation.outage_fraction))
        if observation.battery_charge > 0.24 and error > 0.045:
            return GridAction(battery_dispatch=min(config.max_battery_dispatch, 1.5 * error + 0.25 * integral))
        if error > 0.060:
            return GridAction(demand_response=min(config.max_demand_response, 0.55 * error))
        if observation.price_index < 0.78 and observation.renewable_fraction > 1.0:
            return GridAction(renewable_curtailment=min(config.max_renewable_curtailment, 0.08))
        return GridAction()

    return policy


def grid_rule_based(observation: GridObservation, config: SmartGridConfig) -> GridAction:
    if observation.outage_fraction > 0.050:
        return GridAction(backup_generation=config.max_backup_generation)
    if observation.imbalance_fraction > 0.10 and observation.battery_charge > 0.28:
        return GridAction(battery_dispatch=config.max_battery_dispatch)
    if observation.imbalance_fraction > 0.08:
        return GridAction(demand_response=config.max_demand_response * 0.75)
    if observation.renewable_fraction > 1.25 and observation.price_index < 0.80:
        return GridAction(renewable_curtailment=config.max_renewable_curtailment * 0.50)
    return GridAction()


def grid_look_ahead(config: SmartGridConfig) -> GridPolicy:
    candidates = [
        GridAction(),
        GridAction(demand_response=0.08),
        GridAction(demand_response=0.16),
        GridAction(battery_dispatch=0.18),
        GridAction(battery_dispatch=0.32),
        GridAction(backup_generation=0.16),
        GridAction(renewable_curtailment=0.12),
    ]

    def policy(observation: GridObservation, config: SmartGridConfig) -> GridAction:
        scored = [(_score_grid_action(observation, action, config), action) for action in candidates]
        scored.sort(key=lambda item: item[0])
        return scored[0][1]

    return policy


def _score_grid_action(
    observation: GridObservation,
    action: GridAction,
    config: SmartGridConfig,
) -> float:
    imbalance = observation.imbalance_fraction
    battery = observation.battery_charge
    price = observation.price_index
    outage = observation.outage_fraction
    scores = []
    for _ in range(10):
        shortage_pressure = max(0.0, imbalance - action.demand_response - 0.35 * action.backup_generation)
        battery_effect = min(battery, action.battery_dispatch)
        imbalance = max(0.0, 0.75 * imbalance + 0.04 - 0.55 * battery_effect - 0.50 * action.backup_generation - 0.20 * action.demand_response)
        battery = float(np.clip(battery - 0.30 * action.battery_dispatch + 0.025 * (0.55 - battery), 0.0, 1.0))
        outage = max(0.0, 0.84 * outage + 0.16 * shortage_pressure)
        price = float(np.clip(0.88 * price + 0.12 + 1.6 * outage + 0.6 * imbalance + 0.08 * action.backup_generation, 0.45, 2.5))
        scores.append(
            2.0 * abs(imbalance - config.target_imbalance)
            + abs(battery - config.target_battery_charge)
            + 1.4 * abs(price - config.target_price)
            + 2.4 * outage
            + 0.04 * sum(action.__dict__.values())
        )
    return float(np.mean(scores))


def _single_grid_action(action: GridAction, config: SmartGridConfig) -> GridAction:
    amounts = {
        "demand_response": min(max(action.demand_response, 0.0), config.max_demand_response),
        "battery_dispatch": min(max(action.battery_dispatch, 0.0), config.max_battery_dispatch),
        "backup_generation": min(max(action.backup_generation, 0.0), config.max_backup_generation),
        "renewable_curtailment": min(max(action.renewable_curtailment, 0.0), config.max_renewable_curtailment),
    }
    choice = max(amounts, key=amounts.get)
    if amounts[choice] <= 0.0:
        return GridAction()
    return GridAction(**{choice: amounts[choice]})


def _exogenous_power(
    step: int,
    config: SmartGridConfig,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    day = 2.0 * np.pi * (step % 48) / 48.0
    load = config.base_load * (1.0 + 0.18 * np.sin(day - 1.0) + 0.08 * np.sin(2.0 * day))
    load += rng.normal(0.0, config.load_noise)
    renewable = config.renewable_capacity * max(0.0, np.sin(day)) * rng.uniform(0.72, 1.08)
    renewable += 120.0 * rng.random()
    if rng.random() < config.weather_shock_probability:
        renewable *= rng.uniform(0.15, 0.55)
        load *= rng.uniform(1.06, 1.20)
    return max(250.0, float(load)), max(0.0, float(renewable))


def _update_price(
    price: float,
    imbalance: float,
    outage: float,
    action: GridAction,
    config: SmartGridConfig,
) -> float:
    next_price = (
        0.86 * price
        + 0.14 * config.target_price
        + 0.65 * imbalance
        + 1.15 * outage
        + 0.10 * action.backup_generation
        - 0.08 * action.renewable_curtailment
    )
    return float(np.clip(next_price, 0.45, 2.50))


def _record(
    variables: Dict[str, np.ndarray],
    step: int,
    imbalance: float,
    battery: float,
    price: float,
    outage: float,
) -> None:
    variables["imbalance_fraction"][step] = float(imbalance)
    variables["battery_charge"][step] = float(battery)
    variables["price_index"][step] = float(price)
    variables["outage_fraction"][step] = float(outage)
