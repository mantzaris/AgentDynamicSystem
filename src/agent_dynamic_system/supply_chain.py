from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from agent_dynamic_system.benchmark import GenericRun, GenericScenario, MetricSpec


@dataclass(frozen=True)
class SupplyChainConfig:
    steps: int = 500
    suppliers: int = 18
    initial_inventory: float = 1150.0
    target_inventory: float = 1050.0
    initial_price: float = 1.0
    target_price: float = 1.0
    base_demand: float = 105.0
    demand_noise: float = 12.0
    demand_shock_probability: float = 0.035
    demand_shock_scale: float = 70.0
    base_capacity_per_supplier: float = 7.0
    supplier_recovery_rate: float = 0.020
    supplier_stress_decay: float = 0.035
    max_inventory_release: float = 0.30
    max_production_boost: float = 0.28
    max_rationing: float = 0.28
    max_supplier_subsidy: float = 0.26


@dataclass(frozen=True)
class SupplyAction:
    inventory_release: float = 0.0
    production_boost: float = 0.0
    rationing: float = 0.0
    supplier_subsidy: float = 0.0


@dataclass(frozen=True)
class SupplyObservation:
    step: int
    inventory: float
    unmet_demand_fraction: float
    price: float
    supplier_health: float
    demand: float


SupplyPolicy = Callable[[SupplyObservation, SupplyChainConfig], SupplyAction]


def build_supply_chain_scenarios(config: SupplyChainConfig) -> List[GenericScenario]:
    return [
        GenericScenario("baseline", lambda seed: simulate_supply_chain(config, seed, no_supply_control)),
        GenericScenario("control_theory", lambda seed: simulate_supply_chain(config, seed, supply_feedback_control())),
        GenericScenario("rule_based", lambda seed: simulate_supply_chain(config, seed, supply_rule_based)),
        GenericScenario("look_ahead", lambda seed: simulate_supply_chain(config, seed, supply_look_ahead(config))),
    ]


def supply_chain_metric_spec(config: SupplyChainConfig) -> MetricSpec:
    return MetricSpec(
        target={
            "inventory": config.target_inventory,
            "price": config.target_price,
            "unmet_demand_fraction": 0.02,
            "supplier_health": 0.92,
        },
        safety_min={
            "inventory": 240.0,
            "supplier_health": 0.45,
        },
        safety_max={
            "price": 1.65,
            "unmet_demand_fraction": 0.28,
            "inventory": 1800.0,
        },
        cost_weight=0.045,
    )


def supply_chain_labels() -> Tuple[Dict[str, str], Dict[str, str]]:
    variables = {
        "inventory": "Inventory",
        "unmet_demand_fraction": "Unmet demand fraction",
        "price": "Price index",
        "supplier_health": "Supplier health",
    }
    actions = {
        "inventory_release": "Inventory release",
        "production_boost": "Production boost",
        "rationing": "Demand rationing",
        "supplier_subsidy": "Supplier subsidy",
    }
    return variables, actions


def simulate_supply_chain(
    config: SupplyChainConfig,
    seed: int,
    policy: SupplyPolicy,
) -> GenericRun:
    rng = np.random.default_rng(seed)
    supplier_healths = np.clip(rng.normal(0.92, 0.035, size=config.suppliers), 0.65, 1.0)
    inventory = config.initial_inventory
    price = config.initial_price
    demand = config.base_demand

    time = np.arange(config.steps + 1)
    variables = {
        "inventory": np.zeros(config.steps + 1),
        "unmet_demand_fraction": np.zeros(config.steps + 1),
        "price": np.zeros(config.steps + 1),
        "supplier_health": np.zeros(config.steps + 1),
    }
    actions = {
        "inventory_release": np.zeros(config.steps + 1),
        "production_boost": np.zeros(config.steps + 1),
        "rationing": np.zeros(config.steps + 1),
        "supplier_subsidy": np.zeros(config.steps + 1),
    }
    variables["inventory"][0] = inventory
    variables["unmet_demand_fraction"][0] = 0.0
    variables["price"][0] = price
    variables["supplier_health"][0] = float(np.mean(supplier_healths))

    for step in range(1, config.steps + 1):
        observation = SupplyObservation(
            step=step - 1,
            inventory=inventory,
            unmet_demand_fraction=variables["unmet_demand_fraction"][step - 1],
            price=price,
            supplier_health=float(np.mean(supplier_healths)),
            demand=demand,
        )
        action = _single_supply_action(policy(observation, config), config)
        demand = _next_demand(demand, config, rng)
        effective_demand = demand * (1.0 - action.rationing)
        capacity = (
            config.base_capacity_per_supplier
            * config.suppliers
            * float(np.mean(supplier_healths))
            * (1.0 + action.production_boost)
        )
        release = action.inventory_release * inventory
        available = inventory + capacity + release
        shipped = min(available, effective_demand)
        unmet_fraction = max(0.0, effective_demand - shipped) / max(effective_demand, 1.0)
        inventory = max(0.0, inventory + capacity - shipped - release)
        supplier_healths = _update_supplier_healths(
            supplier_healths,
            capacity,
            demand,
            unmet_fraction,
            action,
            config,
            rng,
        )
        price = _update_price(price, inventory, unmet_fraction, action, config)

        actions["inventory_release"][step] = action.inventory_release
        actions["production_boost"][step] = action.production_boost
        actions["rationing"][step] = action.rationing
        actions["supplier_subsidy"][step] = action.supplier_subsidy
        variables["inventory"][step] = inventory
        variables["unmet_demand_fraction"][step] = unmet_fraction
        variables["price"][step] = price
        variables["supplier_health"][step] = float(np.mean(supplier_healths))

    return GenericRun(time=time, variables=variables, actions=actions)


def no_supply_control(observation: SupplyObservation, config: SupplyChainConfig) -> SupplyAction:
    return SupplyAction()


def supply_feedback_control() -> SupplyPolicy:
    inventory_integral = 0.0
    price_integral = 0.0

    def policy(observation: SupplyObservation, config: SupplyChainConfig) -> SupplyAction:
        nonlocal inventory_integral, price_integral
        inventory_error = (config.target_inventory - observation.inventory) / config.target_inventory
        price_error = observation.price - config.target_price
        inventory_integral = float(np.clip(0.92 * inventory_integral + inventory_error, -2.0, 2.0))
        price_integral = float(np.clip(0.90 * price_integral + price_error, -2.0, 2.0))
        if observation.supplier_health < 0.72:
            return SupplyAction(supplier_subsidy=min(config.max_supplier_subsidy, 0.10 + 0.7 * (0.72 - observation.supplier_health)))
        if observation.unmet_demand_fraction > 0.12:
            return SupplyAction(rationing=min(config.max_rationing, 0.08 + observation.unmet_demand_fraction))
        if inventory_error > 0.12:
            return SupplyAction(production_boost=min(config.max_production_boost, 0.12 * inventory_error + 0.025 * inventory_integral))
        if price_error > 0.16 and observation.inventory > config.target_inventory * 0.75:
            return SupplyAction(inventory_release=min(config.max_inventory_release, 0.10 * price_error + 0.02 * price_integral))
        return SupplyAction()

    return policy


def supply_rule_based(observation: SupplyObservation, config: SupplyChainConfig) -> SupplyAction:
    if observation.supplier_health < 0.68:
        return SupplyAction(supplier_subsidy=config.max_supplier_subsidy)
    if observation.unmet_demand_fraction > 0.20:
        return SupplyAction(rationing=config.max_rationing)
    if observation.inventory < config.target_inventory * 0.55:
        return SupplyAction(production_boost=config.max_production_boost)
    if observation.price > 1.30 and observation.inventory > config.target_inventory * 0.85:
        return SupplyAction(inventory_release=config.max_inventory_release * 0.60)
    return SupplyAction()


def supply_look_ahead(config: SupplyChainConfig) -> SupplyPolicy:
    candidates = [
        SupplyAction(),
        SupplyAction(inventory_release=0.12),
        SupplyAction(inventory_release=0.24),
        SupplyAction(production_boost=0.14),
        SupplyAction(production_boost=0.26),
        SupplyAction(rationing=0.12),
        SupplyAction(supplier_subsidy=0.18),
    ]

    def policy(observation: SupplyObservation, config: SupplyChainConfig) -> SupplyAction:
        scored = [(_score_supply_action(observation, action, config), action) for action in candidates]
        scored.sort(key=lambda item: item[0])
        return scored[0][1]

    return policy


def _score_supply_action(
    observation: SupplyObservation,
    action: SupplyAction,
    config: SupplyChainConfig,
) -> float:
    inventory = observation.inventory
    price = observation.price
    health = observation.supplier_health
    unmet = observation.unmet_demand_fraction
    demand = observation.demand
    scores = []
    for _ in range(10):
        demand = 0.86 * demand + 0.14 * config.base_demand
        capacity = config.base_capacity_per_supplier * config.suppliers * health * (1.0 + action.production_boost)
        effective_demand = demand * (1.0 - action.rationing)
        release = action.inventory_release * inventory
        shipped = min(inventory + capacity + release, effective_demand)
        unmet = max(0.0, effective_demand - shipped) / max(effective_demand, 1.0)
        inventory = max(0.0, inventory + capacity - shipped - release)
        health = min(1.0, max(0.0, health + config.supplier_recovery_rate * (1.0 - health) + 0.10 * action.supplier_subsidy - 0.10 * unmet))
        price = _aggregate_price(price, inventory, unmet, action, config)
        scores.append(
            abs(inventory - config.target_inventory) / config.target_inventory
            + 1.8 * abs(price - config.target_price)
            + 2.2 * unmet
            + 0.9 * abs(health - 0.92)
            + 0.04 * sum(action.__dict__.values())
        )
    return float(np.mean(scores))


def _single_supply_action(action: SupplyAction, config: SupplyChainConfig) -> SupplyAction:
    amounts = {
        "inventory_release": min(max(action.inventory_release, 0.0), config.max_inventory_release),
        "production_boost": min(max(action.production_boost, 0.0), config.max_production_boost),
        "rationing": min(max(action.rationing, 0.0), config.max_rationing),
        "supplier_subsidy": min(max(action.supplier_subsidy, 0.0), config.max_supplier_subsidy),
    }
    choice = max(amounts, key=amounts.get)
    if amounts[choice] <= 0.0:
        return SupplyAction()
    return SupplyAction(**{choice: amounts[choice]})


def _next_demand(
    previous_demand: float,
    config: SupplyChainConfig,
    rng: np.random.Generator,
) -> float:
    shock = 0.0
    if rng.random() < config.demand_shock_probability:
        shock = float(rng.gamma(2.0, config.demand_shock_scale))
    noise = float(rng.normal(0.0, config.demand_noise))
    return max(20.0, 0.82 * previous_demand + 0.18 * config.base_demand + noise + shock)


def _update_supplier_healths(
    supplier_healths: np.ndarray,
    capacity: float,
    demand: float,
    unmet_fraction: float,
    action: SupplyAction,
    config: SupplyChainConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    load = demand / max(capacity, 1.0)
    stress = config.supplier_stress_decay * max(0.0, load - 0.85) + 0.08 * unmet_fraction
    recovery = config.supplier_recovery_rate * (1.0 - supplier_healths)
    subsidy = 0.11 * action.supplier_subsidy
    supplier_healths = supplier_healths + recovery + subsidy - stress
    supplier_healths += rng.normal(0.0, 0.006, size=supplier_healths.shape)
    return np.clip(supplier_healths, 0.0, 1.0)


def _update_price(
    price: float,
    inventory: float,
    unmet_fraction: float,
    action: SupplyAction,
    config: SupplyChainConfig,
) -> float:
    return _aggregate_price(price, inventory, unmet_fraction, action, config)


def _aggregate_price(
    price: float,
    inventory: float,
    unmet_fraction: float,
    action: SupplyAction,
    config: SupplyChainConfig,
) -> float:
    scarcity = max(0.0, (config.target_inventory - inventory) / config.target_inventory)
    surplus = max(0.0, (inventory - config.target_inventory) / config.target_inventory)
    next_price = (
        0.88 * price
        + 0.12 * config.target_price
        + 0.42 * scarcity
        + 0.75 * unmet_fraction
        - 0.16 * surplus
        - 0.15 * action.inventory_release
    )
    return float(np.clip(next_price, 0.45, 2.50))
