from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, List, Tuple

import numpy as np

from agent_dynamic_system.config import SimulationConfig
from agent_dynamic_system.controllers import ControlAction, Controller, Observation


@dataclass
class Animal:
    x: int
    y: int
    energy: float


@dataclass
class SimulationRun:
    time: np.ndarray
    grass: np.ndarray
    rabbits: np.ndarray
    foxes: np.ndarray
    cut_fraction: np.ndarray
    fertilizer_fraction: np.ndarray


def simulate(
    config: SimulationConfig,
    seed: int,
    controller: Controller,
) -> SimulationRun:
    """Run one independent grass/rabbit/fox simulation."""

    rng = np.random.default_rng(seed)
    controller.reset(config, seed)

    grass = _initial_grass(config, rng)
    rabbits = _initial_animals(
        config.initial_rabbits,
        config,
        rng,
        config.rabbit_initial_energy_min,
        config.rabbit_initial_energy_max,
    )
    foxes = _initial_animals(
        config.initial_foxes,
        config,
        rng,
        config.fox_initial_energy_min,
        config.fox_initial_energy_max,
    )

    time = np.arange(config.steps + 1)
    grass_series = np.zeros(config.steps + 1, dtype=float)
    rabbit_series = np.zeros(config.steps + 1, dtype=float)
    fox_series = np.zeros(config.steps + 1, dtype=float)
    cut_series = np.zeros(config.steps + 1, dtype=float)
    fertilizer_series = np.zeros(config.steps + 1, dtype=float)

    _record(0, grass, rabbits, foxes, grass_series, rabbit_series, fox_series)

    for step in range(1, config.steps + 1):
        observation = Observation(
            step=step - 1,
            grass_biomass=float(grass.sum()),
            rabbits=len(rabbits),
            foxes=len(foxes),
            grass_capacity_total=config.grass_capacity_total,
        )
        action = _single_action_choice(controller.act(observation), config)

        _apply_control(grass, action, config)
        _regrow_grass(grass, config, rng)
        rabbits = _step_rabbits(rabbits, grass, config, rng)
        rabbits, foxes = _step_foxes(rabbits, foxes, config, rng)

        cut_series[step] = action.cut_fraction
        fertilizer_series[step] = action.fertilizer_fraction
        _record(step, grass, rabbits, foxes, grass_series, rabbit_series, fox_series)

    return SimulationRun(
        time=time,
        grass=grass_series,
        rabbits=rabbit_series,
        foxes=fox_series,
        cut_fraction=cut_series,
        fertilizer_fraction=fertilizer_series,
    )


def _initial_grass(config: SimulationConfig, rng: np.random.Generator) -> np.ndarray:
    low = config.grass_capacity * config.initial_grass_min_fraction
    high = config.grass_capacity * config.initial_grass_max_fraction
    return rng.uniform(low, high, size=(config.grid_height, config.grid_width))


def _initial_animals(
    count: int,
    config: SimulationConfig,
    rng: np.random.Generator,
    energy_min: float,
    energy_max: float,
) -> List[Animal]:
    animals = []
    for _ in range(count):
        animals.append(
            Animal(
                x=int(rng.integers(0, config.grid_width)),
                y=int(rng.integers(0, config.grid_height)),
                energy=float(rng.uniform(energy_min, energy_max)),
            )
        )
    return animals


def _record(
    step: int,
    grass: np.ndarray,
    rabbits: List[Animal],
    foxes: List[Animal],
    grass_series: np.ndarray,
    rabbit_series: np.ndarray,
    fox_series: np.ndarray,
) -> None:
    grass_series[step] = float(grass.sum())
    rabbit_series[step] = float(len(rabbits))
    fox_series[step] = float(len(foxes))


def _apply_control(
    grass: np.ndarray,
    action: ControlAction,
    config: SimulationConfig,
) -> None:
    cut_fraction = action.cut_fraction
    fertilizer_fraction = action.fertilizer_fraction

    if cut_fraction:
        grass *= 1.0 - cut_fraction
    if fertilizer_fraction:
        grass += fertilizer_fraction * (config.grass_capacity - grass)
    np.clip(grass, 0.0, config.grass_capacity, out=grass)


def _single_action_choice(
    action: ControlAction,
    config: SimulationConfig,
) -> ControlAction:
    cut_fraction = min(max(action.cut_fraction, 0.0), config.max_cut_fraction)
    fertilizer_fraction = min(
        max(action.fertilizer_fraction, 0.0),
        config.max_fertilizer_fraction,
    )

    if cut_fraction <= 0.0 and fertilizer_fraction <= 0.0:
        return ControlAction()
    if cut_fraction >= fertilizer_fraction:
        return ControlAction(cut_fraction=cut_fraction)
    return ControlAction(fertilizer_fraction=fertilizer_fraction)


def _regrow_grass(
    grass: np.ndarray,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> None:
    grass += config.grass_regrowth_rate * (config.grass_capacity - grass)
    if config.grass_regrowth_noise:
        grass += rng.normal(0.0, config.grass_regrowth_noise, size=grass.shape)
    np.clip(grass, 0.0, config.grass_capacity, out=grass)


def _step_rabbits(
    rabbits: List[Animal],
    grass: np.ndarray,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> List[Animal]:
    next_rabbits = []

    for rabbit in rabbits:
        _move(rabbit, config, rng)
        rabbit.energy -= config.rabbit_metabolic_cost

        eaten = min(config.rabbit_eat_rate, float(grass[rabbit.y, rabbit.x]))
        grass[rabbit.y, rabbit.x] -= eaten
        rabbit.energy += eaten * config.rabbit_energy_gain

        if rabbit.energy <= 0.0:
            continue

        next_rabbits.append(rabbit)

        if (
            len(next_rabbits) < config.max_rabbits
            and rabbit.energy >= config.rabbit_reproduction_energy
            and rng.random() < config.rabbit_reproduction_probability
        ):
            child_energy = rabbit.energy * config.rabbit_child_energy_fraction
            rabbit.energy -= child_energy
            child = Animal(rabbit.x, rabbit.y, child_energy)
            _move(child, config, rng)
            next_rabbits.append(child)

    return next_rabbits[: config.max_rabbits]


def _step_foxes(
    rabbits: List[Animal],
    foxes: List[Animal],
    config: SimulationConfig,
    rng: np.random.Generator,
) -> Tuple[List[Animal], List[Animal]]:
    rabbit_alive = [True] * len(rabbits)
    rabbits_by_cell = _index_rabbits_by_cell(rabbits)
    next_foxes = []

    for fox in foxes:
        _move(fox, config, rng)
        fox.energy -= config.fox_metabolic_cost

        cell = (fox.x, fox.y)
        if rabbits_by_cell[cell]:
            rabbit_index = rabbits_by_cell[cell].pop()
            rabbit_alive[rabbit_index] = False
            fox.energy += config.fox_hunt_gain

        if fox.energy <= 0.0:
            continue

        next_foxes.append(fox)

        if (
            len(next_foxes) < config.max_foxes
            and fox.energy >= config.fox_reproduction_energy
            and rng.random() < config.fox_reproduction_probability
        ):
            child_energy = fox.energy * config.fox_child_energy_fraction
            fox.energy -= child_energy
            child = Animal(fox.x, fox.y, child_energy)
            _move(child, config, rng)
            next_foxes.append(child)

    surviving_rabbits = [
        rabbit for rabbit, alive in zip(rabbits, rabbit_alive) if alive and rabbit.energy > 0.0
    ]
    return surviving_rabbits, next_foxes[: config.max_foxes]


def _index_rabbits_by_cell(
    rabbits: List[Animal],
) -> DefaultDict[Tuple[int, int], List[int]]:
    rabbits_by_cell = defaultdict(list)
    for index, rabbit in enumerate(rabbits):
        rabbits_by_cell[(rabbit.x, rabbit.y)].append(index)
    return rabbits_by_cell


def _move(
    animal: Animal,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> None:
    animal.x = (animal.x + int(rng.integers(-1, 2))) % config.grid_width
    animal.y = (animal.y + int(rng.integers(-1, 2))) % config.grid_height
