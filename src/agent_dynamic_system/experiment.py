from dataclasses import dataclass
from typing import Callable, List, Sequence

import numpy as np

from agent_dynamic_system.config import SimulationConfig
from agent_dynamic_system.controllers import Controller
from agent_dynamic_system.simulation import SimulationRun, simulate


ControllerFactory = Callable[[SimulationConfig], Controller]


@dataclass(frozen=True)
class Scenario:
    name: str
    controller_factory: ControllerFactory


@dataclass
class MonteCarloResult:
    scenario_name: str
    controller_name: str
    seeds: Sequence[int]
    time: np.ndarray
    grass: np.ndarray
    rabbits: np.ndarray
    foxes: np.ndarray
    cut_fraction: np.ndarray
    fertilizer_fraction: np.ndarray


def run_scenario(
    config: SimulationConfig,
    scenario: Scenario,
    runs: int,
    base_seed: int,
) -> MonteCarloResult:
    """Run repeated simulations for one scenario without saving per-run files."""

    seeds = [base_seed + run_index for run_index in range(runs)]
    simulations: List[SimulationRun] = []
    controller_name = scenario.name

    for seed in seeds:
        controller = scenario.controller_factory(config)
        controller_name = controller.name
        simulations.append(simulate(config, seed, controller))

    return MonteCarloResult(
        scenario_name=scenario.name,
        controller_name=controller_name,
        seeds=seeds,
        time=simulations[0].time,
        grass=np.vstack([run.grass for run in simulations]),
        rabbits=np.vstack([run.rabbits for run in simulations]),
        foxes=np.vstack([run.foxes for run in simulations]),
        cut_fraction=np.vstack([run.cut_fraction for run in simulations]),
        fertilizer_fraction=np.vstack(
            [run.fertilizer_fraction for run in simulations]
        ),
    )
