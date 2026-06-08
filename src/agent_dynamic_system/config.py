from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters for the grass/rabbit/fox agent-based model."""

    grid_width: int = 45
    grid_height: int = 45
    steps: int = 500

    initial_rabbits: int = 300
    initial_foxes: int = 20
    rabbit_safety_threshold: int = 50
    fox_safety_threshold: int = 10
    max_rabbits: int = 900
    max_foxes: int = 250

    grass_capacity: float = 6.0
    initial_grass_min_fraction: float = 0.45
    initial_grass_max_fraction: float = 1.0
    grass_regrowth_rate: float = 0.035
    grass_regrowth_noise: float = 0.012

    rabbit_initial_energy_min: float = 4.0
    rabbit_initial_energy_max: float = 8.0
    rabbit_metabolic_cost: float = 0.38
    rabbit_eat_rate: float = 1.35
    rabbit_energy_gain: float = 1.0
    rabbit_reproduction_energy: float = 9.0
    rabbit_reproduction_probability: float = 0.15
    rabbit_child_energy_fraction: float = 0.48

    fox_initial_energy_min: float = 6.0
    fox_initial_energy_max: float = 10.0
    fox_metabolic_cost: float = 0.55
    fox_hunt_gain: float = 6.0
    fox_reproduction_energy: float = 14.0
    fox_reproduction_probability: float = 0.10
    fox_child_energy_fraction: float = 0.45

    max_cut_fraction: float = 0.28
    max_fertilizer_fraction: float = 0.28

    @property
    def cell_count(self) -> int:
        return self.grid_width * self.grid_height

    @property
    def grass_capacity_total(self) -> float:
        return self.cell_count * self.grass_capacity
