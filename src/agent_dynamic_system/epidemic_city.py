from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from agent_dynamic_system.benchmark import GenericRun, GenericScenario, MetricSpec


@dataclass(frozen=True)
class EpidemicCityConfig:
    districts: int = 24
    steps: int = 500
    population_per_district: int = 4200
    initial_infected_fraction: float = 0.012
    base_transmission: float = 0.34
    recovery_rate: float = 0.055
    waning_immunity_rate: float = 0.002
    mobility_rate: float = 0.045
    event_shock_probability: float = 0.025
    event_shock_scale: float = 0.18
    max_vaccination: float = 0.030
    max_testing_isolation: float = 0.30
    max_mobility_reduction: float = 0.35
    max_hospital_surge: float = 0.28
    target_infected_fraction: float = 0.018
    target_hospital_load: float = 0.36
    infected_safety_max: float = 0.10
    hospital_safety_max: float = 0.88


@dataclass(frozen=True)
class EpidemicAction:
    vaccination: float = 0.0
    testing_isolation: float = 0.0
    mobility_reduction: float = 0.0
    hospital_surge: float = 0.0


@dataclass(frozen=True)
class EpidemicObservation:
    step: int
    infected_fraction: float
    susceptible_fraction: float
    recovered_fraction: float
    hospital_load: float


EpidemicPolicy = Callable[[EpidemicObservation, EpidemicCityConfig], EpidemicAction]


def build_epidemic_city_scenarios(config: EpidemicCityConfig) -> List[GenericScenario]:
    return [
        GenericScenario("baseline", lambda seed: simulate_epidemic_city(config, seed, no_epidemic_control)),
        GenericScenario("control_theory", lambda seed: simulate_epidemic_city(config, seed, epidemic_feedback_control())),
        GenericScenario("rule_based", lambda seed: simulate_epidemic_city(config, seed, epidemic_rule_based)),
        GenericScenario("look_ahead", lambda seed: simulate_epidemic_city(config, seed, epidemic_look_ahead(config))),
    ]


def epidemic_city_metric_spec(config: EpidemicCityConfig) -> MetricSpec:
    return MetricSpec(
        target={
            "infected_fraction": config.target_infected_fraction,
            "hospital_load": config.target_hospital_load,
            "susceptible_fraction": 0.72,
            "recovered_fraction": 0.20,
        },
        safety_min={"susceptible_fraction": 0.38},
        safety_max={
            "infected_fraction": config.infected_safety_max,
            "hospital_load": config.hospital_safety_max,
        },
        cost_weight=0.04,
    )


def epidemic_city_labels() -> Tuple[Dict[str, str], Dict[str, str]]:
    variables = {
        "infected_fraction": "Infected fraction",
        "hospital_load": "Hospital load",
        "susceptible_fraction": "Susceptible fraction",
        "recovered_fraction": "Recovered fraction",
    }
    actions = {
        "vaccination": "Vaccination",
        "testing_isolation": "Testing/isolation",
        "mobility_reduction": "Mobility reduction",
        "hospital_surge": "Hospital surge capacity",
    }
    return variables, actions


def simulate_epidemic_city(
    config: EpidemicCityConfig,
    seed: int,
    policy: EpidemicPolicy,
) -> GenericRun:
    rng = np.random.default_rng(seed)
    population = float(config.population_per_district)
    susceptible = np.full(config.districts, population * (1.0 - config.initial_infected_fraction))
    infected = np.full(config.districts, population * config.initial_infected_fraction)
    recovered = np.zeros(config.districts)
    care_capacity = population * config.districts * 0.018

    time = np.arange(config.steps + 1)
    variables = {name: np.zeros(config.steps + 1) for name in epidemic_city_labels()[0]}
    actions = {name: np.zeros(config.steps + 1) for name in epidemic_city_labels()[1]}
    _record(variables, 0, susceptible, infected, recovered, care_capacity)

    for step in range(1, config.steps + 1):
        observation = EpidemicObservation(
            step=step - 1,
            infected_fraction=variables["infected_fraction"][step - 1],
            susceptible_fraction=variables["susceptible_fraction"][step - 1],
            recovered_fraction=variables["recovered_fraction"][step - 1],
            hospital_load=variables["hospital_load"][step - 1],
        )
        action = _single_epidemic_action(policy(observation, config), config)
        susceptible, infected, recovered, care_capacity = _epidemic_step(
            susceptible, infected, recovered, care_capacity, action, config, rng
        )
        actions["vaccination"][step] = action.vaccination
        actions["testing_isolation"][step] = action.testing_isolation
        actions["mobility_reduction"][step] = action.mobility_reduction
        actions["hospital_surge"][step] = action.hospital_surge
        _record(variables, step, susceptible, infected, recovered, care_capacity)

    return GenericRun(time=time, variables=variables, actions=actions)


def no_epidemic_control(observation: EpidemicObservation, config: EpidemicCityConfig) -> EpidemicAction:
    return EpidemicAction()


def epidemic_feedback_control() -> EpidemicPolicy:
    integral = 0.0

    def policy(observation: EpidemicObservation, config: EpidemicCityConfig) -> EpidemicAction:
        nonlocal integral
        infection_error = observation.infected_fraction - config.target_infected_fraction
        hospital_error = observation.hospital_load - config.target_hospital_load
        integral = float(np.clip(0.92 * integral + infection_error, -0.45, 0.45))
        if observation.hospital_load > config.hospital_safety_max * 0.72:
            return EpidemicAction(hospital_surge=min(config.max_hospital_surge, 0.08 + 0.45 * hospital_error))
        if infection_error > 0.018:
            return EpidemicAction(testing_isolation=min(config.max_testing_isolation, 1.8 * infection_error + 0.5 * integral))
        if observation.susceptible_fraction > 0.50 and observation.infected_fraction > config.target_infected_fraction * 0.7:
            return EpidemicAction(vaccination=config.max_vaccination)
        if infection_error > 0.006:
            return EpidemicAction(mobility_reduction=min(config.max_mobility_reduction, 2.2 * infection_error))
        return EpidemicAction()

    return policy


def epidemic_rule_based(observation: EpidemicObservation, config: EpidemicCityConfig) -> EpidemicAction:
    if observation.hospital_load > 0.72:
        return EpidemicAction(hospital_surge=config.max_hospital_surge)
    if observation.infected_fraction > 0.065:
        return EpidemicAction(testing_isolation=config.max_testing_isolation)
    if observation.infected_fraction > 0.040:
        return EpidemicAction(mobility_reduction=config.max_mobility_reduction * 0.7)
    if observation.susceptible_fraction > 0.55 and observation.infected_fraction > 0.020:
        return EpidemicAction(vaccination=config.max_vaccination)
    return EpidemicAction()


def epidemic_look_ahead(config: EpidemicCityConfig) -> EpidemicPolicy:
    candidates = [
        EpidemicAction(),
        EpidemicAction(vaccination=0.020),
        EpidemicAction(testing_isolation=0.18),
        EpidemicAction(testing_isolation=0.30),
        EpidemicAction(mobility_reduction=0.22),
        EpidemicAction(hospital_surge=0.20),
    ]

    def policy(observation: EpidemicObservation, config: EpidemicCityConfig) -> EpidemicAction:
        scored = [(_score_action(observation, action, config), action) for action in candidates]
        scored.sort(key=lambda item: item[0])
        return scored[0][1]

    return policy


def _score_action(
    observation: EpidemicObservation,
    action: EpidemicAction,
    config: EpidemicCityConfig,
) -> float:
    infected = observation.infected_fraction
    susceptible = observation.susceptible_fraction
    recovered = observation.recovered_fraction
    hospital = observation.hospital_load
    scores = []
    for _ in range(10):
        effective_beta = config.base_transmission * (1.0 - action.testing_isolation) * (1.0 - 0.75 * action.mobility_reduction)
        new_cases = effective_beta * infected * susceptible
        recoveries = config.recovery_rate * infected
        vaccinated = action.vaccination * susceptible
        susceptible = max(0.0, susceptible - new_cases - vaccinated + config.waning_immunity_rate * recovered)
        infected = max(0.0, infected + new_cases - recoveries)
        recovered = max(0.0, min(1.0, recovered + recoveries + vaccinated - config.waning_immunity_rate * recovered))
        hospital = max(0.0, min(1.5, 0.82 * hospital + 3.8 * infected - action.hospital_surge))
        scores.append(
            2.0 * abs(infected - config.target_infected_fraction)
            + 1.8 * max(0.0, hospital - config.target_hospital_load)
            + 0.4 * abs(susceptible - 0.72)
            + 0.04 * sum(action.__dict__.values())
        )
    return float(np.mean(scores))


def _single_epidemic_action(action: EpidemicAction, config: EpidemicCityConfig) -> EpidemicAction:
    amounts = {
        "vaccination": min(max(action.vaccination, 0.0), config.max_vaccination),
        "testing_isolation": min(max(action.testing_isolation, 0.0), config.max_testing_isolation),
        "mobility_reduction": min(max(action.mobility_reduction, 0.0), config.max_mobility_reduction),
        "hospital_surge": min(max(action.hospital_surge, 0.0), config.max_hospital_surge),
    }
    choice = max(amounts, key=amounts.get)
    if amounts[choice] <= 0.0:
        return EpidemicAction()
    return EpidemicAction(**{choice: amounts[choice]})


def _epidemic_step(
    susceptible: np.ndarray,
    infected: np.ndarray,
    recovered: np.ndarray,
    care_capacity: float,
    action: EpidemicAction,
    config: EpidemicCityConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    population = susceptible + infected + recovered
    total_population = np.sum(population)
    mobility = config.mobility_rate * (1.0 - action.mobility_reduction)
    mixed_infected = (1.0 - mobility) * infected / np.maximum(population, 1.0) + mobility * np.mean(infected / np.maximum(population, 1.0))
    event_shock = 0.0
    if rng.random() < config.event_shock_probability:
        event_shock = float(rng.gamma(2.0, config.event_shock_scale))
    transmission = config.base_transmission * (1.0 + event_shock)
    transmission *= 1.0 - 0.75 * action.testing_isolation
    new_cases = transmission * mixed_infected * susceptible
    new_cases += rng.poisson(np.maximum(new_cases * 0.03, 0.0))
    new_cases = np.minimum(new_cases, susceptible)
    recoveries = np.minimum(infected, config.recovery_rate * infected)
    vaccinations = np.minimum(susceptible, action.vaccination * susceptible)
    waning = config.waning_immunity_rate * recovered
    susceptible = susceptible - new_cases - vaccinations + waning
    infected = infected + new_cases - recoveries
    recovered = recovered + recoveries + vaccinations - waning
    care_capacity *= 1.0 + 0.03 * action.hospital_surge
    care_capacity = min(care_capacity, total_population * 0.04)
    return susceptible, infected, recovered, care_capacity


def _record(
    variables: Dict[str, np.ndarray],
    step: int,
    susceptible: np.ndarray,
    infected: np.ndarray,
    recovered: np.ndarray,
    care_capacity: float,
) -> None:
    total = float(np.sum(susceptible + infected + recovered))
    infected_total = float(np.sum(infected))
    variables["infected_fraction"][step] = infected_total / max(total, 1.0)
    variables["susceptible_fraction"][step] = float(np.sum(susceptible)) / max(total, 1.0)
    variables["recovered_fraction"][step] = float(np.sum(recovered)) / max(total, 1.0)
    variables["hospital_load"][step] = (0.08 * infected_total) / max(care_capacity, 1.0)
