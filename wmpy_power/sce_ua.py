import numpy as np
from os import linesep
from typing import Callable, Tuple


def shuffled_complex_evolution_algorithm(
    initial_population: np.ndarray,
    scoring_function: Callable,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    maximum_trials: int,
    maximum_evolution_loops: int,
    minimum_geometric_range: float,
    minimum_change_criteria: float,
    number_of_complexes: int,
    additional_function_arguments: dict = None,
    include_initial_point: bool = False,
    alpha: float = None,
    beta: float = None,
    random_number_generator: np.random.Generator = None,
    seed: int = None,
) -> Tuple[np.ndarray, float, str]:
    """
    Attempt to find the optimal minimization of a parameter space for a given scoring function.
    Implemented based on Duan et al., 1993: https://doi.org/10.1007/BF00939380.
    This implementation assumes that parallelization gains will be realized outside of this method,
    i.e. that multiple problems are solving in parallel rather than implementing a parallel version of this algorithm.
    """

    rng = random_number_generator
    if rng is None:
        rng = np.random.default_rng(seed)

    log_messages = []

    number_of_members_in_a_complex = 2 * len(initial_population) + 1
    number_of_members_in_a_simplex = len(initial_population) + 1
    number_of_evolution_steps = number_of_members_in_a_complex
    number_of_points = number_of_members_in_a_complex * number_of_complexes

    bound = upper_bound - lower_bound

    # initial population
    population = (
        lower_bound + rng.random((number_of_points, len(initial_population))) * bound
    )
    if include_initial_point:
        population[0, :] = initial_population

    # score the populations
    scores = np.zeros(population.shape[0])
    for i in range(len(scores)):
        scores[i] = scoring_function(
            population[i, :],
            **(
                additional_function_arguments
                if additional_function_arguments is not None
                else {}
            ),
        )
    iterations = number_of_points

    # sort the population by score
    sorted_index = np.argsort(scores, axis=0)
    scores = scores[sorted_index]
    population = population[sorted_index]

    normalized_geometric_range = np.exp(
        np.mean(
            np.log((np.max(population, axis=0) - np.min(population, axis=0)) / bound)
        )
    )

    # check convergence
    if iterations >= maximum_trials:
        log_messages.append("Maximum trials exceeded.")
    if normalized_geometric_range < minimum_geometric_range:
        log_messages.append(
            "The population converged to a pre-specified small parameter space."
        )

    # begin evolution loops
    loop = 0
    criteria = np.zeros(maximum_evolution_loops)
    criteria_change = np.inf
    while (
        (iterations < maximum_trials)
        and (normalized_geometric_range > minimum_geometric_range)
        and (criteria_change > minimum_change_criteria)
    ):
        loop += 1
        # iterate through complexes
        for i in range(number_of_complexes):

            # partition the population into complexes (subpopulations)
            complex_indices = (
                np.array(range(number_of_members_in_a_complex)) * number_of_complexes
                + i
            )
            complex_population = population[complex_indices]
            complex_scores = scores[complex_indices]

            # evolve the subpopulation
            for j in range(number_of_evolution_steps):
                # select a simplex by sampling the complex using a linear probability distribution
                simplex_indices = np.sort(
                    rng.choice(
                        number_of_members_in_a_complex,
                        number_of_members_in_a_simplex,
                        replace=False,
                    )
                )
                simplex_population = complex_population[simplex_indices]
                simplex_scores = complex_scores[simplex_indices]

                # get a new point
                new_point, new_score, tries = generate_new_point(
                    simplex_population,
                    simplex_scores,
                    lower_bound,
                    upper_bound,
                    scoring_function,
                    additional_function_arguments,
                    alpha=alpha,
                    beta=beta,
                    random_number_generator=rng,
                )
                iterations += tries

                # replace the worst point in the simplex with the new point
                simplex_population[-1] = new_point
                simplex_scores[-1] = new_score

                # replace the simplex into the complex
                complex_population[simplex_indices] = simplex_population
                complex_scores[simplex_indices] = simplex_scores

                # sort the complex
                complex_sorted_index = np.argsort(complex_scores, axis=0)
                complex_scores = complex_scores[complex_sorted_index]
                complex_population = complex_population[complex_sorted_index]

            # replace the complex back into the population
            population[complex_indices] = complex_population
            scores[complex_indices] = complex_scores

        # sort population by the new scores
        sorted_index = np.argsort(scores, axis=0)
        scores = scores[sorted_index]
        population = population[sorted_index]

        normalized_geometric_range = np.exp(
            np.mean(
                np.log(
                    (np.max(population, axis=0) - np.min(population, axis=0)) / bound
                )
            )
        )

        if iterations >= maximum_trials:
            log_messages.append("Maximum trials exceeded.")
        if normalized_geometric_range < minimum_geometric_range:
            log_messages.append(
                "The population converged to a pre-specified small parameter space."
            )
        criteria = np.roll(criteria, -1)
        criteria[-1] = scores[0]
        if loop >= maximum_evolution_loops:
            criteria_change = np.abs(criteria[-1] - criteria[0]) * 100
            criteria_change = criteria_change / np.mean(np.abs(criteria))
            if criteria_change < minimum_change_criteria:
                log_messages.append(
                    "The population converged because the best point did not improve above the threshold."
                )

    log_messages.append(f"Search took {iterations} trials.")
    log_messages.append(
        f"Normalized geometric range: {np.round(normalized_geometric_range, 2)}."
    )
    if criteria_change != np.inf:
        log_messages.append(
            f"The best point improved by {np.round(criteria_change, 2)}% in the last {maximum_evolution_loops} loops."
        )

    # return the best population, score, and log messages
    return population[0], scores[0], f"{linesep}\t".join(log_messages)


def generate_new_point(
    simplex_population: np.ndarray,
    simplex_scores: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    scoring_function: Callable,
    additional_function_arguments: dict = None,
    alpha: float = 1.0,
    beta: float = 0.5,
    random_number_generator: np.random.Generator = None,
    seed: int = None,
) -> Tuple[np.ndarray, float, int]:
    """
    Generate a new point within the parameter space.
    First try a reflection about the centroid.
    If score is worse, try a contraction about the centroid.
    If score is still worse, return a random point within the bounds.
    """

    rng = random_number_generator
    if rng is None:
        rng = np.random.default_rng(seed)

    # compute centroid of the simplex excluding worst point
    centroid = np.mean(simplex_population[:-1, :], axis=0)

    # try a reflection point
    new_point = centroid + alpha * (centroid - simplex_population[-1])

    # check if it is in bounds
    if np.all(new_point > lower_bound) and np.all(new_point < upper_bound):
        # if so, get the score
        new_score = scoring_function(
            new_point,
            **(
                additional_function_arguments
                if additional_function_arguments is not None
                else {}
            ),
        )
        # if new score is better than worst, return this point
        if new_score < simplex_scores[-1]:
            return new_point, new_score, 1

    # try a contraction point
    new_point = simplex_population[-1] + beta * (centroid - simplex_population[-1])
    new_score = scoring_function(
        new_point,
        **(
            additional_function_arguments
            if additional_function_arguments is not None
            else {}
        ),
    )

    # if new score is better than worst, return this point
    if new_score < simplex_scores[-1]:
        return new_point, new_score, 2

    # otherwise, create a random point
    new_point = lower_bound + rng.random(len(centroid)) * (upper_bound - lower_bound)
    new_score = scoring_function(
        new_point,
        **(
            additional_function_arguments
            if additional_function_arguments is not None
            else {}
        ),
    )
    return new_point, new_score, 3
