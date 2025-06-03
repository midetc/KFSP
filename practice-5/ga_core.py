import random
import math
import db_utils


def initialize_population(pop_size):
    population = []
    for _ in range(pop_size):
        individual = random.uniform(0, 10)
        population.append(individual)
    return population


def evaluate_fitness(individual):
    return individual * math.sin(individual)


def tournament_selection(population, fitnesses, k):
    tournament_indices = random.sample(range(len(population)), k)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    best_index = tournament_indices[
        tournament_fitnesses.index(max(tournament_fitnesses))
    ]
    return population[best_index]


def crossover(parent1, parent2):
    child1 = (parent1 + parent2) / 2 + random.uniform(-0.5, 0.5)
    child2 = (parent1 + parent2) / 2 + random.uniform(-0.5, 0.5)
    child1 = max(0, min(10, child1))
    child2 = max(0, min(10, child2))
    return child1, child2


def mutate(individual, mutation_prob):
    if random.random() < mutation_prob:
        mutation_strength = random.uniform(-0.5, 0.5)
        individual += mutation_strength
        individual = max(0, min(10, individual))
    return individual


def create_offspring(population, fitnesses, crossover_prob,
                     mutation_prob, tournament_size):
    offspring = []
    pop_size = len(population)
    for _ in range(pop_size // 2):
        parent1 = tournament_selection(population, fitnesses, tournament_size)
        parent2 = tournament_selection(population, fitnesses, tournament_size)
        if random.random() < crossover_prob:
            child1, child2 = crossover(parent1, parent2)
        else:
            child1, child2 = parent1, parent2
        child1 = mutate(child1, mutation_prob)
        child2 = mutate(child2, mutation_prob)
        offspring.extend([child1, child2])
    if len(offspring) < pop_size:
        parent = tournament_selection(population, fitnesses, tournament_size)
        child = mutate(parent, mutation_prob)
        offspring.append(child)
    return offspring[:pop_size]


def elitist_replacement(old_population, old_fitnesses,
                        new_population, new_fitnesses):
    best_old_index = old_fitnesses.index(max(old_fitnesses))
    best_old_individual = old_population[best_old_index]
    best_old_fitness = old_fitnesses[best_old_index]
    worst_new_index = new_fitnesses.index(min(new_fitnesses))
    worst_new_fitness = new_fitnesses[worst_new_index]
    if best_old_fitness > worst_new_fitness:
        new_population[worst_new_index] = best_old_individual
        new_fitnesses[worst_new_index] = best_old_fitness
    return new_population, new_fitnesses


def termination_condition(generation, max_generations,
                          last_improvement_gen, patience=10):
    if generation >= max_generations:
        return True
    if generation - last_improvement_gen >= patience:
        return True
    return False


def print_stats(population, fitnesses, generation):
    avg_fitness = sum(fitnesses) / len(fitnesses)
    best_fitness = max(fitnesses)
    best_individual = population[fitnesses.index(best_fitness)]
    print(f"Покоління {generation}: Сер. придатність = {avg_fitness:.4f}, "
          f"Найкраща придатність = {best_fitness:.4f}, "
          f"Найкраща особина = {best_individual:.4f}")
    return avg_fitness, best_fitness, best_individual


def run_genetic_algorithm(params):
    pop_size = params['pop_size']
    max_generations = params['num_generations']
    crossover_prob = params['crossover_prob']
    mutation_prob = params['mutation_prob']
    tournament_size = params['tournament_size']
    population = initialize_population(pop_size)
    fitnesses = [evaluate_fitness(individual) for individual in population]
    best_fitness_ever = max(fitnesses)
    last_improvement_gen = 0
    for generation in range(1, max_generations + 1):
        avg_fitness, best_fitness, best_individual = print_stats(
            population, fitnesses, generation
        )
        db_utils.save_stats_to_db(
            'ga.db', generation, avg_fitness, best_fitness, best_individual
        )
        if best_fitness > best_fitness_ever:
            best_fitness_ever = best_fitness
            last_improvement_gen = generation
        if termination_condition(generation, max_generations,
                                 last_improvement_gen):
            break
        new_population = create_offspring(
            population, fitnesses, crossover_prob,
            mutation_prob, tournament_size
        )
        new_fitnesses = [
            evaluate_fitness(individual) for individual in new_population
        ]
        population, fitnesses = elitist_replacement(
            population, fitnesses, new_population, new_fitnesses
        )
    best_index = fitnesses.index(max(fitnesses))
    return population[best_index], fitnesses[best_index]
