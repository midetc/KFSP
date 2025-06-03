import unittest
import sys
import math
import os
import ga_core
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestGeneticAlgorithm(unittest.TestCase):

    def test_initialize_population(self):
        pop_size = 10
        population = ga_core.initialize_population(pop_size)

        self.assertEqual(len(population), pop_size)

        for individual in population:
            self.assertTrue(0.0 <= individual <= 10.0)
            self.assertIsInstance(individual, float)

    def test_evaluate_fitness(self):
        self.assertAlmostEqual(ga_core.evaluate_fitness(0), 0.0, places=6)
        x = math.pi
        expected = x * math.sin(x)
        actual = ga_core.evaluate_fitness(x)
        self.assertAlmostEqual(actual, expected, places=6)
        result = ga_core.evaluate_fitness(5.0)
        self.assertIsInstance(result, float)

    def test_tournament_selection(self):
        population = [1.0, 5.0, 8.0]
        fitnesses = [ga_core.evaluate_fitness(x) for x in population]
        tournament_size = 2
        for _ in range(10):
            selected = ga_core.tournament_selection(
                population, fitnesses, tournament_size
            )
            self.assertIn(selected, population)
            self.assertIsInstance(selected, float)

    def test_crossover(self):
        parent1, parent2 = 2.0, 8.0
        child1, child2 = ga_core.crossover(parent1, parent2)
        self.assertTrue(0.0 <= child1 <= 10.0)
        self.assertTrue(0.0 <= child2 <= 10.0)
        self.assertIsInstance(child1, float)
        self.assertIsInstance(child2, float)
        children = [ga_core.crossover(parent1, parent2) for _ in range(5)]
        unique_children = set(child for pair in children for child in pair)
        self.assertGreater(len(unique_children), 1)

    def test_mutate(self):
        original = 5.0
        mutated = ga_core.mutate(original, 0.0)
        self.assertEqual(mutated, original)
        mutations = [ga_core.mutate(original, 1.0) for _ in range(10)]
        for mut in mutations:
            self.assertTrue(0.0 <= mut <= 10.0)
        different_mutations = [m for m in mutations if m != original]
        self.assertGreater(len(different_mutations), 0)

    def test_create_offspring(self):
        population = [1.0, 3.0, 7.0, 9.0]
        fitnesses = [ga_core.evaluate_fitness(x) for x in population]
        crossover_prob = 0.8
        mutation_prob = 0.1
        tournament_size = 2
        offspring = ga_core.create_offspring(
            population, fitnesses, crossover_prob,
            mutation_prob, tournament_size
        )
        self.assertEqual(len(offspring), len(population))
        for individual in offspring:
            self.assertTrue(0.0 <= individual <= 10.0)
            self.assertIsInstance(individual, float)

    def test_elitist_replacement(self):
        old_population = [0.0, 2.0, 7.0]
        old_fitnesses = [ga_core.evaluate_fitness(x) for x in old_population]
        new_population = [1.0, 1.5, 2.5]
        new_fitnesses = [ga_core.evaluate_fitness(x) for x in new_population]
        best_old = old_population[old_fitnesses.index(max(old_fitnesses))]
        result_pop, result_fit = ga_core.elitist_replacement(
            old_population, old_fitnesses,
            new_population.copy(), new_fitnesses.copy()
        )
        self.assertIn(best_old, result_pop)
        self.assertEqual(len(result_pop), 3)
        self.assertEqual(len(result_fit), 3)

    def test_termination_condition(self):
        self.assertTrue(ga_core.termination_condition(100, 100, 90))
        self.assertFalse(ga_core.termination_condition(50, 100, 45))
        self.assertTrue(
            ga_core.termination_condition(50, 100, 30, patience=10)
        )
        self.assertFalse(
            ga_core.termination_condition(50, 100, 45, patience=10)
        )

    def test_print_stats(self):
        population = [2.0, 5.0, 8.0]
        fitnesses = [ga_core.evaluate_fitness(x) for x in population]
        generation = 1
        avg_fitness, best_fitness, best_individual = ga_core.print_stats(
            population, fitnesses, generation
        )
        self.assertIsInstance(avg_fitness, float)
        self.assertIsInstance(best_fitness, float)
        self.assertIsInstance(best_individual, float)
        expected_avg = sum(fitnesses) / len(fitnesses)
        self.assertAlmostEqual(avg_fitness, expected_avg, places=6)
        self.assertEqual(best_fitness, max(fitnesses))
        self.assertIn(best_individual, population)


class TestIntegration(unittest.TestCase):
    def test_full_algorithm_run(self):
        params = {
            'pop_size': 10,
            'num_generations': 5,
            'crossover_prob': 0.8,
            'mutation_prob': 0.1,
            'tournament_size': 3
        }
        original_save = ga_core.db_utils.save_stats_to_db
        ga_core.db_utils.save_stats_to_db = lambda *args: None
        try:
            best_individual, best_fitness = ga_core.run_genetic_algorithm(
                params
            )
            self.assertIsInstance(best_individual, float)
            self.assertIsInstance(best_fitness, float)
            self.assertTrue(0.0 <= best_individual <= 10.0)
        finally:
            ga_core.db_utils.save_stats_to_db = original_save


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        unittest.main(argv=[sys.argv[0]])
    else:
        unittest.main()
