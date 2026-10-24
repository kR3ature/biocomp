"""Genetic algorithm for graph coloring.

Graphs are stored as adjacency lists. A candidate coloring is a binary genome:
each vertex uses the minimum number of bits that can represent ``k`` colors.
Fitness is the fraction of edges whose endpoints have different, valid colors.
"""

import copy
import random

from deap import base, creator, tools


class AdjNode:
    """Singly linked adjacency-list node."""

    def __init__(self, vertex, nxt=None):
        self.vertex = vertex
        self.next = nxt


class Graph:
    """Undirected graph with ``n`` vertices and ``k`` available colors."""

    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.graph = [None] * n

    def add_edge(self, u, v):
        self.graph[u] = AdjNode(v, self.graph[u])
        if u != v:
            self.graph[v] = AdjNode(u, self.graph[v])


def init_graph(file_path):
    """Read a graph file and return a populated ``Graph``.

    File format::

        <number_of_nodes>
        <number_of_colors>
        u v
        u v
        ...
    """
    with open(file_path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    if len(lines) < 2:
        raise ValueError(f"Graph file is missing the node and color headers: {file_path}")

    n = int(lines[0])
    k = int(lines[1])
    graph = Graph(n, k)
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        graph.add_edge(int(parts[0]), int(parts[1]))
    return graph


def _bits_per_node(num_colors):
    """Bits required to encode one vertex color in ``0 .. num_colors - 1``."""
    if num_colors <= 1:
        return 1
    return (num_colors - 1).bit_length()


def decode_individual(graph, individual):
    """Decode a binary genome into a color for every vertex.

    Colors are read most-significant bit first. Returns ``None`` when the
    genome length is wrong or any vertex encodes a color outside ``0 .. k-1``.
    """
    bits = _bits_per_node(graph.k)
    if individual is None or len(individual) != graph.n * bits:
        return None

    colors = []
    for node in range(graph.n):
        color = 0
        start = node * bits
        for bit in individual[start:start + bits]:
            color = (color << 1) | int(bit)
        if color >= graph.k:
            return None
        colors.append(color)
    return colors


def eval_graph(graph, individual):
    """Return ``(satisfied_edges / total_edges,)``.

    An individual with any invalid color encoding scores ``(0.0,)``.
    A graph with no edges scores ``(1.0,)``.
    """
    colors = decode_individual(graph, individual)
    if colors is None:
        return (0.0,)

    total = 0
    satisfied = 0
    for u in range(graph.n):
        node = graph.graph[u]
        while node is not None:
            v = node.vertex
            if u < v:
                total += 1
                if colors[u] != colors[v]:
                    satisfied += 1
            node = node.next

    if total == 0:
        return (1.0,)
    return (satisfied / total,)


def _repair(individual, num_nodes, num_colors, bits):
    """Resample any vertex whose bits do not encode a legal color."""
    if (1 << bits) == num_colors:
        return individual
    for node in range(num_nodes):
        start = node * bits
        color = 0
        for bit in individual[start:start + bits]:
            color = (color << 1) | int(bit)
        if color >= num_colors:
            color = random.randrange(num_colors)
            for offset, shift in enumerate(range(bits - 1, -1, -1)):
                individual[start + offset] = (color >> shift) & 1
    return individual


def _random_individual(container, num_nodes, num_colors, bits):
    genes = []
    for _ in range(num_nodes):
        color = random.randrange(max(num_colors, 1))
        for shift in range(bits - 1, -1, -1):
            genes.append((color >> shift) & 1)
    return container(genes)


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


def run_ga(file_path, population_size=300, ngen=200, cxpb=0.8, mutpb=0.3):
    """Evolve colorings and return the best fitness found.

    Parameters match the assignment defaults. Per-bit mutation probability is
    ``1 / genome_length`` so a mutated individual changes about one bit, and
    tournament size 3 keeps selection pressure moderate. The current best
    individual is copied into the next generation so a good coloring is not
    lost to crossover or mutation.
    """
    graph = init_graph(file_path)
    bits = _bits_per_node(graph.k)
    genome_length = graph.n * bits
    if genome_length <= 0 or population_size <= 0:
        return 1.0

    def make_individual():
        return _random_individual(creator.Individual, graph.n, graph.k, bits)

    def mate(child1, child2):
        tools.cxTwoPoint(child1, child2)
        _repair(child1, graph.n, graph.k, bits)
        _repair(child2, graph.n, graph.k, bits)
        return child1, child2

    def mutate(individual):
        tools.mutFlipBit(individual, indpb=1.0 / genome_length)
        _repair(individual, graph.n, graph.k, bits)
        return (individual,)

    toolbox = base.Toolbox()
    toolbox.register("clone", copy.deepcopy)
    toolbox.register("individual", make_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_graph, graph)
    toolbox.register("mate", mate)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=population_size)
    halloffame = tools.HallOfFame(1)

    def evaluate(pop):
        for individual in pop:
            individual.fitness.values = toolbox.evaluate(individual)

    evaluate(population)
    halloffame.update(population)

    for _ in range(ngen):
        offspring = [
            toolbox.clone(individual)
            for individual in toolbox.select(population, len(population))
        ]

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)

        offspring[0] = toolbox.clone(halloffame[0])

        evaluate(offspring)
        population[:] = offspring
        halloffame.update(population)

    return float(halloffame[0].fitness.values[0])


if __name__ == "__main__":
    import os

    graph_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")
    for name in sorted(os.listdir(graph_dir)):
        if not name.endswith(".txt"):
            continue
        path = os.path.join(graph_dir, name)
        print(f"{name}: {run_ga(path):.4f}")
