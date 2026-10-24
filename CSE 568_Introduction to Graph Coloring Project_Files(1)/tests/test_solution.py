import os

import pytest

import student_solution as sol


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPHS = os.path.join(ROOT, "graphs")


def adjacency(graph, node):
    result = []
    current = graph.graph[node]
    while current is not None:
        result.append(current.vertex)
        current = current.next
    return sorted(result)


def test_init_graph_builds_graph_1():
    graph = sol.init_graph(os.path.join(GRAPHS, "graph_1.txt"))
    assert graph.n == 4
    assert graph.k == 3
    assert adjacency(graph, 0) == [2]
    assert adjacency(graph, 1) == [2, 3]
    assert adjacency(graph, 2) == [0, 1]
    assert adjacency(graph, 3) == [1]


def test_init_graph_builds_graph_2():
    graph = sol.init_graph(os.path.join(GRAPHS, "graph_2.txt"))
    assert graph.n == 5
    assert graph.k == 3
    assert adjacency(graph, 0) == [2, 3, 4]
    assert adjacency(graph, 1) == [3, 4]
    assert adjacency(graph, 2) == [0, 3]
    assert adjacency(graph, 3) == [0, 1, 2, 4]
    assert adjacency(graph, 4) == [0, 1, 3]


def test_decode_and_eval_graph_1_cases():
    graph = sol.init_graph(os.path.join(GRAPHS, "graph_1.txt"))
    perfect = [0, 0, 0, 0, 0, 1, 1, 0]
    invalid = [1, 1, 0, 0, 0, 1, 1, 0]
    partial = [0, 0, 0, 1, 0, 1, 1, 0]

    assert sol.decode_individual(graph, perfect) == [0, 0, 1, 2]
    assert sol.decode_individual(graph, invalid) is None
    assert sol.decode_individual(graph, partial) == [0, 1, 1, 2]

    assert sol.eval_graph(graph, perfect) == pytest.approx((1.0,))
    assert sol.eval_graph(graph, invalid) == pytest.approx((0.0,))
    assert sol.eval_graph(graph, partial) == pytest.approx((2.0 / 3.0,))


def test_eval_graph_rejects_wrong_length():
    graph = sol.init_graph(os.path.join(GRAPHS, "graph_1.txt"))
    assert sol.eval_graph(graph, [0, 1, 0]) == (0.0,)


def test_run_ga_reaches_high_fitness_on_graph_3():
    fitness = sol.run_ga(os.path.join(GRAPHS, "graph_3.txt"), population_size=200, ngen=100)
    assert isinstance(fitness, float)
    assert fitness >= 0.90
