import numpy as np
import math
from funkcje_coco import fun_list, fun_name_list

################
# Algorithms
################

def CDE(func_adapt, bounds, F=0.8, CR=0.7, vectors_num=5, iterations=1000):
    vector_len = len(bounds)
    population = np.random.rand(vectors_num, vector_len)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    population_denorm = min_b + population * diff
    fitness = np.asarray([func_adapt(ind) for ind in population_denorm])
    best_idx = np.argmin(fitness)
    best = population_denorm[best_idx]
    for i in range(iterations):
        for j in range(vectors_num):
            indexes_without_j = [idx for idx in range(vectors_num) if idx != j]
            a, b, c = population[np.random.choice(indexes_without_j, 3, replace = False)]
            mutant = np.clip(a + F * (b - c), 0, 1)
            cross_points = np.random.rand(vector_len) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, vector_len)] = True
            trial = np.where(cross_points, mutant, population[j])
            trial_denorm = min_b + trial * diff
            f = func_adapt(trial_denorm)
            if f < fitness[j]:
                fitness[j] = f
                population[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        yield best, fitness[best_idx]


def BBDE(func_adapt, bounds, vectors_num=5, iterations=1000):
    vector_len = len(bounds)
    population = np.random.rand(vectors_num, vector_len)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    population_denorm = min_b + population * diff
    fitness = np.asarray([func_adapt(ind) for ind in population_denorm])
    best_idx = np.argmin(fitness)
    best = population_denorm[best_idx]
    for i in range(iterations):
        for j in range(vectors_num):
            indexes_without_j = [idx for idx in range(vectors_num) if idx != j]
            b, c = population[np.random.choice(indexes_without_j, 2, replace = False)]
            mutant = np.clip(population[j] + math.exp(np.random.normal()) * (b - c), 0, 1)
            mutant_denorm =  min_b + mutant * diff
            f = func_adapt(mutant_denorm)
            if f < fitness[j]:
                fitness[j] = f
                population[j] = mutant
                if f < fitness[best_idx]:
                    best_idx = j
                    best = mutant_denorm
        yield best, fitness[best_idx]

