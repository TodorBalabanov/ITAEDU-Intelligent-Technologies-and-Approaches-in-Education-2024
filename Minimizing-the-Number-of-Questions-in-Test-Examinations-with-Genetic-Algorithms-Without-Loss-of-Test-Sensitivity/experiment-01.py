import array
import random
import time
import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from datetime import datetime

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_bool", random.randint, 0, 1)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 1000)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Fitness function
def evalOneMax(individual):
    return sum(individual),

# Time measurements
time_start = time.process_time()
def interval(individual):
    return time.process_time() - time_start

toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=1, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Single entry point
def main():
    random.seed(datetime.now().timestamp())

    population = toolbox.population(n=39)
    best = tools.HallOfFame(1)
    statistics = tools.Statistics(lambda ind: ind.fitness.values)
    statistics.register("sec", interval)
    statistics.register("avg", numpy.mean)
    statistics.register("std", numpy.std)
    statistics.register("min", numpy.min)
    statistics.register("max", numpy.max)

    population, log = algorithms.eaSimple(population, toolbox, cxpb=0.95, mutpb=0.02, ngen=10000, stats=statistics, halloffame=best, verbose=True)

    return population, log, best

if __name__ == "__main__":
    population, log, best = main()
    print( best )
