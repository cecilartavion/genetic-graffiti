import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
from deap import base,creator,tools,gp, algorithms
import random
import operator
import math
import numpy
import networkx as nx

import shelve
from tabulate import tabulate
import grinpy as gr




'''
Create division function.
'''
def protectedDiv(left,right):
   try:
      return left/right
   except ZeroDivisionError:
      return 1

'''
Generate all of the operations to be used between two invariants.
'''
pset = gp.PrimitiveSet("MAIN", 2)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub,2)
pset.addPrimitive(operator.mul,2)
pset.addPrimitive(protectedDiv,2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(max,2)
pset.addPrimitive(min,2)
pset.addEphemeralConstant("rand101", lambda: random.randint(0,5))


'''
Generate set of all the graphs.
'''
graph_set = []
for N in range(4,9):
    graph_set = graph_set + nx.read_graph6("graph{}c.g6".format(N))

#We could add other classes of graphs pretty easily.
#cycles = [nx.cycle_graph(n) for n in range(3,20)]
#paths = [nx.path_graph(n) for n in range(3,20)]
#More graph generators at
#https://networkx.github.io/documentation/networkx-1.10/reference/generators.html

'''
Turn this into a data set like what we used for the digits training examples we were doing in October.
Note that we use two functions because we have two variables above.
'''

#We could add other functions pretty easily.
#See: https://networkx.github.io/documentation/networkx-1.10/reference/algorithms.html
#Most of those would give quantitative features of a graph, although in some cases
#a bit of modification may be needed to produce a number rather than a subgraph.


X_train = [[nx.graph_clique_number(hc),nx.diameter(hc)]for hc in graph_set]
Y_train = [nx.radius(hc) for hc in graph_set]

#set up the fitness function
#this gives the number of times that func perfectly predicts the output
#other fitness functions might work better

def evalSymbReg(individual):
   #Transform the tree expression into a callable function
   func = toolbox.compile(expr=individual)
   hits = sum([func(*X_train[i]) > Y_train[i] for i in range(len(Y_train))])
   return hits,


#We need an individual with a genotype

creator.create("FitnessMax", base.Fitness, weights = (1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

#Add some parameters from the toolbox

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_ = 2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evalSymbReg)
toolbox.register("select", tools.selTournament, tournsize = 3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_ = 0, max_ = 2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value = 17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value = 17))

#now we set up statistics

stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg",numpy.mean)
mstats.register("std",numpy.std)
mstats.register("min",numpy.min)
mstats.register("max",numpy.max)

pop = toolbox.population(n=300)
hof = tools.HallOfFame(10)
pop, log = algorithms.eaSimple(pop, toolbox, 0.5,0.1,20, stats=mstats, halloffame=hof,verbose=True)

#print hof
#print([str(x) for x in hof])
for x in hof:
   print(str(x) + "\n")
   print(str(evalSymbReg(x)) + "\n")


