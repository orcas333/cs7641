import sys
import os
import time
# import numpy as np
# import pandas as pd


sys.path.append("/Users/jenny/Documents/_Homework/cs7641/hw2/test_folder/ABAGAIL/ABAGAIL.jar")
import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random


import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.EvaluationFunction as EvaluationFunction
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
import opt.ga.CrossoverFunction as CrossoverFunction
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.prob.ProbabilisticOptimizationProblem as ProbabilisticOptimizationProblem
import shared.FixedIterationTrainer as FixedIterationTrainer

from array import array

def run_rhc(hcp, ef, iterations=200000):

    rhc = RandomizedHillClimbing(hcp)
    fit = FixedIterationTrainer(rhc, iterations)
    fit.train()

    optimal_result = str(ef.value(rhc.getOptimal()))
    print "RHC: " + optimal_result

    return optimal_result, iterations


def run_sa(hcp, ef, iterations=200000):
    sa = SimulatedAnnealing(1E11, .95, hcp)
    fit = FixedIterationTrainer(sa, iterations)
    fit.train()

    optimal_result = str(ef.value(sa.getOptimal()))
    print "SA: " + optimal_result

    return optimal_result, iterations

def run_ga(gap, ef, iterations=1000):

    ga = StandardGeneticAlgorithm(200, 100, 10, gap)
    fit = FixedIterationTrainer(ga, iterations)
    fit.train()
    optimal_result = str(ef.value(ga.getOptimal()))
    print "GA: " + optimal_result

    return optimal_result, iterations


def run_mimic(pop, ef, iterations=1000):

    mimic = MIMIC(200, 20, pop)
    fit = FixedIterationTrainer(mimic, iterations)
    fit.train()
    optimal_result = str(ef.value(mimic.getOptimal()))
    print "MIMIC: " + optimal_result

    return optimal_result, iterations


def run_algorithm_test(T, ranges, algorithms, output_file_name, trial_number, iterations=False):

    with open(output_file_name,'w') as f:
        f.write('algorithm,optimal_result,iterations,time,trial\n')

    ef = FourPeaksEvaluationFunction(T)
    odd = DiscreteUniformDistribution(ranges)
    nf = DiscreteChangeOneNeighbor(ranges)
    mf = DiscreteChangeOneMutation(ranges)
    cf = SingleCrossOver()
    df = DiscreteDependencyTree(.1, ranges)

    hcp = GenericHillClimbingProblem(ef, odd, nf)
    gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
    pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

    for trial in range(trial_number):
        if iterations is False:
            for item in algorithms:
                start_time = time.time()
                if item in ['rhc']:
                    optimal_result, run_iters = run_rhc(hcp, ef)
                elif item in ['sa']:
                    optimal_result, run_iters = run_sa(hcp, ef)
                elif item in ['ga']:
                    optimal_result, run_iters = run_ga(gap, ef)
                elif item in ['mimic']:
                    optimal_result, run_iters = run_mimic(pop, ef)
                else:
                    print "The algorithm type {} is not supported.".format(item)

                end_time = time.time()
                time_elapsed = end_time - start_time

                run_output = '{},{},{},{},{}\n'.format(item, optimal_result, run_iters, time_elapsed, trial)
                with open(output_file_name,'a') as f:
                    f.write(run_output)
        else:
            for iter in iterations:
                for item in algorithms:
                    start_time = time.time()
                    if item in ['rhc']:
                        optimal_result, run_iters = run_rhc(hcp, ef, iter)
                    elif item in ['sa']:
                        optimal_result, run_iters = run_sa(hcp, ef, iter)
                    elif item in ['ga']:
                        optimal_result, run_iters = run_ga(gap, ef, iter)
                    elif item in ['mimic']:
                        optimal_result, run_iters = run_mimic(pop, ef, iter)
                    else:
                        print "The algorithm type {} is not supported.".format(item)

                    end_time = time.time()
                    time_elapsed = end_time - start_time

                    run_output = '{},{},{},{},{}\n'.format(item, optimal_result, run_iters, time_elapsed, trial)

                    with open(output_file_name,'a') as f:
                        f.write(run_output)

    print "time elapsed is {}".format(time_elapsed)

    return

if __name__ == '__main__':

    output_path = '/Users/jenny/Documents/_Homework/cs7641/hw2/test_folder/output'

    N=130 #200
    T=N/5
    fill = [2] * N
    ranges = array('i', fill)
    trial_number = 15
    # iteration_list = [10, 1000, 5000, 200000]
    # iteration_list = [10, 100, 300, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    # iteration_list = [10, 30, 50, 70, 100, 150, 180, 200, 230, 250, 270, 300]
    iteration_list = [10, 30, 50, 70, 100, 150, 180, 200, 230, 250, 270, 300, 330, 350, 370, 400, 430, 450, 470, 500]

    # for trial in range(trial_number):
    output_file_name = '{}/output_fourpeaks_N130_runs{}.csv'.format(output_path,trial_number)
    run_algorithm_test(T=T,
                       ranges=ranges,
                       algorithms=['rhc', 'sa', 'ga', 'mimic'],
                       output_file_name=output_file_name,
                       trial_number=trial_number,
                       iterations=iteration_list)