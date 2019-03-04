"""
RHD NN training on Digits data

"""
import sys
sys.path.append("/Users/jenny/Documents/_Homework/cs7641/hw2/test_folder/ABAGAIL/ABAGAIL.jar")
import os
import csv
import time
from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem
from func.nn.backprop import RPROPUpdateRule, BatchBackPropagationTrainer
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
from func.nn.activation import RELU
import logging

# Network parameters
INPUT_LAYER = 65
HIDDEN_LAYER1 = 50
HIDDEN_LAYER2 = 10
HIDDEN_LAYER3 = 10
OUTPUT_LAYER = 1
TRAINING_ITERATIONS = 500

def initialize_instances(infile):
    """Read the m_trg.csv CSV data into a list of instances."""
    instances = []

    # Read in the CSV file
    with open(infile, "r") as dat:
        reader = csv.reader(dat)
        for row in reader:
            instance = Instance([float(value) for value in row[1:-1]])
            instance.setLabel(Instance(float(row[-1])))
            instances.append(instance)

    return instances


def errorOnDataSet(network,ds,measure):
    N = len(ds)
    error = 0.
    correct = 0
    incorrect = 0

    count = 0
    for instance in ds:
        count +=1
        network.setInputValues(instance.getData())
        network.run()
        actual = instance.getLabel().getContinuous()
        predicted = network.getOutputValues().get(0)
        predicted = max(min(predicted,1),0)
        if abs(predicted - actual) < 0.5:
            correct += 1
        else:
            incorrect += 1
        output = instance.getLabel()
        output_values = network.getOutputValues()
        example = Instance(output_values, Instance(output_values.get(0)))
        error += measure.value(output, example)
    MSE = error/float(N)
    acc = correct/float(correct+incorrect)
    return MSE,acc


def train(oa, network, oaName, training_ints,validation_ints,testing_ints, measure):
    """Train a given network on a set of instances.
    """
    print ("\nError results for %s\n---------------------------" % (oaName,))
    times = [0]
    for iteration in xrange(TRAINING_ITERATIONS):
        print ("This is iteration number", iteration)
        start = time.clock()
        oa.train()
        elapsed = time.clock()-start
        times.append(times[-1]+elapsed)
        if iteration % 10 == 0:
            MSE_trg, acc_trg = errorOnDataSet(network,training_ints,measure)
            MSE_val, acc_val = errorOnDataSet(network,validation_ints,measure)
            MSE_tst, acc_tst = errorOnDataSet(network,testing_ints,measure)
            # txt = '{},{},{},{},{},{},{},{}\n'.format(iteration,MSE_trg,MSE_trg,MSE_trg,acc_trg,acc_trg,acc_trg,times[-1])
            txt = '{},{},{},{},{},{},{},{}\n'.format(iteration,MSE_trg,MSE_val,MSE_tst,acc_trg,acc_val,acc_tst,times[-1])
            print (txt)
            with open(oaName,'a+') as f:
                f.write(txt)

def main(output_filename):
    """Run this experiment"""
    training_ints = initialize_instances('out_digits_train.csv')
    testing_ints = initialize_instances('out_digits_test.csv')
    validation_ints = initialize_instances('out_digits_test.csv')

    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(training_ints)

    relu = RELU()
    rule = RPROPUpdateRule()

    with open(output_filename,'w') as f:
        f.write('{},{},{},{},{},{},{},{}\n'.format('iteration','MSE_trg','MSE_val','MSE_tst','acc_trg','acc_val','acc_tst','elapsed'))
    classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER1,HIDDEN_LAYER2,HIDDEN_LAYER3, OUTPUT_LAYER],relu)
    nnop = NeuralNetworkOptimizationProblem(data_set, classification_network, measure)
    # oa = SimulatedAnnealing(1E10, 0.95, nnop)
    oa = RandomizedHillClimbing(nnop)

    train(oa, classification_network, output_filename, training_ints,validation_ints,testing_ints, measure)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')


    output_path = '/Users/jenny/Documents/_Homework/cs7641/hw2/test_folder/output'
    output_filename = '{}/testing_outputs_digits_SA.csv'.format(output_path)

    main(output_filename)