#! /usr/bin/env python3
# coding: utf-8

import sys
from sys import argv
sys.path.insert(0, '../ELINA/python_interface/')


import numpy as np
import re
import csv
import pdb
from elina_box import *
from elina_interval import *
from elina_abstract0 import *
from elina_manager import *
from elina_dimension import *
from elina_scalar import *
from elina_interval import *
from elina_linexpr0 import *
from elina_lincons0 import *
import ctypes
from ctypes.util import find_library
from gurobipy import *
import time

libc = CDLL(find_library('c'))
cstdout = c_void_p.in_dll(libc, 'stdout')


class layers:
    def __init__(self):
        self.layertypes = []
        self.weights = []
        self.biases = []
        self.numlayer = 0
        self.ffn_counter = 0


def parse_bias(text):
    if len(text) < 1 or text[0] != '[':
        raise Exception("expected '['")
    if text[-1] != ']':
        raise Exception("expected ']'")
    v = np.array([*map(lambda x: np.double(x.strip()), text[1:-1].split(','))])
    # return v.reshape((v.size,1))
    return v


def parse_vector(text):
    if len(text) < 1 or text[0] != '[':
        raise Exception("expected '['")
    if text[-1] != ']':
        raise Exception("expected ']'")
    v = np.array([*map(lambda x: np.double(x.strip()), text[1:-1].split(','))])
    return v.reshape((v.size, 1))
    # return v


def balanced_split(text):
    i = 0
    bal = 0
    start = 0
    result = []
    while i < len(text):
        if text[i] == '[':
            bal += 1
        elif text[i] == ']':
            bal -= 1
        elif text[i] == ',' and bal == 0:
            result.append(text[start:i])
            start = i+1
        i += 1
    if start < i:
        result.append(text[start:i])
    return result


def parse_matrix(text):
    i = 0
    if len(text) < 1 or text[0] != '[':
        raise Exception("expected '['")
    if text[-1] != ']':
        raise Exception("expected ']'")
    return np.array([*map(lambda x: parse_vector(x.strip()).flatten(), balanced_split(text[1:-1]))])


def parse_net(text):
    lines = [*filter(lambda x: len(x) != 0, text.split('\n'))]
    i = 0
    res = layers()
    while i < len(lines):
        if lines[i] in ['ReLU', 'Affine']:
            W = parse_matrix(lines[i+1])
            b = parse_bias(lines[i+2])
            res.layertypes.append(lines[i])
            res.weights.append(W)
            res.biases.append(b)
            res.numlayer += 1
            i += 3
        else:
            raise Exception('parse error: '+lines[i])
    return res


def parse_spec(text):
    text = text.replace("[", "")
    text = text.replace("]", "")
    with open('dummy', 'w') as my_file:
        my_file.write(text)
    data = np.genfromtxt('dummy', delimiter=',', dtype=np.double)
    low = np.copy(data[:, 0])
    high = np.copy(data[:, 1])
    return low, high


def get_perturbed_image(x, epsilon):
    image = x[1:len(x)]
    num_pixels = len(image)
    LB_N0 = image - epsilon
    UB_N0 = image + epsilon

    for i in range(num_pixels):
        if(LB_N0[i] < 0):
            LB_N0[i] = 0
        if(UB_N0[i] > 1):
            UB_N0[i] = 1
    return LB_N0, UB_N0


def generate_linexpr0(weights, bias, size):
    linexpr0 = elina_linexpr0_alloc(
        ElinaLinexprDiscr.ELINA_LINEXPR_DENSE, size)
    cst = pointer(linexpr0.contents.cst)
    elina_scalar_set_double(cst.contents.val.scalar, bias)
    for i in range(size):
        elina_linexpr0_set_coeff_scalar_double(linexpr0, i, weights[i])
    return linexpr0


def elina_bounds(nn, LB_N0, UB_N0, start):
    start_size = len(LB_N0)
    nn.ffn_counter = start
    numlayer = nn.numlayer

    hi_lower_bounds = []
    hi_upper_bounds = []

    man = elina_box_manager_alloc()
    itv = elina_interval_array_alloc(start_size)
    for i in range(start_size):
        elina_interval_set_double(itv[i], LB_N0[i], UB_N0[i])

    # construct input abstraction
    element = elina_abstract0_of_box(man, 0, start_size, itv)
    elina_interval_array_free(itv, start_size)
    for layerno in range(start, numlayer):
        if(nn.layertypes[layerno] in ['ReLU', 'Affine']):
            weights = nn.weights[nn.ffn_counter]
            biases = nn.biases[nn.ffn_counter]
            dims = elina_abstract0_dimension(man, element)
            num_in_pixels = dims.intdim + dims.realdim
            num_out_pixels = len(weights)

            dimadd = elina_dimchange_alloc(0, num_out_pixels)
            for i in range(num_out_pixels):
                dimadd.contents.dim[i] = num_in_pixels
            elina_abstract0_add_dimensions(man, True, element, dimadd, False)
            elina_dimchange_free(dimadd)
            np.ascontiguousarray(weights, dtype=np.double)
            np.ascontiguousarray(biases, dtype=np.double)
            var = num_in_pixels

            # handle affine layer
            for i in range(num_out_pixels):
                tdim = ElinaDim(var)
                linexpr0 = generate_linexpr0(
                    weights[i], biases[i], num_in_pixels)
                element = elina_abstract0_assign_linexpr_array(
                    man, True, element, tdim, linexpr0, 1, None)
                var += 1
            dimrem = elina_dimchange_alloc(0, num_in_pixels)
            for i in range(num_in_pixels):
                dimrem.contents.dim[i] = i
            elina_abstract0_remove_dimensions(man, True, element, dimrem)
            elina_dimchange_free(dimrem)

            # get bounds for each neuron
            bounds = elina_abstract0_to_box(man, element)

            layer_lower_bounds = np.zeros(num_out_pixels)
            layer_upper_bounds = np.zeros(num_out_pixels)

            for i in range(num_out_pixels):
                layer_lower_bounds[i] = bounds[i].contents.inf.contents.val.dbl
                layer_upper_bounds[i] = bounds[i].contents.sup.contents.val.dbl

            hi_lower_bounds.append(layer_lower_bounds)
            hi_upper_bounds.append(layer_upper_bounds)

            elina_interval_array_free(bounds, num_out_pixels)

            # handle ReLU layer
            if(nn.layertypes[layerno] == 'ReLU'):
                element = relu_box_layerwise(
                    man, True, element, 0, num_out_pixels)
            nn.ffn_counter += 1
        else:
            print(' net type not supported')

    out_lower_bounds = np.zeros(num_out_pixels)
    out_upper_bounds = np.zeros(num_out_pixels)

    bounds = elina_abstract0_to_box(man, element)

    for i in range(num_out_pixels):
        out_lower_bounds[i] = bounds[i].contents.inf.contents.val.dbl
        out_upper_bounds[i] = bounds[i].contents.sup.contents.val.dbl

    elina_interval_array_free(bounds, num_out_pixels)

    elina_abstract0_free(man, element)
    elina_manager_free(man)

    return hi_lower_bounds, hi_upper_bounds, out_lower_bounds, out_upper_bounds


def gurobi_bounds(nn, lower_bounds, upper_bounds, steps):
    m = Model(name='NN')
    m.setParam('OutputFlag', False)

    # Set variables

    his = [[(m.addVar(lb=lower_bounds[i][j], ub=upper_bounds[i][j], vtype=GRB.CONTINUOUS, name='h_' + str(i) + '_' + str(j))
             if i == 0 else
             m.addVar(lb=-np.inf, vtype=GRB.CONTINUOUS, name='h_' + str(i) + '_' + str(j)))
            for j in range(lower_bounds[i].size)]
           for i in range(steps)]

    ris = [[m.addVar(lb=-np.inf, vtype=GRB.CONTINUOUS, name='r_' + str(i) + '_' + str(j))
            if nn.layertypes[i] in ['ReLU'] else his[i][j]
            for j in range(lower_bounds[i].size)]
           for i in range(steps - 1)]

    # Weights & biases operation

    for i in range(1, steps):
        for j in range(lower_bounds[i].size):
            m.addConstr(his[i][j] == LinExpr(
                nn.weights[i][j, :], ris[i-1]) + nn.biases[i][j])

    for i in range(0, steps - 1):
        if nn.layertypes[i] not in ['ReLU']:
            continue
        for j in range(lower_bounds[i].size):
            inf, sup = lower_bounds[i][j], upper_bounds[i][j]

            if (inf >= 0):
                m.addConstr(ris[i][j] == his[i][j])
            elif (sup <= 0):
                m.addConstr(ris[i][j] == 0)
            else:
                k, t = sup / (sup - inf), -sup * inf / (sup - inf)

                m.addConstr(ris[i][j] >= 0)
                m.addConstr(ris[i][j] >= his[i][j])
                m.addConstr(ris[i][j] <= k * his[i][j] + t)

    output_lower_bounds, output_upper_bounds = (
        [None for _ in enumerate(his[-1])],
        [None for _ in enumerate(his[-1])]
    )

    for i in range(len(his[-1])):
        m.setObjective(his[-1][i], GRB.MINIMIZE)
        m.write("models/model_c_min.lp")
        m.optimize()
        try:
            output_lower_bounds[i] = m.objVal
        except:
            print(f"Can't find lower bound for neuron {i}")

        m.setObjective(his[-1][i], GRB.MAXIMIZE)
        m.write("models/model_c_max.lp")
        m.optimize()
        try:
            output_upper_bounds[i] = m.objVal
        except:
            print(f"Can't find upper bound for neuron {i}")

    # Apply box relu for the output
    if nn.layertypes[steps - 1] in ['ReLU']:
        for i in range(len(his[-1])):
            output_lower_bounds[i] = max(output_lower_bounds[i], 0)
            output_upper_bounds[i] = max(output_upper_bounds[i], 0)

    return output_lower_bounds, output_upper_bounds


def predict_label(lb, ub):
    nr_labels = len(lb)
    predicted_label = 0
    predicted_flag = False

    for i in range(nr_labels):
        flag = True
        for j in range(nr_labels):
            if(j != i):
                if(lb[i] <= ub[j]):
                    flag = False
                    break

        if(flag):
            predicted_label = i
            predicted_flag = True
            break

    return predicted_label, predicted_flag


def verify(lb, ub, label):
    nr_labels = len(lb)
    verified = True

    for j in range(nr_labels):
        if(j != label):
            if(lb[label] <= ub[j]):
                verified = False
                break

    return verified


if __name__ == '__main__':
    if len(argv) < 3:
        print('usage: python3.6 ' + argv[0] + ' net.txt spec.txt [timeout]')
        exit(1)

    netname = argv[1]
    specname = argv[2]
    epsilon = float(argv[3])
    gurobi_steps = int(argv[4])

    with open(netname, 'r') as netfile:
        netstring = netfile.read()

    with open(specname, 'r') as specfile:
        specstring = specfile.read()

    nn = parse_net(netstring)
    x0_low, x0_high = parse_spec(specstring)

    # Image without perturbations
    lb_clear, ub_clear = get_perturbed_image(x0_low, 0)
    _, _, out_lower, out_upper = elina_bounds(nn, lb_clear, ub_clear, 0)
    predicted_label, predicted_flag = predict_label(out_lower, out_upper)

    if(not predicted_flag):
        print('Image not correctly classified by the network')
        print('Expected label:', int(x0_low[0]))
        print('Classified label:', predicted_label)
        sys.exit()

    print('Classified label:', predicted_label)

    # Image with perturbations
    lb_noisy, ub_noisy = get_perturbed_image(x0_low, epsilon)

    start = time.time()

    hi_lower, hi_upper, out_lower, out_upper = elina_bounds(
        nn, lb_noisy, ub_noisy, 0)
    # print('ELINA')
    # print('Lower hi:', hi_lower[-1])
    # print('Upper hi:', hi_upper[-1])
    # print('Lower out:', out_lower)
    # print('Upper out:', out_upper)

    out_lower, out_upper = gurobi_bounds(
        nn, hi_lower, hi_upper, gurobi_steps)

    # print('GUROBI')
    # print('Lower out:', out_lower)
    # print('Upper out:', out_upper)

    if(gurobi_steps < nn.numlayer):
        hi_lower, hi_upper, out_lower, out_upper = elina_bounds(
            nn, out_lower, out_upper, gurobi_steps)
        # print('Mix')

    print('Lower out:', out_lower)
    print('Upper out:', out_upper)

    verified = verify(out_lower, out_upper, predicted_label)

    end = time.time()

    print('Verified') if verified else print('Cannot be verified')
    print('Analysis time:', (end-start), 'seconds')
