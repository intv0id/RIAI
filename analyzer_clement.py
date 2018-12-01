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


def get_bounds(nn, LB_N0, UB_N0):
    num_pixels = len(LB_N0)
    nn.ffn_counter = 0
    numlayer = nn.numlayer

    lower_bounds = []
    upper_bounds = []

    man = elina_box_manager_alloc()
    itv = elina_interval_array_alloc(num_pixels)
    for i in range(num_pixels):
        elina_interval_set_double(itv[i], LB_N0[i], UB_N0[i])

    # construct input abstraction
    element = elina_abstract0_of_box(man, 0, num_pixels, itv)
    elina_interval_array_free(itv, num_pixels)
    for layerno in range(numlayer):
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

            nn.ffn_counter += 1

            # get bounds for each neuron
            bounds = elina_abstract0_to_box(man, element)

            layer_lower_bounds = np.zeros(num_out_pixels)
            layer_upper_bounds = np.zeros(num_out_pixels)

            for i in range(num_out_pixels):
                layer_lower_bounds[i] = bounds[i].contents.inf.contents.val.dbl
                layer_upper_bounds[i] = bounds[i].contents.sup.contents.val.dbl

            lower_bounds.append(layer_lower_bounds)
            upper_bounds.append(layer_upper_bounds)

            elina_interval_array_free(bounds, num_out_pixels)
        else:
            print(' net type not supported')

    elina_abstract0_free(man, element)
    elina_manager_free(man)

    return lower_bounds, upper_bounds


def gurobi_bounds(nn, lower_bounds, upper_bounds):
    m = Model(name='NN')
    m.setParam('OutputFlag', False)

    # Set variables

    his = [[(m.addVar(lb=lower_bounds[i][j],ub=upper_bounds[i][j],vtype=GRB.CONTINUOUS, name='h' + str(i) + str(j))
            if i == 0 else 
            m.addVar(lb=-np.inf, vtype=GRB.CONTINUOUS, name='h' + str(i) + str(j)))
        for j in range(lower_bounds[i].size)] 
        for i in range(len(lower_bounds))]


    ris = [[m.addVar(lb=-np.inf, vtype=GRB.CONTINUOUS, name='r' + str(i) + str(j)) 
        for j in range(lower_bounds[i].size)] 
        for i in range(len(lower_bounds) -1)]

    m.update()

    # Weights & biases operation

    for i in range(1, len(lower_bounds)):
        for j in range(lower_bounds[i].size):
            m.addConstr(his[i][j] == LinExpr(nn.weights[i][j, :], ris[i-1]) + nn.biases[i][j])

    m.update()

    for i in range(0, len(lower_bounds)-1):
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

    m.update()

    output_lower_bounds, output_upper_bounds = (
        [None for _ in enumerate(his[-1])], 
        [None for _ in enumerate(his[-1])]
    )

    for i in range(len(his[-1])):
        m.reset()
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

    return output_lower_bounds, output_upper_bounds


if __name__ == '__main__':
    if len(argv) < 3 or len(argv) > 4:
        print('usage: python3.6 ' + argv[0] + ' net.txt spec.txt [timeout]')
        exit(1)

    netname = argv[1]
    specname = argv[2]
    epsilon = float(argv[3])

    with open(netname, 'r') as netfile:
        netstring = netfile.read()

    with open(specname, 'r') as specfile:
        specstring = specfile.read()

    nn = parse_net(netstring)
    x0_low, x0_high = parse_spec(specstring)
    LB_N0, UB_N0 = get_perturbed_image(x0_low, epsilon)

    lower_bounds, upper_bounds = get_bounds(nn, LB_N0, UB_N0)

    output_lower_bounds, output_upper_bounds = gurobi_bounds(
        nn, lower_bounds, upper_bounds)

    print(output_lower_bounds, output_upper_bounds)