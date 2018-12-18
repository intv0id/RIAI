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


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


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


def elina_bounds(nn, LB_N0, UB_N0, start=0, finish=None):
    start_size = len(LB_N0)
    numlayer = finish if finish != None else nn.numlayer

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
            weights = nn.weights[layerno]
            biases = nn.biases[layerno]
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


def gurobi_create_model(nn, lb, ub):
    m = Model(name='NN')
    m.setParam('OutputFlag', False)
    m.setParam('Threads', 1)

    ris = []
    # Start with the input layer bounds
    for i in range(len(lb[0])):
        ri = m.addVar(lb=lb[0][i], ub=ub[0][i],
                      vtype=GRB.CONTINUOUS, name='i_' + str(i))
        ris.append(ri)

    for i in range(1, len(lb)):
        his = []
        # Add the next layer constraints using (weights, biases, previous layer (ReLU | Affine))
        for j in range(len(lb[i])):
            hi = m.addVar(
                lb=lb[i][j], ub=ub[i][j], vtype=GRB.CONTINUOUS, name='h_' + str(i) + '_' + str(j))
            m.addConstr(hi == LinExpr(
                nn.weights[i - 1][j, :], ris) + nn.biases[i - 1][j])
            his.append(hi)

        # Add the ReLU approximation if required
        if nn.layertypes[i - 1] in ['ReLU']:
            ris = []

            for j in range(len(lb[i])):
                inf = lb[i][j]
                sup = ub[i][j]

                ri = m.addVar(lb=0, vtype=GRB.CONTINUOUS,
                              name='r_' + str(i) + '_' + str(j))

                if inf >= 0:
                    m.addConstr(ri == his[j])
                elif sup <= 0:
                    m.addConstr(ri == 0)
                else:
                    k, t = sup / (sup - inf), -sup * inf / (sup - inf)

                    m.addConstr(ri >= his[j])
                    m.addConstr(ri <= k * his[j] + t)

                ris.append(ri)
        else:
            ris = his

    his = []
    last_index = len(lb) - 1
    # Write the constraints for the layer we are trying to predict
    for i in range(len(nn.biases[last_index])):
        hi = m.addVar(lb=-np.inf, vtype=GRB.CONTINUOUS, name='o_' + str(i))
        m.addConstr(hi == LinExpr(
            nn.weights[last_index][i, :], ris) + nn.biases[last_index][i])
        his.append(hi)

    return m, his


def gurobi_optimize_bounds(m, his, is_relu):
    out_lb = np.zeros(len(his))
    out_ub = np.zeros(len(his))

    for i in range(len(his)):
        # s = time.clock()
        m.setObjective(his[i], GRB.MINIMIZE)
        m.optimize()
        try:
            out_lb[i] = m.objVal
        except:
            print(f"Can't find lower bound for neuron {i}")

        m.setObjective(his[i], GRB.MAXIMIZE)
        m.optimize()
        try:
            out_ub[i] = m.objVal
        except:
            print(f"Can't find upper bound for neuron {i}")
        # t = time.clock()
        # print(f'Optimize neuron {i}: {t - s}')

    out_lb_relu = np.copy(out_lb)
    out_ub_relu = np.copy(out_ub)

    # Apply box relu for the output
    if is_relu:
        for i in range(len(his)):
            out_lb_relu[i] = max(out_lb[i], 0)
            out_ub_relu[i] = max(out_ub[i], 0)

    return out_lb, out_ub, out_lb_relu, out_ub_relu


def gurobi_optimize_last(m, his, is_relu, label):
    out_lb = np.zeros(len(his))
    out_ub = np.zeros(len(his))

    for i in range(len(his)):
        # s = time.clock()
        if i == label:
            m.setObjective(his[i], GRB.MINIMIZE)
            m.optimize()
            try:
                out_lb[i] = m.objVal
            except:
                print(f"Can't find lower bound for neuron {i}")
        else:
            m.setObjective(his[i], GRB.MAXIMIZE)
            m.optimize()
            try:
                out_ub[i] = m.objVal
            except:
                print(f"Can't find upper bound for neuron {i}")
        # t = time.clock()
        # print(f'Optimize o{i}: {t - s}')

    out_lb_relu = np.copy(out_lb)
    out_ub_relu = np.copy(out_ub)

    # Apply box relu for the output
    if is_relu:
        for i in range(len(his)):
            out_lb_relu[i] = max(out_lb[i], 0)
            out_ub_relu[i] = max(out_ub[i], 0)

    return out_lb, out_ub, out_lb_relu, out_ub_relu


def gurobi_bounds(nn, lb, ub, label):
    m, his = gurobi_create_model(nn, lb, ub)
    is_relu = nn.layertypes[len(lb) - 1] in ['ReLU']
    if len(lb) == nn.numlayer:
        return gurobi_optimize_last(m, his, is_relu, label)
    else:
        return gurobi_optimize_bounds(m, his, is_relu)


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


def classify(nn, img):
    lb, ub = get_perturbed_image(img, 0)
    _, _, out_lb, out_ub = elina_bounds(nn, lb, ub)
    predicted_label, predicted_flag = predict_label(out_lb, out_ub)

    if(not predicted_flag):
        print('Image not correctly classified by the network')
        print('Expected label:', int(x0_low[0]))
        print('Classified label:', predicted_label)
    else:
        print('Classified label:', predicted_label)

    return predicted_label, predicted_flag


def gurobi_incremental(nn, input_lb, input_ub, label):
    lb = [input_lb]
    ub = [input_ub]

    for i in range(nn.numlayer):
        s = time.clock()
        out_lb, out_ub, out_lb_relu, out_ub_relu = gurobi_bounds(
            nn, lb, ub, label)
        t = time.clock()
        print(f'Optimize h{i + 1}: {t - s}')

        lb.append(out_lb)
        ub.append(out_ub)

    print('Lower out:', out_lb_relu)
    print('Upper out:', out_ub_relu)

    return verify(out_lb_relu, out_ub_relu, label)


def gurobi_early_stopping(nn, input_lb, input_ub, label):
    lb = [input_lb]
    ub = [input_ub]

    out_lb_first, out_ub_first, _, _ = elina_bounds(
        nn, input_lb, input_ub, finish=1)

    lb.append(out_lb_first[-1])
    ub.append(out_ub_first[-1])

    for i in range(1, nn.numlayer):
        s = time.clock()
        out_lb, out_ub, out_lb_relu, out_ub_relu = gurobi_bounds(
            nn, lb, ub, label)
        t = time.clock()
        print(f'Optimize h{i + 1}: {t - s}')

        lb.append(out_lb)
        ub.append(out_ub)

        if nn.numlayer / 2 < i + 1 < nn.numlayer:
            s = time.clock()
            _, _, out_lb_relu, out_ub_relu = elina_bounds(
                nn, out_lb_relu, out_ub_relu, start=i + 1)
            t = time.clock()
            print(f'ELINA after h{i + 1}: {t - s}')

            if verify(out_lb_relu, out_ub_relu, label):
                return True

    print('Lower out:', out_lb_relu)
    print('Upper out:', out_ub_relu)

    return verify(out_lb_relu, out_ub_relu, label)


def gurobi_earlier_stopping(nn, input_lb, input_ub, label):
    lb = [input_lb]
    ub = [input_ub]

    out_lb_first, out_ub_first, out_lb_relu, out_ub_relu = elina_bounds(
        nn, input_lb, input_ub, finish=1)

    lb.append(out_lb_first[-1])
    ub.append(out_ub_first[-1])

    for i in range(1, nn.numlayer):
        create_s = time.clock()
        m, his = gurobi_create_model(nn, lb, ub)
        create_t = time.clock()
        # print(f'Create model h{i + 1}: {create_t - create_s}')

        is_relu = nn.layertypes[i] in ['ReLU']

        if nn.numlayer / 2 < i + 1 < nn.numlayer:
            pre_s = time.clock()
            out_lb_all, out_ub_all, out_lb_relu, out_ub_relu = elina_bounds(
                nn, out_lb_relu, out_ub_relu, start=i, finish=i + 1)
            pre_t = time.clock()
            # print(f'Pre ELINA h{i + 1}: {pre_t - pre_s}')

            out_lb = out_lb_all[-1]
            out_ub = out_ub_all[-1]
        else:
            out_lb = np.zeros(len(his))
            out_ub = np.zeros(len(his))
            out_lb_relu = np.zeros(len(his))
            out_ub_relu = np.zeros(len(his))

        weight = np.arange(len(his), dtype='float32')
        if i + 1 < nn.numlayer:
            for j in range(len(weight)):
                weight[j] = np.abs(nn.weights[i + 1][:, j]
                                   ).sum() * out_ub_relu[j]

        perm = np.argsort(-weight)

        batch_size = 20
        layer_s = time.clock()
        for j_batch in batch(perm, n=batch_size):
            tbo = []
            for j in j_batch:
                tbo.append(his[j])

            batch_s = time.clock()
            if i + 1 == nn.numlayer:
                perm_lb, perm_ub, perm_lb_relu, perm_ub_relu = gurobi_optimize_last(
                    m, tbo, is_relu, perm[label])
            else:
                perm_lb, perm_ub, perm_lb_relu, perm_ub_relu = gurobi_optimize_bounds(
                    m, tbo, is_relu)
            batch_t = time.clock()
            print(f'Optimize {batch_size} neurons in h{i + 1}: {batch_t - batch_s}')

            for j in range(len(j_batch)):
                out_lb[j_batch[j]] = perm_lb[j]
                out_ub[j_batch[j]] = perm_ub[j]
                out_lb_relu[j_batch[j]] = perm_lb_relu[j]
                out_ub_relu[j_batch[j]] = perm_ub_relu[j]

            if nn.numlayer / 2 < i + 1 < nn.numlayer:
                elina_s = time.clock()
                _, _, early_lb_relu, early_ub_relu = elina_bounds(
                    nn, out_lb_relu, out_ub_relu, start=i + 1)
                elina_t = time.clock()
                # print(f'ELINA at h{i + 1}: {elina_t - elina_s}')

                if verify(early_lb_relu, early_ub_relu, label):
                    print(f'Verified early :)')
                    return True
        layer_t = time.clock()
        print(f'Optimize h{i + 1}: {layer_t - layer_s}')

        lb.append(out_lb)
        ub.append(out_ub)

    # print('Lower out:', out_lb_relu)
    # print('Upper out:', out_ub_relu)

    return verify(out_lb_relu, out_ub_relu, label)


def gurobi_4_1024(nn, input_lb, input_ub, label):
    lb = [input_lb]
    ub = [input_ub]

    out_lb_first, out_ub_first, out_lb_relu, out_ub_relu = elina_bounds(
        nn, input_lb, input_ub, finish=1)

    lb.append(out_lb_first[-1])
    ub.append(out_ub_first[-1])

    for i in range(1, nn.numlayer):
        s = time.clock()
        out_lb_all, out_ub_all, out_lb_relu, out_ub_relu = elina_bounds(
            nn, out_lb_relu, out_ub_relu, start=i, finish=i + 1)
        t = time.clock()
        print(f'Pre ELINA h{i + 1}: {t - s}')

        out_lb = out_lb_all[-1]
        out_ub = out_ub_all[-1]

        if i + 1 not in [3]:
            s = time.clock()
            m, his = gurobi_create_model(nn, lb, ub)
            t = time.clock()
            # print(f'Create model h{i + 1}: {t - s}')

            is_relu = nn.layertypes[i] in ['ReLU']

            weight = np.arange(len(his), dtype='float')
            if i + 1 < nn.numlayer:
                for j in range(len(weight)):
                    weight[j] = np.abs(nn.weights[i + 1][:, j]
                                       ).sum() * out_ub_relu[j]

            perm = np.argsort(-weight)
            if i + 1 == 2:
                perm = perm[:250]

            tbo = []
            for j in range(len(perm)):
                tbo.append(his[perm[j]])

            s = time.clock()
            if i + 1 == nn.numlayer:
                perm_lb, perm_ub, perm_lb_relu, perm_ub_relu = gurobi_optimize_last(
                    m, tbo, is_relu, perm[label])
            else:
                perm_lb, perm_ub, perm_lb_relu, perm_ub_relu = gurobi_optimize_bounds(
                    m, tbo, is_relu)
            t = time.clock()
            # print(f'Optimize h{i + 1}: {t - s}')

            for j in range(len(tbo)):
                out_lb[perm[j]] = perm_lb[j]
                out_ub[perm[j]] = perm_ub[j]
                out_lb_relu[perm[j]] = perm_lb_relu[j]
                out_ub_relu[perm[j]] = perm_ub_relu[j]

        lb.append(out_lb)
        ub.append(out_ub)

    # print('Lower out:', out_lb_relu)
    # print('Upper out:', out_ub_relu)

    return verify(out_lb_relu, out_ub_relu, label)


if __name__ == '__main__':
    if len(argv) < 3:
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

    gurobi_steps = int(argv[4]) if len(argv) > 4 else nn.numlayer

    # Classify the image
    predicted_label, predicted_flag = classify(nn, x0_low)
    if not predict_label:
        sys.exit()

    # Get noisy lower and upper bounds
    lb_noisy, ub_noisy = get_perturbed_image(x0_low, epsilon)

    # Gurobi incremental
    # start = time.time()
    # verified = gurobi_incremental(nn, lb_noisy, ub_noisy, predicted_label)
    # end = time.time()
    # print(f'Gurobi incremental: verified={verified}, time={end-start}')

    # Gurobi early stopping
    # start = time.time()
    # verified = gurobi_early_stopping(nn, lb_noisy, ub_noisy, predicted_label)
    # end = time.time()
    # print(f'Gurobi early stopping: verified={verified}, time={end-start}')

    if nn.numlayer != 4:
        # Gurobi earlier stopping
        start = time.time()
        verified = gurobi_earlier_stopping(
            nn, lb_noisy, ub_noisy, predicted_label)
        end = time.time()
        print(f'Gurobi earlier stopping: verified={verified}, time={end-start}')
    else:
        # Gurobi 4 1024
        start = time.time()
        verified = gurobi_4_1024(nn, lb_noisy, ub_noisy, predicted_label)
        end = time.time()
        print(f'Gurobi 4 1024: verified={verified}, time={end-start}')
