#!/usr/bin/env python
# coding: utf-8


import sys
sys.path.insert(0, '../ELINA/python_interface/')


import numpy as np
import re
import csv

from elina_box import *
from elina_interval import *
from elina_abstract0 import *
from elina_manager import *
from elina_dimension import *
from elina_scalar import *
from elina_interval import *
from elina_linexpr0 import *
from elina_lincons0 import *

from sys import argv
import ctypes
from ctypes.util import find_library
from gurobipy import *
import time
sys.path.insert(0, "../ELINA/python_interface/")


libc = CDLL(find_library("c"))
cstdout = c_void_p.in_dll(libc, "stdout")


class layers:
    def __init__(self):
        self.layertypes = []
        self.weights = []
        self.biases = []
        self.numlayer = 0
        self.ffn_counter = 0


def parse_bias(text):
    if len(text) < 1 or text[0] != "[":
        raise Exception("expected '['")
    if text[-1] != "]":
        raise Exception("expected ']'")
    v = np.array([*map(lambda x: np.double(x.strip()), text[1:-1].split(","))])
    return v


def parse_vector(text):
    if len(text) < 1 or text[0] != "[":
        raise Exception("expected '['")
    if text[-1] != "]":
        raise Exception("expected ']'")
    v = np.array([*map(lambda x: np.double(x.strip()), text[1:-1].split(","))])
    return v.reshape((v.size, 1))


def balanced_split(text):
    i = 0
    bal = 0
    start = 0
    result = []
    while i < len(text):
        if text[i] == "[":
            bal += 1
        elif text[i] == "]":
            bal -= 1
        elif text[i] == "," and bal == 0:
            result.append(text[start:i])
            start = i + 1
        i += 1
    if start < i:
        result.append(text[start:i])
    return result


def parse_matrix(text):
    i = 0
    if len(text) < 1 or text[0] != "[":
        raise Exception("expected '['")
    if text[-1] != "]":
        raise Exception("expected ']'")
    return np.array(
        [*map(lambda x: parse_vector(x.strip()).flatten(), balanced_split(text[1:-1]))]
    )


def parse_net(text):
    lines = [*filter(lambda x: len(x) != 0, text.split("\n"))]
    i = 0
    res = layers()
    while i < len(lines):
        if lines[i] in ["ReLU", "Affine"]:
            W = parse_matrix(lines[i + 1])
            b = parse_bias(lines[i + 2])
            res.layertypes.append(lines[i])
            res.weights.append(W)
            res.biases.append(b)
            res.numlayer += 1
            i += 3
        else:
            raise Exception("parse error: " + lines[i])
    return res


def parse_spec(text):
    text = text.replace("[", "")
    text = text.replace("]", "")
    with open("dummy", "w") as my_file:
        my_file.write(text)
    data = np.genfromtxt("dummy", delimiter=",", dtype=np.double)
    low = np.copy(data[:, 0])
    high = np.copy(data[:, 1])
    return low, high


def get_perturbed_image(x, epsilon):
    image = x[1 : len(x)]
    num_pixels = len(image)
    LB_N0 = image - epsilon
    UB_N0 = image + epsilon

    for i in range(num_pixels):
        if LB_N0[i] < 0:
            LB_N0[i] = 0
        if UB_N0[i] > 1:
            UB_N0[i] = 1
    return LB_N0, UB_N0


def generate_linexpr0(weights, bias, size):
    linexpr0 = elina_linexpr0_alloc(ElinaLinexprDiscr.ELINA_LINEXPR_DENSE, size)
    cst = pointer(linexpr0.contents.cst)
    elina_scalar_set_double(cst.contents.val.scalar, bias)
    for i in range(size):
        elina_linexpr0_set_coeff_scalar_double(linexpr0, i, weights[i])
    return linexpr0


class ElinaToolbox:
    @staticmethod
    def hypercubeToNumpy(man, HC, nb_elts):
        """Return inf and sup bounds"""
        bounds = elina_abstract0_to_box(man, HC)
        return (
            [elt.contents.inf.contents.val.dbl for elt in bounds[:nb_elts]],
            [elt.contents.sup.contents.val.dbl for elt in bounds[:nb_elts]],
        )

    @staticmethod
    def NumpyToHypercube(man, LB, UB):
        num_pixels = len(LB)
        itv = elina_interval_array_alloc(num_pixels)
        for i in range(num_pixels):
            elina_interval_set_double(itv[i], LB[i], UB[i])
        HC = elina_abstract0_of_box(man, 0, num_pixels, itv)
        elina_interval_array_free(itv, num_pixels)
        return HC


class SolverLayer:
    SUPPORTED_LT = ["ReLU", "Affine"]

    def __init__(self, layertypes, weights, biases):
        for layertype in layertypes:
            assert layertype in self.SUPPORTED_LT, "Net type not supported"
        self.layertypes = layertypes
        self.weights = weights
        self.biases = biases


class ElinaLayers(SolverLayer):
    def __init__(self, layertypes, weights, biases, man):
        super().__init__(layertypes, weights, biases)
        self.total_layers = len(weights)
        self.man = man
        self.HC = None

    def __del__(self):
        elina_abstract0_free(self.man, self.HC)

    def compute(self, LB, UB, input_layerno=None, output_layerno=None):
        """
            Return the output values range for i=input_layerno + 1 to i = output_layerno + 1
        """
        self.HC = ElinaToolbox.NumpyToHypercube(self.man, LB, UB)
        lower_bounds = [0.0 for _ in range(self.total_layers)]
        upper_bounds = [0.0 for _ in range(self.total_layers)]
        for layerno in range(self.total_layers):
            layerdim = len(self.biases[layerno])
            self.propagate_hypercube(layerno)
            LB_layerno, UB_layerno = ElinaToolbox.hypercubeToNumpy(
                self.man, self.HC, layerdim
            )
            lower_bounds[layerno], upper_bounds[layerno] = LB_layerno, UB_layerno
            if self.layertypes[layerno] == "ReLU":
                self.compute_relu()

        return lower_bounds, upper_bounds

    def propagate_hypercube(self, layer_no):
        weights, biases = self.weights[layer_no], self.biases[layer_no]
        np.ascontiguousarray(weights, dtype=np.double)
        np.ascontiguousarray(biases, dtype=np.double)

        dims = elina_abstract0_dimension(self.man, self.HC)

        self.num_in_pixels = dims.intdim + dims.realdim
        self.num_out_pixels = len(weights)

        self.add_output_dim()
        self.compute_affine(weights, biases)
        self.remove_input_dim()

    def add_output_dim(self):
        dimadd = elina_dimchange_alloc(0, self.num_out_pixels)
        for i in range(self.num_out_pixels):
            dimadd.contents.dim[i] = self.num_in_pixels
        elina_abstract0_add_dimensions(self.man, True, self.HC, dimadd, False)
        elina_dimchange_free(dimadd)

    def remove_input_dim(self):
        dimrem = elina_dimchange_alloc(0, self.num_in_pixels)
        for i in range(self.num_in_pixels):
            dimrem.contents.dim[i] = i
        elina_abstract0_remove_dimensions(self.man, True, self.HC, dimrem)
        elina_dimchange_free(dimrem)

    def compute_affine(self, weights, biases):
        for i in range(self.num_out_pixels):
            self.HC = elina_abstract0_assign_linexpr_array(
                man=self.man,
                destructive=True,
                org=self.HC,
                tdim=ElinaDim(self.num_in_pixels + i),
                linexpr_array=generate_linexpr0(
                    weights[i], biases[i], self.num_in_pixels
                ),
                size=1,
                dest=None,
            )

    def compute_relu(self):
        self.HC = relu_box_layerwise(
            man=self.man,
            destructive=True,
            elem=self.HC,
            start_offset=0,
            num_dim=self.num_out_pixels,
        )


class GurobiLayers(SolverLayer):
    def __init__(self, layertypes, weights, biases, lower_bounds, upper_bounds):
        super().__init__(layertypes, weights, biases)
        self.m = Model(name="NN")
        self.m.setParam("OutputFlag", False)
        self.dim = [len(elt) for _, elt in enumerate(biases)]
        self.lower_bounds, self.upper_bounds = lower_bounds, upper_bounds

        self.define_variables()
        self.add_relu_constraints()
        self.add_hidden_layer_constraints()

    def define_variables(self):
        self.his = [
            [
                (
                    self.m.addVar(
                        lb=self.lower_bounds[i][j],
                        ub=self.upper_bounds[i][j],
                        vtype=GRB.CONTINUOUS,
                        name=f"Hidden_{i}_{j}",
                    )
                    if i == 0
                    else self.m.addVar(
                        lb=-np.inf, vtype=GRB.CONTINUOUS, name=f"Hidden_{i}_{j}"
                    )
                )
                for j in range(self.dim[i])
            ]
            for i in range(len(self.dim))
        ]

        self.ris = [
            [
                self.m.addVar(lb=-np.inf, vtype=GRB.CONTINUOUS, name=f"ReLU_{i}_{j}")
                for j in range(self.dim[i])
            ]
            for i in range(len(self.dim))
        ]

        self.m.update()

    def add_relu_constraints(self):
        for i in range(len(self.dim)):
            if self.layertypes[i] == "ReLU":
                for j in range(self.dim[i]):
                    inf, sup = self.lower_bounds[i][j], self.upper_bounds[i][j]
                    assert inf <= sup
                    if inf >= 0:
                        self.m.addConstr(self.ris[i][j] == self.his[i][j])
                    elif sup <= 0:
                        self.m.addConstr(self.ris[i][j] == 0)
                    else:
                        k, t = sup / (sup - inf), -sup * inf / (sup - inf)

                        self.m.addConstr(self.ris[i][j] >= 0)
                        self.m.addConstr(self.ris[i][j] >= self.his[i][j])
                        self.m.addConstr(self.ris[i][j] <= k * self.his[i][j] + t)
            else:
                for j in range(self.dim[i]):
                    self.m.addConstr(self.ris[i][j] == self.his[i][j])

        self.m.update()

    def add_hidden_layer_constraints(self):
        for i in range(1, len(self.dim)):
            for j in range(self.dim[i]):
                self.m.addConstr(
                    self.his[i][j]
                    == LinExpr(self.weights[i][j, :], self.ris[i - 1])
                    + self.biases[i][j]
                )

        self.m.update()

    def optimize_one(self, value, minimize=True, write=False):
        self.m.reset()
        self.m.setObjective(value, GRB.MINIMIZE if minimize else GRB.MAXIMIZE)
        if write:
            self.m.write(f"models/model_{value}_min={minimize}.lp")
        self.m.optimize()
        try:
            return self.m.objVal
        except:
            print(
                "Can't find {bound} bound for value {value}".format(
                    bound="lower" if minimize else "upper", j=j
                )
            )
            return None

    def verify_out(self, true_label, write=False):
        mf = self.m.addVar(lb=-np.inf, vtype=GRB.CONTINUOUS, name="Maxi_false")
        mt = self.m.addVar(lb=-np.inf, vtype=GRB.CONTINUOUS, name="Mini_true")

        self.m.addConstr(
            mt
            == (
                max_(self.his[-1][true_label], 0)
                if self.layertypes[-1] == "ReLU"
                else self.his[-1][true_label]
            )
        )

        self.m.addConstr(
            mf
            == (
                max_(self.his[-1][:true_label] + self.his[-1][true_label + 1 :], 0)
                if self.layertypes[-1] == "ReLU"
                else max_(self.his[-1][:true_label] + self.his[-1][true_label + 1 :])
            )
        )

        diff = self.optimize_one(value=mt - mf, minimize=True, write=write)

        try:
            return diff > 0
        except:
            print("Can't optimize output difference")
            return None

    def compute(self, write=False, out=False, true_label=None):
        if out:
            assert true_label is not None, "Need the true label"
            return true_label, self.verify_out(true_label=true_label, write=write)
        else:
            return (
                [
                    self.optimize_one(value=self.his[-1][j], minimize=True, write=write)
                    for j in range(self.dim[-1])
                ],
                [
                    self.optimize_one(
                        value=self.his[-1][j], minimize=False, write=write
                    )
                    for j in range(self.dim[-1])
                ],
            )


class Pipeline:
    def __init__(self, netname, specname, epsilon):
        self.man = elina_box_manager_alloc()
        nn, self.LB_N0, self.UB_N0 = Pipeline.open_files(
            netname, specname, epsilon, self.man
        )
        self.layertypes, self.weights, self.biases = (
            nn.layertypes,
            nn.weights,
            nn.biases,
        )
        self.true_label = Pipeline.get_true_label(netname, specname, self.man)
        self.nb_layers = len(self.biases)

    @staticmethod
    def open_files(netname, specname, epsilon, man):
        with open(netname, "r") as netfile:
            netstring = netfile.read()
        with open(specname, "r") as specfile:
            specstring = specfile.read()

        nn = parse_net(netstring)
        x0_low, x0_high = parse_spec(specstring)
        LB_N0, UB_N0 = get_perturbed_image(x0_low, epsilon)

        return nn, LB_N0, UB_N0

    @staticmethod
    def get_true_label(netname, specname, man):
        nn, LB_N0, UB_N0 = Pipeline.open_files(netname, specname, 0, man)
        return np.argmax(
            ElinaLayers(nn.layertypes, nn.weights, nn.biases, man).compute(
                LB_N0, UB_N0
            )[1][-1]
        )

    def compute(self):
        pass

    def __del__(self):
        elina_manager_free(self.man)


class Strategy1(Pipeline):
    """Processing all at once"""

    def compute(self):
        lb, ub = ElinaLayers(
            self.layertypes, self.weights, self.biases, self.man
        ).compute(self.LB_N0, self.UB_N0)
        gl = GurobiLayers(self.layertypes, self.weights, self.biases, lb, ub)
        label, verified = gl.compute(write=False, out=True, true_label=self.true_label)
        return label, verified


class Strategy2(Pipeline):
    """ Processing `step` layers per `step` layers"""

    step = 3

    def compute(self):
        lb0, ub0 = self.LB_N0, self.UB_N0

        for i in range(0, self.nb_layers, self.step):
            j = min(i + self.step, self.nb_layers)

            lb, ub = ElinaLayers(
                self.layertypes[i:j], self.weights[i:j], self.biases[i:j], self.man
            ).compute(lb0, ub0)
            gl = GurobiLayers(
                self.layertypes[i:j], self.weights[i:j], self.biases[i:j], lb, ub
            )

            if j < self.nb_layers:
                lb0, ub0 = gl.compute(write=False, out=False)
            else:
                return gl.compute(write=False, out=True, true_label=self.true_label)


class Strategy3(Pipeline):
    """Processing neurons 1024 at a time"""

    neuron_steps = 1024

    def indexes(self):
        idx = [0]
        counter = 0
        for i, l in enumerate(map(len, self.biases)):
            if counter >= self.neuron_steps:
                idx.append(i)
                counter = 0
            counter += l
        idx.append(self.nb_layers)
        return idx

    def compute(self):
        lb0, ub0 = self.LB_N0, self.UB_N0
        idx = self.indexes()

        for k in range(len(idx) - 1):
            [i, j] = idx[k : k + 2]
            lb, ub = ElinaLayers(
                self.layertypes[i:j], self.weights[i:j], self.biases[i:j], self.man
            ).compute(lb0, ub0)
            gl = GurobiLayers(
                self.layertypes[i:j], self.weights[i:j], self.biases[i:j], lb, ub
            )

            if j < self.nb_layers:
                lb0, ub0 = gl.compute(write=False, out=False)
            else:
                return gl.compute(write=False, out=True, true_label=self.true_label)


def test_strat(strategies, netname, specname, epsilon):
    for s in strategies:
        analyzer = s(netname, specname, epsilon)

        start = time.time()
        pred_label, res = analyzer.compute()
        end = time.time()

        del analyzer

        print(f"Strategy <{s.__name__}>")
        print("{neg}verified".format(neg="" if res else "can not be "))
        print("analysis time: ", (end - start), " seconds")
        print("\n")


def main():
    if len(argv) not in range(3, 5):
        print("usage: python3.6 " + argv[0] + " net.txt spec.txt [timeout]")
        exit(1)

    netname = argv[1]
    specname = argv[2]
    epsilon = float(argv[3])

    test_strat(
        strategies=[Strategy3],
        netname=netname,
        specname=specname,
        epsilon=epsilon,
    )


if __name__ == "__main__":
    main()
