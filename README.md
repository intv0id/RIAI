# RIAI

Reliable Interpretable Artificial Intelligence Project (FALL 2018 @ ETHZ)

## Authors

* Rubin Deliallisi
* Clément Trassoudaine

## Setup

Download the [VM](https://files.sri.inf.ethz.ch/website/teaching/riai2018/materials/project/riai.ova)

Clone the repo

```
cd ~
mv analyzer analyzer.old
git clone https://github.com/intv0id/RIAI.git analyzer
```

Install the packages

``` bash
cd ~/ELINA && make && sudo make install
cd ~/analyzer.old && cat ./setup_gurobi.sh >> ~/.bashrc && source ~/.bashrc
```

## Run the analyse

``` bash
cd ~/analyzer
# Original analyzer
./analyzer.py [netname] [image] [epsilon]
# Improved WIP analyzer
./analyzer_tests.py [netname] [image] [epsilon]


# Gurobi steps must be > 0.
./analyzer_new.py [netname] [image] [epsilon] [gurobisteps]
```

**example**

``` bash
cd ~/analyzer
./analyzer.py ../mnist_nets/mnist_relu_6_20.txt ../mnist_images/img10.txt 0.1
./analyzer_tests.py ../mnist_nets/mnist_relu_6_20.txt ../mnist_images/img10.txt 0.1



# In this case the transition h1 - h2 - h3 is approximated using the linear solver.
./analyzer_new.py ../mnist_nets/mnist_relu_6_20.txt ../mnist_images/img10.txt 0.1 3
```
