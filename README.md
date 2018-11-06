# RIAI

Reliable Interpretable Artificial Intelligence Project (FALL 2018 @ ETHZ)

## Authors

* Rubin Deliallisi
* Clément Trassoudaine

## Setup

Download the [VM](https://files.sri.inf.ethz.ch/website/teaching/riai2018/materials/project/riai.ova)

``` bash
cd ~/ELINA && make && sudo make install
cd ~/analyser && cat ./setup_gurobi.sh >> ~/.bashrc && source ~/.bashrc
```

## Run the analyse

``` bash
cd ~/analyser
./analyser.py [netname] [image] [epsilon] 
```

**example**

```
cd ~/analyser
./analyser.py ../mnist_nets/mnist_relu_6_20.txt ../mnist_images/img10.txt 0.2
```
