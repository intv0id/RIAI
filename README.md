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
mv analyser analyser.old
git clone https://github.com/intv0id/RIAI.git analyser
```

Install the packages

``` bash
cd ~/ELINA && make && sudo make install
cd ~/analyser.old && cat ./setup_gurobi.sh >> ~/.bashrc && source ~/.bashrc
```

## Run the analyse

``` bash
cd ~/analyser
./analyser.py [netname] [image] [epsilon] 
```

**example**

``` bash
cd ~/analyser
./analyser.py ../mnist_nets/mnist_relu_6_20.txt ../mnist_images/img10.txt 0.2
```
