# RIAI

Reliable Interpretable Artificial Intelligence Project (FALL 2018 @ ETHZ)

## Authors

* Rubin Deliallisi
* Clément Trassoudaine

## Setup

Download the [VM](https://files.sri.inf.ethz.ch/website/teaching/riai2018/materials/project/riai.ova)

``` bash
cd ~/ELINA && make && sudo make install
cd ~/analyser && bash ./setup_gurobi.sh
```

## Run the analyse

``` bash
cd ~/analyser
./analyser.py --netname [netname] --image [image] [epsilon] 
```
