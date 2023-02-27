#!/bin/bash

rm -rf {dense_network,shallow_network,split_network,other_rules,cmnist}/{*.pdf,*.png,*.txt,__pycache__/}
cd other_rules/
rm -rf {anti_hebbian,contrastive_hebbian,hebbian,predictive_coding}/{*.pdf,*.png,*.txt}
cd ..
