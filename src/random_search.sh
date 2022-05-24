#! /bin/bash


config_dir=./config
data_dir=~/Data/changing_alpha/
eval_dir=~/Data/changing_alpha/evaluation_flows/
results_dir=../results/CLR/random_search/

for i in $(eval echo "{0..$@}")
do
	echo `python train_model_CLR.py -cp $config_dir/config_$i.yml -dp $data_dir -ep $eval_dir -rp $results_dir`
done
       	
