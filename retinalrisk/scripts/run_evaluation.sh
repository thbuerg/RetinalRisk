#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

script_path='/home/wildb/dev/projects/ehrgraphs/ehrgraphs/scripts/train_recordgraphs.py'
config_path='/home/wildb/dev/projects/ehrgraphs/config/'

for partition in {0..21}
do
  sbatch -N 1-1 --cpus-per-gpu=32 -p gpu --gres=gpu:1 --mem=64G --time 96:00:0 $script_path --config-path $config_path model=covariates head=mlp 'datamodule/covariates=agesex' setup.name=evaluation220222_only_agesex_$partition datamodule.partition=$partition
  sbatch -N 1-1 --cpus-per-gpu=32 -p gpu --gres=gpu:1 --mem=64G --time 96:00:0 $script_path --config-path $config_path model=identity head=mlp 'datamodule/covariates=agesex' setup.name=evaluation220222_identity_agesex_$partition datamodule.partition=$partition
  sbatch -N 1-1 --cpus-per-gpu=32 -p gpu --gres=gpu:1 --mem=64G --time 96:00:0 $script_path --config-path $config_path model=identity head=mlp 'datamodule/covariates=no_covariates' setup.name=evaluation220222_identity_no_covariates_$partition datamodule.partition=$partition
  sbatch -N 1-1 --cpus-per-gpu=32 -p pgpu --gres=gpu:1 --mem=64G --time 96:00:0 $script_path --config-path $config_path model=gnn head=mlp 'datamodule/covariates=agesex' setup.name=evaluation220222_gnn_agesex_$partition datamodule.partition=$partition
  sbatch -N 1-1 --cpus-per-gpu=32 -p pgpu --gres=gpu:1 --mem=64G --time 96:00:0 $script_path --config-path $config_path model=gnn head=mlp 'datamodule/covariates=no_covariates' setup.name=evaluation220222_gnn_no_covariates_$partition datamodule.partition=$partition
done
