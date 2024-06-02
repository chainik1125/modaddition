#!/bin/bash
#SBATCH --job-name=wdswp
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --partition=eng-research-gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-user=dmanningcoe@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --array=1-1
module load anaconda/2023-Mar/3
module load cuda/11.7
# nvcc --version
# nvidia-smi

sleep $(($SLURM_ARRAY_TASK_ID * 5))

# Activate the Conda environment
source activate torch_env

config=config_average.txt

# Extract the sample name for the current $SLURM_ARRAY_TASK_ID
data_seed_start=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
data_seed_end=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)
sgd_seed_start=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $config)
sgd_seed_end=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $5}' $config)
init_seed_start=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $6}' $config)
init_seed_end=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $7}' $config)

# Extract the sex for the current $SLURM_ARRAY_TASK_ID
wd=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $8}' $config)

grok=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $9}' $config)

train_size=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $10}' $config)

hl_size=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $11}' $config)
lr_input=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $12}' $config)

train_type=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $13}' $config)
P=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $14}' $config)
train_fraction=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $15}' $config)
#Run that file hombre - note the one for the cluster argument so I don't have to keep changing between local and cluster!
srun python3 cluster_run_average.py ${SLURM_ARRAY_TASK_ID} ${data_seed_start} ${data_seed_end} ${sgd_seed_start} ${sgd_seed_end} ${init_seed_start} ${init_seed_end} ${wd} ${grok} ${train_size} ${hl_size} ${lr_input} ${train_type} ${P} ${train_fraction} 1

# Print to a file a message that includes the current $SLURM_ARRAY_TASK_ID, the same name, and the sex of the sample
echo "This is array task ${SLURM_ARRAY_TASK_ID}, ${data_seed_start} ${data_seed_end} ${sgd_seed_start} ${sgd_seed_end} ${init_seed_start} ${init_seed_end} ${wd} ${grok} ${train_size} ${hl_size} ${lr_input} ${train_type} ${P} ${train_fraction}" >> output.txt

#conda install pytorch torchvision -c pytorch
# Run your Python script
# Replace 'your_script.py' with the path to your script


# Deactivate the environment
# conda deactivate
