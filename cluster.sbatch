#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=7-0
#SBATCH --mem=100GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=skp454@nyu.edu
#SBATCH --job-name=slurm_%j 
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

module purge
module load python3/intel/3.6.3
source ~/bigdata/bdpy/bin/activate

python Train.py  > print.txt



