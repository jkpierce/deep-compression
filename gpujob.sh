#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1            
#SBATCH --cpus-per-task=1   
#SBATCH --mem=32000M             
#SBATCH --time=5-00:00         
#SBATCH --account=def-marwanh

module load python/3.6  
source $HOME/jupyter_py3/bin/activate
python3 training_script.py
