#!/bin/bash
#SBATCH --gres=gpu:1               
#SBATCH --mem=32000M             
#SBATCH --time=00:30:00         
#SBATCH --account=def-marwanh
#SBATCH --mail-user=kevinkayaks@gmail.com
#SBATCH --mail-type=ALL

module load python/3.6  
source $HOME/jupyter_py3/bin/activate
python3 learning2.py
