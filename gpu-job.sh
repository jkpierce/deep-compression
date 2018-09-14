#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1              # request GPU "generic resource"
#SBATCH --cpus-per-task=2    #Maximum of CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32G               # memory per node
#SBATCH --time=0-02:00            # time (DD-HH:MM)
#SBATCH --account=def-marwanh
module load python/3.7 cuda/9.0.176 cudnn
source $HOME/jupyter_py3/bin/activate
$VIRTUAL_ENV/bin/notebook.sh
