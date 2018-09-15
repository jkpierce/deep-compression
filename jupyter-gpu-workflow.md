sshuttle -Nr kpierce@cedar.computecanada.ca -x cedar.computecanada.ca


salloc --time=1:0:0 --ntasks=1 --cpus-per-task=2 --mem-per-cpu=32000M --gres=gpu:1 --account=def-marwanh srun $VIRTUAL_ENV/bin/notebook.sh


ssh -N -f -L localhost:6006:computenode:6006 userid@cluster.computecanada.ca

Replace computenode with the node hostname you retrieved from the preceding step, userid by your Compute Canada username, cluster by the cluster hostname (i.e.: Cedar, Graham, etc.). 

http://cdr544.int.cedar.computecanada.ca:8888/?token=7ed7059fad64446f837567e3
       └────────────────┬───────────────────┘        └──────────┬───────────┘
                  hostname:port                               token

ssh -L 8888:<hostname:port> <username>@<cluster>.computecanada.ca

ssh -L 8888:cdr353.int.cedar.computecanada.ca:8888 kpierce@cedar.computecanada.ca


     http://cdr352.int.cedar.computecanada.ca:8888/?token=66a93c3a60a4caa8aa3573a7630c66c5aba1c7073e429dba


/?token=9100a37b757173dc289ae3652ee79d969560a0a980f1cd33


 lsof -ti:8888 | xargs kill -9


#########################################################################
This is the one that works 
#########################################################################

ssh kpierce@cedar.computecanada.ca
module load python/3.6 cuda/9
source $HOME/jupyter_py3/bin/activate
salloc --time=1:0:0 --ntasks=1 --cpus-per-task=2 --mem-per-cpu=32000M --gres=gpu:1 --account=def-marwanh srun $VIRTUAL_ENV/bin/notebook.sh
sshuttle -Nr kpierce@cedar.computecanada.ca 0/0 --dns --exclude cedar.computecanada.ca



# transfer files like 
scp /home/kevin/Desktop/jupyter-gpu-workflow.md  kpierce@cedar.computecanada.ca:~



























