GPU_N=$1
TASK_NAME=$2
METHOD_NAME=$3

#for N_SHOT in 1 2 4 8 16 32 64
for N_SHOT in 128
do
        for k in 0 1 2 3 4
        do
	    cd ../prompt_feat
	    bash cmds/vcr/vcr_train.sh $GPU_N $N_SHOT $k 0
	    cd ../Oscar
            bash cmds/vcr/_ft_base.sh $GPU_N $N_SHOT $k $TASK_NAME $METHOD_NAME
            #echo $GPU_N $N_SHOT $k $TASK_NAME $METHOD_NAME
        done
done
