GPU_N=$1
TASK_NAME=$2
METHOD_NAME=$3

bash cmds/vcr/_pt_fsl_base.sh $GPU_N 0 0 $TASK_NAME $METHOD_NAME

for N_SHOT in 4 16 64 128
do
        for k in 0 1 2 3 4
        do
	        cd ../prompt_feat
	        bash cmds/vcr/vcr_train.sh $GPU_N $N_SHOT $k 0
	        cd ../Oscar
            bash cmds/vcr/_pt_fsl_base.sh $GPU_N $N_SHOT $k $TASK_NAME $METHOD_NAME
            #echo $GPU_N $N_SHOT $k $TASK_NAME $METHOD_NAME
        done
done
