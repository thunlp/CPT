GPU=0

# zero shot
bash cmds/vg/_fsl.sh $GPU 0 0

# few shot
for N_SHOT in 1 4 16 32
do
        for k in 0 1 2 3 4
        do
	        cd ../prompt_feat
	        bash cmds/vg/vg_train.sh $GPU $N_SHOT $k
	        cd ../Oscar
          bash cmds/vg/_fsl.sh $GPU $N_SHOT $k
        done
done
