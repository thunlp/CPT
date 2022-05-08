USE_COLOR=0


bash cmds/gqa/_pt_fsl_base.sh  0 0 $USE_COLOR
for N_SHOT in 4 16 64 128
do
        for k in 0 1 2 3 4
        do
            bash cmds/gqa/_pt_fsl_base.sh $N_SHOT $k $USE_COLOR
        done
done
