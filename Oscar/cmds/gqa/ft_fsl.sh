for N_SHOT in 4 
do
        for k in 0 1 2 3 4
        do
            bash cmds/gqa/_ft_fsl_base.sh $N_SHOT $k 
        done
done
