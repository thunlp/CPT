GPU=0
#cd ../prompt_feat/
#bash cmds/refcoco/prepare.sh
#cd ../Oscar

mkdir -p results/refcoco/fsl/0/refcoco/
mkdir -p results/refcoco/fsl/0/refcoco+/
mkdir -p results/refcoco/fsl/0/refcocog/
bash cmds/refcoco/zsl/refcoco.sh > results/refcoco/fsl/0/refcoco/0
bash cmds/refcoco/zsl/refcoco+.sh > results/refcoco/fsl/0/refcoco+/0
bash cmds/refcoco/zsl/refcocog.sh > results/refcoco/fsl/0/refcocog/0


for N_SHOT in 1 2 4 8 16
do
        bash cmds/refcoco/fsl/run.sh refcoco $N_SHOT 500 $GPU
        bash cmds/refcoco/fsl/run.sh refcoco+ $N_SHOT 500 $GPU
        bash cmds/refcoco/fsl/rung.sh refcocog $N_SHOT 500 $GPU
done


