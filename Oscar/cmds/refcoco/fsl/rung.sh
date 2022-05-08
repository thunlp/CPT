SPLIT=$1
N_SHOT=$2
EPOCHS=$3
GPU=$4

if [ $N_SHOT -gt 16  ]
then
	BATCH=16
else
	BATCH=$N_SHOT
fi
echo $BATCH



for i in 0 1 2 3 4
do
	echo $i
	cd ../prompt_feat/
	bash  cmds/refcoco/cpt/"$SPLIT"_train.sh "$N_SHOT" $i $GPU
	cd ../Oscar/
	bash  cmds/refcoco/fsl/cpt_"$SPLIT".sh $BATCH $EPOCHS $GPU
	mkdir -p results/refcoco/fsl/"$N_SHOT"/"$SPLIT"_test/
	mkdir -p results/refcoco/fsl/"$N_SHOT"/"$SPLIT"_val/
	bash cmds/refcoco/fsl/"$SPLIT"/"$SPLIT"_test.sh $GPU > results/refcoco/fsl/"$N_SHOT"/"$SPLIT"_test/$i
	bash cmds/refcoco/fsl/"$SPLIT"/"$SPLIT"_val.sh $GPU > results/refcoco/fsl/"$N_SHOT"/"$SPLIT"_val/$i
done

