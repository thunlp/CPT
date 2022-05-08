
for task in "test_dev"
do
	echo $task
#	echo ""
#	echo "pt"
#	python eval/gqa/show_results.py results/gqa/$task/pt
	echo ""
	echo "pt+cpt"
	python eval/gqa/show_ensemble_results.py results/gqa/$task/pt 3 1
#	echo ""
#	echo "ft"
#	python show_results.py $task/ft
#	echo ""
	echo "------------------------------------"
done

