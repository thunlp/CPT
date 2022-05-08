
for task in "vcr_q_a"  "vcr_qa_r" "vcr_qar"
do
	echo $task
#	echo ""
#	echo "pt"
#	python eval/vcr/show_results.py results/vcr/$task/pt
	echo ""
	echo "pt+cpt"
	python eval/vcr/show_ensemble_results.py results/vcr/$task/pt 3
#	echo ""
#	echo "ft"
#	python show_results.py $task/ft
#	echo ""
	echo "------------------------------------"
done

