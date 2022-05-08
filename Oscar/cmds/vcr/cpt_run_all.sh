GPU=0
# vcr_q_a
bash cmds/vcr/cpt_fsl.sh $GPU vcr_q_a cpt
bash cmds/vcr/pt_fsl.sh $GPU vcr_q_a pt

# vcr_qa_r
bash cmds/vcr/cpt_fsl.sh $GPU vcr_qa_r cpt
bash cmds/vcr/pt_fsl.sh $GPU vcr_qa_r pt

# vcr_qar
bash cmds/vcr/qar_cpt_fsl.sh $GPU vcr_qar cpt
bash cmds/vcr/qar_pt_fsl.sh $GPU vcr_qar pt