    tensorboard --logdir models/CQAModel_lemma_word_char_wF_qF_hT_bTconcat_cF/Baseline3_4

# BERT Server

    bert-serving-start -model_dir=/home/ahashi_syuu/Documents/BERT/uncased_L-12_H-768_A-12/uncased_L-12_H-768_A-12/ -num_worker=1 -pooling_strategy=NONE -max_seq_len=110 -port=5555 -port_out=5556 -pooling_layer=[-5]

# BERT Client