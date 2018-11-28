class Config:
    lr = 1e-3
    dropout = 0.4
    q_max_len = 180
    c_max_len = 180
    char_max_len = 16
    epochs = 30
    batch_size = 20
    char_dim = 15
    l2_weight = 0

    patience = 5
    k_fold = 0
    categories_num = 2
    period = 150

    need_punct = False
    wipe_num = 0

    word_trainable = True
    need_shuffle = True
    use_char_level = False
    load_best_model = True

    model_dir = './models/CQAModel'
    log_dir = './models/CQAModel'
    glove_filename = 'word2vec_dim200_domain_specific.pkl'

    train_list = []
    dev_list = []
    test_list = []
