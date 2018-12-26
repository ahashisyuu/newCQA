class Config:
    lr = 1e-3
    dropout = 0.2

    q_max_len = 110
    c_max_len = 150
    char_max_len = 20
    epochs = 5
    max_steps = 9000
    batch_size = 20
    char_dim = 15
    l2_weight = 0
    margin = 0.1

    patience = 5
    k_fold = 0
    categories_num = 2
    period = 150

    need_punct = False
    wipe_num = 0

    word_trainable = False
    word_train_type = '_wT' if word_trainable else '_wF'
    char_trainable = False
    char_train_type = '_cT' if char_trainable else '_cF'
    cate_trainable = False
    cate_train_type = '_qT' if cate_trainable else '_qF'
    need_shuffle = True
    use_char_level = True
    char_type = '_char' if use_char_level else '_wo_char'
    load_best_model = True

    glove_filename = 'word2vec_dim200_domain_specific.pkl'
    word_type = 'lemma'
    suffix = word_type + char_type + word_train_type + cate_train_type
    suffix = suffix + char_train_type if use_char_level else suffix
    model_dir = './models/CQAModel_' + suffix
    log_dir = './models/CQAModel_' + suffix

    train_list = []
    dev_list = []
    test_list = []
