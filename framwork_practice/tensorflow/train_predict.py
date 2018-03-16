
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from dssm import *

if __name__ == '__main__':
    mode = sys.argv[1]

    training_path = 'data/train.txt'
#    training_path = '../9rd_change_train_bucket_data/training_small.dat'
    test_path = 'data/predict_test.txt'

    output_for_auc_path='../7rd_cnxx_data/for_auc.dat'
    query_dict_path = "data/word_dump.dict"#"data/keyword.dict"
    doc_dict_path = "data/word_dump.dict"#"data/keyword.dict"
#    query_dict_path = "../8rd_more_cateid/query_doc.dict"
#    doc_dict_path = "../8rd_more_cateid/query_doc.dict"

    print '----------start load W dic---------------'
    query_word_vecs, query_W, query_word_idx_map = pickle.load(open(query_dict_path))
    doc_word_vecs, doc_W, doc_word_idx_map = pickle.load(open(doc_dict_path))
    print '----------end load W dic---------------'

    query_voc_len = len(query_word_idx_map)
    doc_voc_len = len(doc_word_idx_map)

    param_dic = {
        'n_negative_samples' : 3,
        'query_max_length' : 20,
        'dim_query_embed' : 128,
        'lstm_query_hiddens' :[128],
        'ff_query_hiddens' : [],
        'dim_doc_embed' : 128,
        'lstm_doc_hiddens' : [128],
        'ff_doc_hiddens' : [],
        'doc_max_length' : 20,
        'batch_size' : 128,
        'predict_batch_size' : 10240,
        'n_epochs' : 50,
        'lr' : 0.0005,
        'smoothing_factor' : 10.0,
        'share_embed' : True,
        'query_trainable' : True,
        'doc_trainable' : True,
        'query_pretrain' : True,
        'doc_pretrain' : True,
        'is_restore' : False
    }

    model_edition = "-10"
    tag = '20171213_bmqd_query_emb_%d_lstm_%s_hid_%s_doc_emb_%d_lstm_%s_ff_%s_lr_%g_sf_%g_qt_%s_dt_%s_qp_%s_dp_%s' % \
          (param_dic['dim_query_embed'],
           '-'.join([str(hid) for hid in param_dic['lstm_query_hiddens']]),
           '-'.join([str(hid) for hid in param_dic['ff_query_hiddens']]),
          param_dic['dim_doc_embed'],
          '-'.join([str(hid) for hid in param_dic['lstm_doc_hiddens']]),
          '-'.join([str(hid) for hid in param_dic['ff_doc_hiddens']]),
          param_dic['lr'],
          param_dic['smoothing_factor'], 
          param_dic['query_trainable'], 
          param_dic['doc_trainable'], 
          param_dic['query_pretrain'], 
          param_dic['doc_pretrain']
          )
     

    if not os.path.exists(tag):
        os.mkdir(tag)
    print tag
    model_path = os.path.join(tag, 'model')
    dict_path = os.path.join(tag, 'training_dict.txt')
    log_path = os.path.join(tag, 'training_log.txt')
    if mode == "train":
        tf.reset_default_graph()
        train(training_path = training_path,
        test_path = test_path,
        output_for_auc_path=output_for_auc_path,
        n_negative_samples=param_dic['n_negative_samples'],
        query_max_length=param_dic['query_max_length'],
        dim_query_embed=param_dic['dim_query_embed'],
        lstm_query_hiddens=param_dic['lstm_query_hiddens'],
        ff_query_hiddens=param_dic['ff_query_hiddens'],
        dim_doc_embed=param_dic['dim_doc_embed'],
        lstm_doc_hiddens=param_dic['lstm_doc_hiddens'],
        ff_doc_hiddens=param_dic['ff_doc_hiddens'],
        doc_max_length=param_dic['doc_max_length'],
        batch_size = param_dic['batch_size'],
        n_epochs = param_dic['n_epochs'],
        lr = param_dic['lr'],
        smoothing_factor=param_dic['smoothing_factor'],
        share_embed=param_dic['share_embed'],
        model_path = model_path,
        model_edition = model_edition,
        log_path = log_path,
        query_dict_path = (query_word_vecs, query_W, query_word_idx_map),
        doc_dict_path = (doc_word_vecs, doc_W, doc_word_idx_map),
        query_trainable = param_dic['query_trainable'],
        doc_trainable = param_dic['doc_trainable'],
        query_pretrain = param_dic['query_pretrain'],
        doc_pretrain = param_dic['doc_pretrain'],
        is_restore = param_dic['is_restore'],
        tag = tag)

    elif mode == "predict":
        #test_path = sys.argv[2]
        #model_path = sys.argv[3]
        test_path = 'data/predict_test2.txt'
        #model_path = '20171206_bmqd_query_emb_256_lstm_128_hid__doc_emb_256_lstm_128_ff__lr_0.0005_sf_10_qt_True_dt_True_qp_False_dp_False/'
        #model_path = '20171213_bmqd_query_emb_128_lstm_128_hid__doc_emb_128_lstm_128_ff__lr_0.0005_sf_10_qt_True_dt_True_qp_True_dp_True/'
        model_path = tag+'/'
        model_edition = sys.argv[2]
        #model_path = sys.argv[3]+'/'
        predict(test_path = test_path,
            n_negative_samples=param_dic['n_negative_samples'],
            query_max_length=param_dic['query_max_length'],
            dim_query_embed=param_dic['dim_query_embed'],
            lstm_query_hiddens=param_dic['lstm_query_hiddens'],
            ff_query_hiddens=param_dic['ff_query_hiddens'],
            dim_doc_embed=param_dic['dim_doc_embed'],
            lstm_doc_hiddens=param_dic['lstm_doc_hiddens'],
            ff_doc_hiddens=param_dic['ff_doc_hiddens'],
            doc_max_length=param_dic['doc_max_length'],
            batch_size = param_dic['predict_batch_size'],
            n_epochs = param_dic['n_epochs'],
            lr = param_dic['lr'],
            smoothing_factor=param_dic['smoothing_factor'],
            share_embed=param_dic['share_embed'],
            model_path = model_path,
            log_path = log_path,
            query_dict_path = (query_word_vecs, query_W, query_word_idx_map),
            doc_dict_path = (doc_word_vecs, doc_W, doc_word_idx_map),
            query_trainable = param_dic['query_trainable'],
            doc_trainable = param_dic['doc_trainable'],
            query_pretrain = param_dic['query_pretrain'],
            doc_pretrain = param_dic['doc_pretrain'],
            tag = tag,
            model_edition = model_edition
        )
    elif mode == "predict_query_vector":
        #with tf.device("/cpu:0"):
        query_path = sys.argv[2]
        model_path = sys.argv[3]
        model_edition = sys.argv[4]
        predict_query_vector(test_path = query_path,
            n_negative_samples=param_dic['n_negative_samples'],
            query_max_length=param_dic['query_max_length'],
            dim_query_embed=param_dic['dim_query_embed'],
            lstm_query_hiddens=param_dic['lstm_query_hiddens'],
            ff_query_hiddens=param_dic['ff_query_hiddens'],
            dim_doc_embed=param_dic['dim_doc_embed'],
            lstm_doc_hiddens=param_dic['lstm_doc_hiddens'],
            ff_doc_hiddens=param_dic['ff_doc_hiddens'],
            doc_max_length=param_dic['doc_max_length'],
            batch_size = param_dic['predict_batch_size'],
            n_epochs = param_dic['n_epochs'],
            lr = param_dic['lr'],
            smoothing_factor=param_dic['smoothing_factor'],
            share_embed=param_dic['share_embed'],
            model_path = model_path,
            log_path = log_path,
            query_dict_path = (query_word_vecs, query_W, query_word_idx_map),
            doc_dict_path = (doc_word_vecs, doc_W, doc_word_idx_map),
            query_trainable = param_dic['query_trainable'],
            doc_trainable = param_dic['doc_trainable'],
            query_pretrain = param_dic['query_pretrain'],
            doc_pretrain = param_dic['doc_pretrain'],
            tag = tag,
            model_edition = model_edition
        )
    elif mode == "predict_doc_vector":
        #doc_path = './for_valid_data/nick_preference.dat'
        doc_path = sys.argv[2]
        model_path = sys.argv[3]
        model_edition = sys.argv[4]
        predict_doc_vector(test_path = doc_path,
            n_negative_samples=param_dic['n_negative_samples'],
            query_max_length=param_dic['query_max_length'],
            dim_query_embed=param_dic['dim_query_embed'],
            lstm_query_hiddens=param_dic['lstm_query_hiddens'],
            ff_query_hiddens=param_dic['ff_query_hiddens'],
            dim_doc_embed=param_dic['dim_doc_embed'],
            lstm_doc_hiddens=param_dic['lstm_doc_hiddens'],
            ff_doc_hiddens=param_dic['ff_doc_hiddens'],
            doc_max_length=param_dic['doc_max_length'],
            batch_size = param_dic['predict_batch_size'],
            n_epochs = param_dic['n_epochs'],
            lr = param_dic['lr'],
            smoothing_factor=param_dic['smoothing_factor'],
            share_embed=param_dic['share_embed'],
            model_path = model_path,
            log_path = log_path,
            query_dict_path = (query_word_vecs, query_W, query_word_idx_map),
            doc_dict_path = (doc_word_vecs, doc_W, doc_word_idx_map),
            query_trainable = param_dic['query_trainable'],
            doc_trainable = param_dic['doc_trainable'],
            query_pretrain = param_dic['query_pretrain'],
            doc_pretrain = param_dic['doc_pretrain'],
            tag = tag,
            model_edition = model_edition
        )
