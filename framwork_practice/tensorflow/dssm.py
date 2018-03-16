import math
import pickle
import traceback
import random
import tensorflow as tf
import time

from utils import *
import sys
import subprocess
import cal_auc
reload(sys)  
sys.setdefaultencoding('utf8')


# Assume that you have 12GB of GPU memory and want to allocate ~4GB:
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

from tensorflow.contrib import rnn
from tensorflow import nn
#tf.nn.bidirectional_dynamic_rnn()

class DSSM_Model():
    def init_weight(self, dim_in, dim_out, name=None, stddev=1.0, trainable=True):
        return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev/math.sqrt(float(dim_in))), name=name, trainable=trainable)

    def init_bias(self, dim_out, name=None):
        return tf.Variable(tf.zeros([dim_out]), name=name)


    def init_query_w_pretrain(self, dim_in, dim_out):
        with tf.device("/cpu:0"):
            query_embedding_placeholder = tf.placeholder(tf.float32, [dim_in, dim_out])
            embedding_init = self.W_query_emb.assign(query_embedding_placeholder)
        return embedding_init, query_embedding_placeholder
 
    def init_doc_w_pretrain(self, dim_in, dim_out):
        with tf.device("/cpu:0"):
            doc_embedding_placeholder = tf.placeholder(tf.float32, [dim_in, dim_out])
            embedding_init = self.W_doc_emb.assign(doc_embedding_placeholder)
        return embedding_init, doc_embedding_placeholder

    def __init__(self, batch_size, n_negative_samples, query_max_length, lstm_query_hiddens, ff_query_hiddens, dim_query_embed,
                 dim_doc_embed, lstm_doc_hiddens, ff_doc_hiddens, doc_max_length, query_n_words, doc_n_words,
                 share_embed=False, smoothing_factor=1.0, query_trainable=True, doc_trainable=True):
        self.query_max_length = query_max_length
        self.batch_size = batch_size
        self.n_negative_samples = n_negative_samples
        self.query_n_words = query_n_words
        self.doc_n_words = doc_n_words
        #self.lstm_query_hiddens = lstm_query_hiddens
        #self.ff_query_hiddens = ff_query_hiddens
        self.dim_query_embed = dim_query_embed
        if share_embed:
            self.dim_doc_embed = dim_query_embed
        else:
            self.dim_doc_embed = dim_doc_embed

        self.doc_max_length = doc_max_length
        self.smoothing_factor = smoothing_factor

        # initializing the embedding
        print "start init W_query_emb"
        with tf.device("/cpu:0"):
            self.W_query_emb = self.init_weight(query_n_words, dim_query_embed, name='W_query_emb', trainable=query_trainable)
            if share_embed:
                self.W_doc_emb = self.W_query_emb
            else:
                self.W_doc_emb = self.init_weight(doc_n_words, dim_doc_embed, name='W_doc_emb', trainable=doc_trainable)
        print "end init W_query_emb W_doc_emb"

        # initializing the lstm query weights
        if len(lstm_query_hiddens) == 1:
            self.lstm_query_fw = rnn.BasicLSTMCell(lstm_query_hiddens[0], state_is_tuple=True)
            self.lstm_query_bw = rnn.BasicLSTMCell(lstm_query_hiddens[0], state_is_tuple=True)
        elif len(lstm_query_hiddens) > 1:
            cells_fw = []
            cells_bw = []
            for lstm_query_hidden in lstm_query_hiddens:
                cells_fw.append(rnn.BasicLSTMCell(lstm_query_hidden, state_is_tuple=True))
                cells_bw.append(rnn.BasicLSTMCell(lstm_query_hidden, state_is_tuple=True))
            self.lstm_query_fw = rnn.MultiRNNCell(cells_fw, state_is_tuple=True)
            self.lstm_query_bw = rnn.MultiRNNCell(cells_bw, state_is_tuple=True)
        else:
            self.lstm_query_fw = None
            self.lstm_query_bw = None

        # initializing the query weights
        self.Ws_ff_query = []
        self.bs_ff_query = []
        if len(lstm_query_hiddens) > 0:
            last_ff_query_hidden = 2 * lstm_query_hiddens[-1]
        else:
            last_ff_query_hidden = self.dim_query_embed
        for layer, ff_query_hidden in enumerate(ff_query_hiddens):
            self.Ws_ff_query.append(self.init_weight(last_ff_query_hidden, ff_query_hidden, name='W_ff_query_%d' % (layer,)))
            last_ff_query_hidden = ff_query_hidden
            self.bs_ff_query.append(self.init_bias(ff_query_hidden, name='b_ff_query_%d' % (layer,)))

        # initializing the lstm doc weights
        if len(lstm_doc_hiddens) == 1:
            self.lstm_doc_fw = rnn.BasicLSTMCell(lstm_doc_hiddens[0], state_is_tuple=True)
            self.lstm_doc_bw = rnn.BasicLSTMCell(lstm_doc_hiddens[0], state_is_tuple=True)
        elif len(lstm_doc_hiddens) > 1:
            cells_fw = []
            cells_bw = []
            for lstm_doc_hidden in lstm_doc_hiddens:
                cells_fw.append(rnn.BasicLSTMCell(lstm_doc_hidden, state_is_tuple=True))
                cells_bw.append(rnn.BasicLSTMCell(lstm_doc_hidden, state_is_tuple=True))
            self.lstm_doc_fw = rnn.MultiRNNCell(cells_fw, state_is_tuple=True)
            self.lstm_doc_bw = rnn.MultiRNNCell(cells_bw, state_is_tuple=True)
        else:
            self.lstm_doc_fw = None
            self.lstm_doc_bw = None

        # initializing the ff doc weights
        self.Ws_ff_doc = []
        self.bs_ff_doc = []
        if len(lstm_doc_hiddens) > 0:
            last_ff_doc_hidden = 2 * lstm_doc_hiddens[-1]
        else:
            last_ff_doc_hidden = self.dim_doc_embed
        for layer, ff_doc_hidden in enumerate(ff_doc_hiddens):
            self.Ws_ff_doc.append(self.init_weight(last_ff_doc_hidden, ff_doc_hidden, name='W_ff_doc_%d' % (layer,)))
            last_ff_doc_hidden = ff_doc_hidden
            self.bs_ff_doc.append(self.init_bias(ff_doc_hidden, name='b_ff_doc_%d' % (layer,)))

    def build_model(self):
        query = tf.placeholder(tf.int32, [self.batch_size, self.query_max_length])
        query_mask = tf.placeholder(tf.int32, [self.batch_size, self.query_max_length])
        docs = tf.placeholder(tf.int32, [1 + self.n_negative_samples, self.batch_size, self.doc_max_length])
        docs_mask = tf.placeholder(tf.int32, [1 + self.n_negative_samples, self.batch_size, self.doc_max_length])
        query_length = tf.reduce_sum(query_mask, 1)
        docs_length = tf.reduce_sum(docs_mask, 2)

        if self.lstm_query_fw is not None:
            # get the query embedding
            query_emb = []
            with tf.device("/cpu:0"):
                print "start lookup w query"
                for i in range(self.query_max_length):
                    query_emb.append(tf.nn.embedding_lookup(self.W_query_emb, query[:, i]))
                print "end lookup w query"
            # RNN
            with tf.variable_scope("BiRNN_query"):
                # rnn_outputs: the list of LSTM outputs, as a list.
                # #   What we want is the latest output, rnn_outputs[-1]
                outputs, _, _ = rnn.static_bidirectional_rnn(self.lstm_query_fw, self.lstm_query_bw, query_emb, dtype=tf.float32)
#outputs, _, _ = nn.bidirectional_dynamic_rnn(self.lstm_query_fw, self.lstm_query_bw, query_emb, dtype=tf.float32)
                query_hidden = outputs[-1]
        else:
            with tf.device("/cpu:0"):
                query_emb = tf.zeros([self.batch_size, self.dim_query_embed])
                for i in range(self.query_max_length):
                    query_emb += tf.nn.embedding_lookup(self.W_query_emb, query[:, i]) * tf.expand_dims(tf.to_float(query_mask[:, i]), 1)
                query_hidden = query_emb

        for W_query, b_query in zip(self.Ws_ff_query, self.bs_ff_query):
#            query_hidden = tf.nn.tanh(tf.matmul(query_hidden, W_query) + b_query)
            query_hidden = tf.nn.relu(tf.matmul(query_hidden, W_query) + b_query)

        query_hidden_norm = tf.sqrt(tf.reduce_sum(tf.square(query_hidden), 1, keep_dims=True))
        epsilon = tf.constant(value=0.01, shape=query_hidden_norm.get_shape())
        query_hidden_norm = query_hidden_norm + epsilon
        normalized_query_hidden = query_hidden / query_hidden_norm

        similarity_scores = []
        for n in range(1 + self.n_negative_samples):
            if self.lstm_doc_fw is not None:
                # get the doc embedding
                doc_emb = []
                print "start lookup w doc"
                with tf.device("/cpu:0"):
                    for i in range(self.doc_max_length):
                        doc_emb.append(tf.nn.embedding_lookup(self.W_doc_emb, docs[n, :, i]))
                print "end lookup w doc"
                # RNN
                with tf.variable_scope("BiRNN_doc"):
                    if n > 0: tf.get_variable_scope().reuse_variables()
                    outputs, _, _ = rnn.static_bidirectional_rnn(self.lstm_doc_fw, self.lstm_doc_bw, doc_emb, dtype=tf.float32)
#outputs, _, _ = nn.bidirectional_dynamic_rnn(self.lstm_doc_fw, self.lstm_doc_bw, doc_emb, dtype=tf.float32)
                    doc_hidden = outputs[-1]
            else:
                with tf.device("/cpu:0"):
                    doc_emb = tf.zeros([self.batch_size, self.dim_doc_embed])
                    for i in range(self.doc_max_length):
                        doc_emb += tf.nn.embedding_lookup(self.W_doc_emb, docs[n, :, i]) * tf.expand_dims(tf.to_float(docs_mask[n, :, i]), 1)
                    doc_hidden = doc_emb
            # forward
            for W_ff_doc, b_ff_doc in zip(self.Ws_ff_doc, self.bs_ff_doc):
#                doc_hidden = tf.nn.tanh(tf.matmul(doc_hidden, W_ff_doc) + b_ff_doc)
                doc_hidden = tf.nn.relu(tf.matmul(doc_hidden, W_ff_doc) + b_ff_doc)
            # similarity
            doc_hidden_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_hidden), 1, keep_dims=True))
            epsilon = tf.constant(value=0.01, shape=doc_hidden_norm.get_shape())
            doc_hidden_norm = doc_hidden_norm + epsilon
            normalized_doc_hidden = doc_hidden / doc_hidden_norm
            similarity_scores.append(tf.reduce_sum(tf.multiply(normalized_query_hidden, normalized_doc_hidden), 1))
        similarity_scores = tf.transpose(tf.stack(similarity_scores)) # batch_size * (1 + self.n_negative_samples)
        #print similarity_scores
        loss = tf.negative(tf.reduce_mean(tf.nn.log_softmax(tf.multiply(similarity_scores, self.smoothing_factor))[:, 0]))
        
        l2_w_query_loss = 0.0
        l2_w_doc_loss = 0.0
        l2_w_lstm_loss = 0.0
        l2_w_ff_loss = 0.0
        
        query_w_std = self.query_n_words * 10.0
        doc_w_std = self.doc_n_words * 10.0
        lstm_w_std = 1000.0
        ff_w_std = 1000.0

        for tf_var in tf.trainable_variables():
            if tf_var.name.find("W_query_emb") >= 0:
                l2_w_query_loss += tf.nn.l2_loss(tf_var)
            elif tf_var.name.find("W_doc_emb") >= 0:
                l2_w_doc_loss += tf.nn.l2_loss(tf_var)
            elif tf_var.name.find("lstm") >= 0:
                l2_w_lstm_loss += tf.nn.l2_loss(tf_var)
            elif tf_var.name.find("W_ff") >= 0:
                l2_w_ff_loss += tf.nn.l2_loss(tf_var)

        l2_w_query_loss = l2_w_query_loss / query_w_std
        l2_w_doc_loss = l2_w_doc_loss / doc_w_std
        l2_w_lstm_loss = l2_w_lstm_loss / lstm_w_std
        l2_w_ff_loss = l2_w_ff_loss / ff_w_std

        return loss, similarity_scores, query, query_mask, docs, docs_mask
        #return loss, similarity_scores, query, query_mask, docs, docs_mask, l2_w_query_loss, l2_w_doc_loss, l2_w_lstm_loss, l2_w_ff_loss

    def get_normalized_query_vector(self):
        query = tf.placeholder(tf.int32, [self.batch_size, self.query_max_length])
        query_mask = tf.placeholder(tf.int32, [self.batch_size, self.query_max_length])
        query_length = tf.reduce_sum(query_mask, 1)
        if self.lstm_query_fw is not None:
            # get the query embedding
            query_emb = []
#with tf.device("/cpu:0"):
            for i in range(self.query_max_length):
                query_emb.append(tf.nn.embedding_lookup(self.W_query_emb, query[:, i]))
            # RNN
            with tf.variable_scope("BiRNN_query"):
                outputs, _, _ = rnn.static_bidirectional_rnn(self.lstm_query_fw, self.lstm_query_bw, query_emb, dtype=tf.float32)
#outputs, _, _ = nn.bidirectional_dynamic_rnn(self.lstm_query_fw, self.lstm_query_bw, doc_emb, dtype=tf.float32)
                query_hidden = outputs[-1]

                # Forward direction
#                with tf.variable_scope("FW"):
#                    output_fw, _ = rnn(self.lstm_query_fw, query_emb, dtype=tf.float32, sequence_length=query_length)
#                    output_fw = _reverse_seq(output_fw, query_length)[0]
#                # Backward direction
#                with tf.variable_scope("BW"):
#                    reversed_query_emb = _reverse_seq(query_emb, query_length)
#                    output_bw, _ = rnn(self.lstm_query_bw, reversed_query_emb, dtype=tf.float32, sequence_length=query_length)
#                    output_bw = _reverse_seq(output_bw, query_length)[0]
#                query_hidden = tf.concat(axis=1, values=[output_fw, output_bw])

        else:
#with tf.device("/cpu:0"):
            query_emb = tf.zeros([self.batch_size, self.dim_query_embed])
            for i in range(self.query_max_length):
                query_emb += tf.nn.embedding_lookup(self.W_query_emb, query[:, i]) * tf.expand_dims(tf.to_float(query_mask[:, i]), 1)
            query_hidden = query_emb

        for W_query, b_query in zip(self.Ws_ff_query, self.bs_ff_query):
            query_hidden = tf.nn.relu(tf.matmul(query_hidden, W_query) + b_query)
#            query_hidden = tf.nn.tanh(tf.matmul(query_hidden, W_query) + b_query)
        query_hidden_norm = tf.sqrt(tf.reduce_sum(tf.square(query_hidden), 1, keep_dims=True))
        normalized_query_hidden = query_hidden / query_hidden_norm
        return normalized_query_hidden, query, query_mask

    def get_normalized_doc_vector(self):
        doc = tf.placeholder(tf.int32, [self.batch_size, self.doc_max_length])
        doc_mask = tf.placeholder(tf.int32, [self.batch_size, self.doc_max_length])
        doc_length = tf.reduce_sum(doc_mask, 1)
        if self.lstm_doc_fw is not None:
            # get the doc embedding
            doc_emb = []
#with tf.device("/cpu:0"):
            for i in range(self.doc_max_length):
                doc_emb.append(tf.nn.embedding_lookup(self.W_doc_emb, doc[:, i]))
            # RNN
            with tf.variable_scope("BiRNN_doc"):
                outputs, _, _ = rnn.static_bidirectional_rnn(self.lstm_doc_fw, self.lstm_doc_bw, doc_emb, dtype=tf.float32)
#outputs, _, _ = nn.bidirectional_dynamic_rnn(self.lstm_query_fw, self.lstm_query_bw, doc_emb, dtype=tf.float32)
                doc_hidden = outputs[-1]

#                # Forward direction
#                with tf.variable_scope("FW"):
#                    output_fw, _ = rnn(self.lstm_doc_fw, doc_emb, dtype=tf.float32, sequence_length=doc_length)
#                    output_fw = _reverse_seq(output_fw, doc_length)[0]
#                # Backward direction
#                with tf.variable_scope("BW"):
#                    reversed_docs_emb = _reverse_seq(doc_emb, doc_length)
#                    output_bw, _ = rnn(self.lstm_doc_bw, reversed_docs_emb, dtype=tf.float32, sequence_length=doc_length)
#                    output_bw = _reverse_seq(output_bw, doc_length)[0]
#                doc_hidden = tf.concat(axis=1, values=[output_fw, output_bw])

        else:
#with tf.device("/cpu:0"):
            doc_hidden = tf.zeros([self.batch_size, self.dim_doc_embed])
            for i in range(self.doc_max_length):
                doc_hidden += tf.nn.embedding_lookup(self.W_doc_emb, doc[:, i]) * tf.expand_dims(tf.to_float(doc_mask[:, i]), 1)
        # forward
        for W_ff_doc, b_ff_doc in zip(self.Ws_ff_doc, self.bs_ff_doc):
            doc_hidden = tf.nn.relu(tf.matmul(doc_hidden, W_ff_doc) + b_ff_doc)
#            doc_hidden = tf.nn.tanh(tf.matmul(doc_hidden, W_ff_doc) + b_ff_doc)
        doc_hidden_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_hidden), 1, keep_dims=True))
        normalized_doc_hidden = doc_hidden / doc_hidden_norm
        return normalized_doc_hidden, doc, doc_mask

    def get_similarity(self):
        normalized_query_vector, query, query_mask = self.get_normalized_query_vector()
        normalized_doc_vector, doc, doc_mask = self.get_normalized_doc_vector()
        similarity_score = tf.reduce_sum(tf.multiply(normalized_query_vector, normalized_doc_vector), 1)
        return similarity_score, query, query_mask, doc, doc_mask

def restore_vars(saver, sess, chkpt_dir):
    """ Restore saved net, global score and step, and epsilons OR
    create checkpoint directory for later storage. """
    sess.run(tf.global_variables_initializer())

    checkpoint_dir = chkpt_dir 

    if not os.path.exists(checkpoint_dir):
        try:
            print("making checkpoint_dir")
            os.makedirs(checkpoint_dir)
            return False
        except OSError:
            raise

    path = tf.train.get_checkpoint_state(checkpoint_dir)
    print("path = ",path)
    if path is None:
        return False
    else:
        saver.restore(sess, path.model_checkpoint_path)
        return True


def train(training_path='./data/training.txt',
          test_path='./data/val.txt',
          output_for_auc_path='./data/for_auc.dat',
          n_negative_samples=4,
          query_max_length=20,
          dim_query_embed=1024,
          lstm_query_hiddens=[512],
          ff_query_hiddens=[512],
          dim_doc_embed=1024,
          lstm_doc_hiddens=[1024],
          ff_doc_hiddens=[512],
          doc_max_length=20,
          batch_size = 128,
          n_epochs = 20,
          lr = 0.001,
          smoothing_factor=10.0,
          model_path = './models/',
          model_edition = -9,
          query_dict_path = (None, None, None),
          doc_dict_path = (None, None, None),
          log_path = './data/log.txt',
          share_embed=False,
          query_trainable = True,
          doc_trainable = False,
          query_pretrain = False,
          doc_pretrain = True,
          is_restore = False,
          tag = ""
          ):

    query_data = \
            trainfile_to_integers_with_dictionary(training_path, query_dict_path[2], doc_dict_path[2])
#for item in query_data:
#        print item


    random.shuffle(query_data)

    #test_query_data = words_to_integers_with_dictionary_return_token(test_path, query_dict_path[2], doc_dict_path[2])
    query_n_words = len(query_dict_path[2])
    doc_n_words = len(doc_dict_path[2])

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    #sess = tf.Session()
    model = DSSM_Model(
        batch_size=batch_size,
        n_negative_samples=n_negative_samples,
        query_max_length=query_max_length,
        lstm_query_hiddens=lstm_query_hiddens,
        ff_query_hiddens=ff_query_hiddens,
        dim_query_embed=dim_query_embed,
        dim_doc_embed=dim_doc_embed,
        lstm_doc_hiddens=lstm_doc_hiddens,
        ff_doc_hiddens=ff_doc_hiddens,
        doc_max_length=doc_max_length,
        query_n_words=query_n_words,
        doc_n_words=doc_n_words,
        share_embed=share_embed,
        smoothing_factor=smoothing_factor,
        query_trainable=query_trainable,
        doc_trainable=doc_trainable
        )

    loss, similarity_scores, query, query_mask, docs, docs_mask = model.build_model()
    #loss, similarity_scores, query, query_mask, docs, docs_mask, l2_w_query_loss, l2_w_doc_loss, l2_w_lstm_loss, l2_w_ff_loss  = model.build_model()

#print loss, similarity_scores, query, query_mask, docs, docs_mask, l2_w_query_loss, l2_w_doc_loss, l2_w_lstm_loss, l2_w_ff_loss

    saver = tf.train.Saver(max_to_keep=200)
    learning_rate = lr
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    print "tf-----var----name"
    for tf_var in tf.trainable_variables():
        print tf_var.name

    if is_restore == True:
        restored = restore_vars(saver, sess, tag)
        print "restore the model from %s" %(tag)

    else:
        if query_pretrain == True:
            _init_query_w, query_embedding_placeholder = model.init_query_w_pretrain(query_n_words, dim_query_embed)
        else:
            _init_query_w = None

        if doc_pretrain == True:
            _init_doc_w, doc_embedding_placeholder = model.init_doc_w_pretrain(doc_n_words, dim_doc_embed)
        else:
            _init_doc_w = None

        init = tf.global_variables_initializer()
        print "start init all var"
        sess.run(init)

        if query_pretrain == True:
            print "start init query W"
            sess.run(_init_query_w, feed_dict={query_embedding_placeholder:query_dict_path[1]})
            print "end init query W"
        if doc_pretrain == True:
            print "start init doc W"
            sess.run(_init_doc_w, feed_dict={doc_embedding_placeholder : doc_dict_path[1]})
            print "end init doc W"
            #print _init_doc_w


    save_model_iter_num = min(8000, len(query_data)/batch_size)
    tmp_k = 0
    for epoch in range(n_epochs):
        #train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        num_batches = 0
        training_loss = 0.0
        #for i in range(save_model_iter_num):
            #current_queries = random.sample(query_data, batch_size)
        for start, end in zip(range(0, len(query_data), batch_size), range(batch_size, len(query_data), batch_size)):
            tmp_k += 1         
            current_queries = query_data[start:end]
            #for i, current_query in enumerate(current_queries):
            #    print current_query
            current_queries_matrix = np.zeros((batch_size, query_max_length), dtype=np.uint32)
            current_queries_mask_matrix = np.zeros((batch_size, query_max_length), dtype=np.uint32)
            current_docs_matrix = np.zeros((1 + n_negative_samples, batch_size, doc_max_length), dtype=np.uint32)
            current_docs_mask_matrix = np.zeros((1 + n_negative_samples, batch_size, doc_max_length), dtype=np.uint32)

            for i, current_query in enumerate(current_queries):
                query_words, positive_doc_words, negative_docs_words = current_query
                if len(query_words) == 0 or len(positive_doc_words) == 0:
                    continue
                # query
                query_words = query_words[:query_max_length]
                current_queries_matrix[i] = query_words + [0] * (query_max_length - len(query_words))
                current_queries_mask_matrix[i, :len(query_words)] = 1

                # docs
                for j, curr_doc in enumerate([positive_doc_words] + negative_docs_words[:n_negative_samples]):
                    curr_doc = curr_doc[:doc_max_length]
                    current_docs_mask_matrix[j, i, :len(curr_doc)] = 1
                    curr_doc += [0] * (doc_max_length - len(curr_doc))
                    current_docs_matrix[j, i] = curr_doc

            #_, loss_value, query_emb, doc_emb = sess.run([train_op, loss, query_emb, doc_emb], feed_dict={
#            tmp_l2_w_query_loss, tmp_l2_w_doc_loss, tmp_l2_w_lstm_loss, tmp_l2_w_ff_loss, loss_value = sess.run([l2_w_query_loss, l2_w_doc_loss, l2_w_lstm_loss, l2_w_ff_loss, loss], feed_dict={
#print current_queries_matrix, current_queries_mask_matrix, current_docs_matrix, current_docs_mask_matrix 
#            print loss, similarity_scores, query, query_mask, docs, docs_mask, l2_w_query_loss, l2_w_doc_loss, l2_w_lstm_loss, l2_w_ff_loss
            #tmp_l2_w_query_loss, tmp_l2_w_doc_loss, tmp_l2_w_lstm_loss, tmp_l2_w_ff_loss, loss_value, s_m = sess.run([l2_w_query_loss, l2_w_doc_loss, l2_w_lstm_loss, l2_w_ff_loss, loss, similarity_scores], feed_dict={
#            print current_queries_matrix
#            print current_docs_matrix
            loss_value = sess.run(loss, feed_dict={
                query: current_queries_matrix,
                query_mask: current_queries_mask_matrix,
                docs: current_docs_matrix,
                docs_mask: current_docs_mask_matrix
                })
            #print "%s\t%s\t%s\tloss:%s" %(time.strftime('%Y%m%d-%H:%M:%S',time.localtime(time.time())), tmp_k, tmp_k % save_model_iter_num, loss_value)
#print "%s\t%s\t%s\tloss:%s\tl2_w_query_loss:%s\tl2_w_doc_loss:%s\tl2_w_lstm_loss:%s\tl2_w_ff_loss:%s" %(time.strftime('%Y%m%d-%H:%M:%S',time.localtime(time.time())), tmp_k, tmp_k % save_model_iter_num, loss_value, tmp_l2_w_query_loss, tmp_l2_w_doc_loss, tmp_l2_w_lstm_loss, tmp_l2_w_ff_loss)
#print s_m
            
            if math.isnan(loss_value) is True:
                continue

            sess.run([train_op], feed_dict={
                query: current_queries_matrix,
                query_mask: current_queries_mask_matrix,
                docs: current_docs_matrix,
                docs_mask: current_docs_mask_matrix
                })

            training_loss += loss_value
            num_batches += 1

            if tmp_k % save_model_iter_num != 0:
                continue
            if num_batches == 0:
                continue
            training_loss /= num_batches
            num_batches = 0
            val_loss = 0.0
            '''
            # get the query vector
            print 'test query vector'
            #f_for_auc = file(output_for_auc_path, 'w')
            for_auc_dic = dict() 
            for start, end in zip(range(0, len(test_query_data), batch_size), range(batch_size, len(test_query_data) + 1, batch_size)):
                current_queries = test_query_data[start:end]
                current_queries_matrix = np.zeros((batch_size, query_max_length), dtype=np.uint32)
                current_queries_mask_matrix = np.zeros((batch_size, query_max_length), dtype=np.uint32)
                current_docs_matrix = np.zeros((1 + n_negative_samples, batch_size, doc_max_length), dtype=np.uint32)
                current_docs_mask_matrix = np.zeros((1 + n_negative_samples, batch_size, doc_max_length), dtype=np.uint32)
                for i, current_query in enumerate(current_queries):
                    query_id, query_words, doc_words, query_text, doc_text, pv, click, nick, raw_cateids, raw_preference  = current_query
                    query_words = query_words[:query_max_length]
                    current_queries_matrix[i] = query_words + [0] * (query_max_length - len(query_words))
                    current_queries_mask_matrix[i, :len(query_words)] = 1

                    doc_words = doc_words[:doc_max_length]
                    current_docs_mask_matrix[0, i, :len(doc_words)] = 1
                    doc_words = doc_words + [0] * (doc_max_length - len(doc_words))
                    current_docs_matrix[0, i] = doc_words
                 
                similarity_scores_result, valid_loss  = sess.run([similarity_scores, loss], feed_dict={
                    query: current_queries_matrix,
                    query_mask: current_queries_mask_matrix,
                    docs: current_docs_matrix,
                    docs_mask: current_docs_mask_matrix
                })

                for i, current_query in enumerate(current_queries):
                    query_id, query_words, doc_words, query_text, doc_text, pv, click, nick, raw_cateids, raw_preference  = current_query
                    query_text = str(query_text)
                    doc_text = str(doc_text)
                    if "all" not in for_auc_dic:
                        for_auc_dic["all"] = list()
                    if raw_cateids not in for_auc_dic:
                        for_auc_dic[raw_cateids] = list()
                    for_auc_dic["all"].append([ int(pv) - int(click), int(click), (similarity_scores_result[i, 0] + 1.0 )/2  ])
                    for_auc_dic[raw_cateids].append([ int(pv) - int(click), int(click), (similarity_scores_result[i, 0] + 1.0 )/2  ])
#                    print similarity_scores_result[i, 0]
            all_auc, avg_auc, nvzhuang_auc = cal_auc.cal_auc(for_auc_dic)

            #        f_for_auc.write("%s\t%s\t%s\n" %(float((similarity_scores_result[i, 0] + 1.0 )/2), int(pv), int(click)))
            #    f_for_auc.flush()
            #f_for_auc.close()
            #auc_value = "0.0"
            #tmp_s = subprocess.Popen('auc -r %s' %(output_for_auc_path), shell=True, stdout=subprocess.PIPE) 
            #tmp_p = "\t".join(map(str, tmp_s.communicate()))
            #for item in tmp_p.split("\n"):
            #    if item.find("auc") >= 0:
            #        auc_value = item.split("auc:")[1].strip()

            print "%s\tEpoch:%s\tTraining_loss:%s\tvalid_loss:%s\tall_auc:%s\tavg_auc:%s\tnvzhuang_auc:%s\n" %(time.strftime('%Y%m%d-%H:%M:%S',time.localtime(time.time())), tmp_k / save_model_iter_num, training_loss, valid_loss, all_auc, avg_auc, nvzhuang_auc)
#print "%s\t%s\t%s\tloss:%s\tl2_w_query_loss:%s\tl2_w_doc_loss:%s\tl2_w_lstm_loss:%s\tl2_w_ff_loss:%s" %(time.strftime('%Y%m%d-%H:%M:%S',time.localtime(time.time())), tmp_k, tmp_k % save_model_iter_num, loss_value, tmp_l2_w_query_loss, tmp_l2_w_doc_loss, tmp_l2_w_lstm_loss, tmp_l2_w_ff_loss)
            '''
            saver.save(sess, model_path, global_step=tmp_k / save_model_iter_num)
            with open(log_path, 'a') as f:
                f.write("%s\t%s\t%s\tloss:%s\n" %(time.strftime('%Y%m%d-%H:%M:%S',time.localtime(time.time())), tmp_k, tmp_k % save_model_iter_num, loss_value))
                #f.write("%s\tEpoch:%s\tTraining_loss:%s\tvalid_loss:%s\tall_auc:%s\tavg_auc:%s\tnvzhuang_auc:%s\n" %(time.strftime('%Y%m%d-%H:%M:%S',time.localtime(time.time())), tmp_k / save_model_iter_num, training_loss, valid_loss, all_auc, avg_auc, nvzhuang_auc))
#f.write("%s\t%s\t%s\tloss:%s\tl2_w_query_loss:%s\tl2_w_doc_loss:%s\tl2_w_lstm_loss:%s\tl2_w_ff_loss:%s\n" %(time.strftime('%Y%m%d-%H:%M:%S',time.localtime(time.time())), tmp_k, tmp_k % save_model_iter_num, loss_value, tmp_l2_w_query_loss, tmp_l2_w_doc_loss, tmp_l2_w_lstm_loss, tmp_l2_w_ff_loss))
                f.flush()


def predict(test_path,
          n_negative_samples,
          query_max_length,
          dim_query_embed,
          lstm_query_hiddens,
          ff_query_hiddens,
          dim_doc_embed,
          lstm_doc_hiddens,
          ff_doc_hiddens,
          doc_max_length,
          batch_size,
          n_epochs,
          lr,
          smoothing_factor,
          model_path,
          query_dict_path,
          doc_dict_path,
          log_path,
          share_embed,
          query_trainable,
          doc_trainable,
          query_pretrain,
          doc_pretrain,
          tag,
          model_edition
          ):

    query_data = words_to_integers_with_dictionary_return_token_tjm(test_path, query_dict_path[2], doc_dict_path[2])

    query_n_words = len(query_dict_path[2])
    doc_n_words = len(doc_dict_path[2])

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    #sess = tf.Session()
    model = DSSM_Model(
        batch_size=batch_size,
        n_negative_samples=n_negative_samples,
        query_max_length=query_max_length,
        lstm_query_hiddens=lstm_query_hiddens,
        ff_query_hiddens=ff_query_hiddens,
        dim_query_embed=dim_query_embed,
        dim_doc_embed=dim_doc_embed,
        lstm_doc_hiddens=lstm_doc_hiddens,
        ff_doc_hiddens=ff_doc_hiddens,
        doc_max_length=doc_max_length,
        query_n_words=query_n_words,
        doc_n_words=doc_n_words,
        share_embed=share_embed,
        smoothing_factor=smoothing_factor,
        query_trainable=query_trainable,
        doc_trainable=doc_trainable
        )


    normalized_query_hidden, query, query_mask = model.get_normalized_query_vector()
    normalized_doc_hidden, doc, doc_mask = model.get_normalized_doc_vector()

    saver = tf.train.Saver()
    saver.restore(sess, model_path + model_edition)
    f = open('data/predict/predict_result_new_'+model_edition,'w')
    # get the query vector
    print 'getting the query vector'
    for start, end in zip(range(0, len(query_data), batch_size), range(batch_size, len(query_data) + 1, batch_size)):
        current_queries = query_data[start:end]
        current_queries_matrix = np.zeros((batch_size, query_max_length), dtype=np.uint32)
        current_queries_mask_matrix = np.zeros((batch_size, query_max_length), dtype=np.uint32)
        current_docs_matrix = np.zeros((batch_size, doc_max_length), dtype=np.uint32)
        current_docs_mask_matrix = np.zeros((batch_size, doc_max_length), dtype=np.uint32)
        for i, current_query in enumerate(current_queries):
            query_id, query_words, doc_words, nick, shop_id,item_id, pv, click, score1,score2,score3,score4,score5  = current_query

            query_words = query_words[:query_max_length]
            current_queries_matrix[i] = query_words + [0] * (query_max_length - len(query_words))
            current_queries_mask_matrix[i, :len(query_words)] = 1

            doc_words = doc_words[:doc_max_length]
            current_docs_mask_matrix[i, :len(doc_words)] = 1
            doc_words = doc_words + [0] * (doc_max_length - len(doc_words))
            current_docs_matrix[i] = doc_words
         
        current_normalized_query_hidden = sess.run(normalized_query_hidden, feed_dict={
            query: current_queries_matrix,
            query_mask: current_queries_mask_matrix
        })

        current_normalized_doc_hidden = sess.run(normalized_doc_hidden, feed_dict={
            doc: current_docs_matrix,
            doc_mask: current_docs_mask_matrix
        })

        similarity_score = tf.reduce_sum(tf.multiply(current_normalized_query_hidden, current_normalized_doc_hidden), 1)
        similarity_score = sess.run(similarity_score)

        for i, current_query in enumerate(current_queries):
            query_id, query_words, doc_words, nick, shop_id,item_id, pv, click, score1,score2,score3,score4,score5  = current_query
            #query_text = str(query_text)
            #doc_text = str(doc_text)
            f.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" %(nick,shop_id,item_id,pv,click,score1,score2,score3,score4,score5,similarity_score[i]))
            
            #print "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" %(query_id, query_text, doc_text, pv, click, (similarity_score[i] + 1.0 )/2, nick, raw_cateids, raw_preference.strip(), ",".join(map(keep_dim, current_normalized_query_hidden[i])), ",".join(map(keep_dim, current_normalized_doc_hidden[i])))
    f.close()

#    # get the nearest docs for each query
#    print 'calculating the similarity'
#    f = open(result_path, 'w')
#    for i in range((len(query_text) / batch_size) * batch_size):
#        f.write(query_text[i] + '\n')
#        scores = (doc_vector * query_vector[i]).sum(axis=1)
#        index = np.argpartition(-scores, range(1, 11))[:10]
#        for j in range(10):
#            f.write('%g\t%s\n' % (scores[index[j]], doc_text[index[j]]))
#        f.write('\n')
#    f.close()

def predict_doc_vector(test_path,
          n_negative_samples,
          query_max_length,
          dim_query_embed,
          lstm_query_hiddens,
          ff_query_hiddens,
          dim_doc_embed,
          lstm_doc_hiddens,
          ff_doc_hiddens,
          doc_max_length,
          batch_size,
          n_epochs,
          lr,
          smoothing_factor,
          model_path,
          query_dict_path,
          doc_dict_path,
          log_path,
          share_embed,
          query_trainable,
          doc_trainable,
          query_pretrain,
          doc_pretrain,
          tag,
          model_edition
          ):

    query_data = single_text_to_integers_with_dictionary_return_token(test_path, -1, doc_dict_path[2])

    batch_size = min(batch_size, len(query_data))

    query_n_words = len(query_dict_path[2])
    doc_n_words = len(doc_dict_path[2])

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    #sess = tf.Session()
    model = DSSM_Model(
        batch_size=batch_size,
        n_negative_samples=n_negative_samples,
        query_max_length=query_max_length,
        lstm_query_hiddens=lstm_query_hiddens,
        ff_query_hiddens=ff_query_hiddens,
        dim_query_embed=dim_query_embed,
        dim_doc_embed=dim_doc_embed,
        lstm_doc_hiddens=lstm_doc_hiddens,
        ff_doc_hiddens=ff_doc_hiddens,
        doc_max_length=doc_max_length,
        query_n_words=query_n_words,
        doc_n_words=doc_n_words,
        share_embed=share_embed,
        smoothing_factor=smoothing_factor,
        query_trainable=query_trainable,
        doc_trainable=doc_trainable
        )

    normalized_doc_hidden, doc, doc_mask = model.get_normalized_doc_vector()

    saver = tf.train.Saver()
    saver.restore(sess, model_path + model_edition)

    # get the doc vector
    print "start"
    for start, end in zip(range(0, len(query_data), batch_size), range(batch_size, len(query_data) + 1, batch_size)):
        print start,end
        current_queries = query_data[start:end]
        current_docs_matrix = np.zeros((batch_size, doc_max_length), dtype=np.uint32)
        current_docs_mask_matrix = np.zeros((batch_size, doc_max_length), dtype=np.uint32)
        for i, current_query in enumerate(current_queries):
            doc_words, no_seged_text, token_list  = current_query
            doc_words = doc_words[:doc_max_length]
            current_docs_mask_matrix[i, :len(doc_words)] = 1
            doc_words = doc_words + [0] * (doc_max_length - len(doc_words))
            current_docs_matrix[i] = doc_words
         
        current_normalized_doc_hidden = sess.run(normalized_doc_hidden, feed_dict={
            doc: current_docs_matrix,
            doc_mask: current_docs_mask_matrix
        })

        for i, current_query in enumerate(current_queries):
            doc_words, no_seged_text, token_list  = current_query
            print "%s\t%s\t%s" %(no_seged_text, "\t".join(token_list), ",".join(map(keep_dim,current_normalized_doc_hidden[i])))

def keep_dim(source):
    return "%.4f" %(source)

def predict_query_vector(test_path,
          n_negative_samples,
          query_max_length,
          dim_query_embed,
          lstm_query_hiddens,
          ff_query_hiddens,
          dim_doc_embed,
          lstm_doc_hiddens,
          ff_doc_hiddens,
          doc_max_length,
          batch_size,
          n_epochs,
          lr,
          smoothing_factor,
          model_path,
          query_dict_path,
          doc_dict_path,
          log_path,
          share_embed,
          query_trainable,
          doc_trainable,
          query_pretrain,
          doc_pretrain,
          tag,
          model_edition
          ):
 
    query_data = single_text_to_integers_with_dictionary_return_token(test_path, -1, query_dict_path[2])

    query_n_words = len(query_dict_path[2])
    doc_n_words = len(doc_dict_path[2])

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    #sess = tf.Session()
    model = DSSM_Model(
        batch_size=batch_size,
        n_negative_samples=n_negative_samples,
        query_max_length=query_max_length,
        lstm_query_hiddens=lstm_query_hiddens,
        ff_query_hiddens=ff_query_hiddens,
        dim_query_embed=dim_query_embed,
        dim_doc_embed=dim_doc_embed,
        lstm_doc_hiddens=lstm_doc_hiddens,
        ff_doc_hiddens=ff_doc_hiddens,
        doc_max_length=doc_max_length,
        query_n_words=query_n_words,
        doc_n_words=doc_n_words,
        share_embed=share_embed,
        smoothing_factor=smoothing_factor,
        query_trainable=query_trainable,
        doc_trainable=doc_trainable
        )

    normalized_query_hidden, query, query_mask = model.get_normalized_query_vector()

    saver = tf.train.Saver()
    saver.restore(sess, model_path + model_edition)

    # get the query vector
    for start, end in zip(range(0, len(query_data), batch_size), range(batch_size, len(query_data) + 1, batch_size)):
        current_queries = query_data[start:end]
        current_queries_matrix = np.zeros((batch_size, query_max_length), dtype=np.uint32)
        current_queries_mask_matrix = np.zeros((batch_size, query_max_length), dtype=np.uint32)
        current_docs_matrix = np.zeros((batch_size, doc_max_length), dtype=np.uint32)
        current_docs_mask_matrix = np.zeros((batch_size, doc_max_length), dtype=np.uint32)
        for i, current_query in enumerate(current_queries):
            query_words, no_seged_text, token_list  = current_query

            query_words = query_words[:query_max_length]
            current_queries_matrix[i] = query_words + [0] * (query_max_length - len(query_words))
            current_queries_mask_matrix[i, :len(query_words)] = 1

        current_normalized_query_hidden = sess.run(normalized_query_hidden, feed_dict={
            query: current_queries_matrix,
            query_mask: current_queries_mask_matrix
        })

        for i, current_query in enumerate(current_queries):
            doc_words, no_seged_text, token_list  = current_query
            print "%s\t%s\t%s" %(no_seged_text, "\t".join(token_list), ",".join(map(keep_dim,current_normalized_query_hidden[i])))


