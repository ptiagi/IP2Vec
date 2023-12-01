import numpy as np

class ip2vec(object):
    
    def __init__(self, X, Y, vocab_size, emb_size, learning_rate, epochs, batch_size=256, parameters=None, print_cost=True, plot_cost=True):
        self.X = X
        self.Y = Y 
        self.vocab_size = vocab_size
        self.emb_size = emb_size 
        self.learning_rate = learning_rate 
        self.epochs = epochs
        self.batch_size = batch_size 
        self.parameters = parameters 
        self.print_cost=print_cost 
        self.plot_cost=plot_cost

    def initialize_wrd_emb(self, vocab_size, emb_size):
        """
        vocab_size: int. vocabulary size of your corpus or training data
        emb_size: int. word embedding size. How many dimensions to represent each vocabulary
        """
        WRD_EMB = np.random.randn(vocab_size, emb_size) * 0.01
        return WRD_EMB

    def initialize_dense(self, input_size, output_size):
        """
        input_size: int. size of the input to the dense layer
        output_szie: int. size of the output out of the dense layer
        """
        W = np.random.randn(output_size, input_size) * 0.01
        return W

    def initialize_parameters(self, vocab_size, emb_size):
        """
        initialize all the trianing parameters
        """
        WRD_EMB = self.initialize_wrd_emb(self.vocab_size, self.emb_size)
        W = self.initialize_dense(emb_size, vocab_size)
        
        parameters = {}
        parameters['WRD_EMB'] = WRD_EMB
        parameters['W'] = W
        
        return parameters

    def ind_to_word_vecs(self, inds, parameters):
        """
        inds: numpy array. shape: (1, m)
        parameters: dict. weights to be trained
        """
        m = inds.shape[1]
        WRD_EMB = parameters['WRD_EMB']
        word_vec = WRD_EMB[inds.flatten(), :].T
        
        assert(word_vec.shape == (WRD_EMB.shape[1], m))
        
        return word_vec

    def linear_dense(self, word_vec, parameters):
        """
        word_vec: numpy array. shape: (emb_size, m)
        parameters: dict. weights to be trained
        """
        m = word_vec.shape[1]
        W = parameters['W']
        Z = np.dot(W, word_vec)
        
        assert(Z.shape == (W.shape[0], m))
        
        return W, Z

    def softmax(self, Z):
        """
        Z: output out of the dense layer. shape: (vocab_size, m)
        """
        softmax_out = np.divide(np.exp(Z), np.sum(np.exp(Z), axis=0, keepdims=True) + 0.001)
        
        assert(softmax_out.shape == Z.shape)

        return softmax_out

    def forward_propagation(self, inds, parameters):
        word_vec = self.ind_to_word_vecs(inds, parameters)
        W, Z = self.linear_dense(word_vec, parameters)
        softmax_out = self.softmax(Z)
        
        caches = {}
        caches['inds'] = inds
        caches['word_vec'] = word_vec
        caches['W'] = W
        caches['Z'] = Z
        
        return softmax_out, caches

    def cross_entropy(self, softmax_out, Y):
        """
        softmax_out: output out of softmax. shape: (vocab_size, m)
        """
        m = softmax_out.shape[1]
        cost = -(1 / m) * np.sum(np.sum(Y * np.log(softmax_out + 0.001), axis=0, keepdims=True), axis=1)
        return cost

    def softmax_backward(self, Y, softmax_out):
        """
        Y: labels of training data. shape: (vocab_size, m)
        softmax_out: output out of softmax. shape: (vocab_size, m)
        """
        dL_dZ = softmax_out - Y
        
        assert(dL_dZ.shape == softmax_out.shape)
        return dL_dZ

    def dense_backward(self, dL_dZ, caches):
        """
        dL_dZ: shape: (vocab_size, m)
        caches: dict. results from each steps of forward propagation
        """
        W = caches['W']
        word_vec = caches['word_vec']
        m = word_vec.shape[1]
        
        dL_dW = (1 / m) * np.dot(dL_dZ, word_vec.T)
        dL_dword_vec = np.dot(W.T, dL_dZ)

        assert(W.shape == dL_dW.shape)
        assert(word_vec.shape == dL_dword_vec.shape)
        
        return dL_dW, dL_dword_vec

    def backward_propagation(self, Y, softmax_out, caches):
        dL_dZ = self.softmax_backward(Y, softmax_out)
        dL_dW, dL_dword_vec = self.dense_backward(dL_dZ, caches)
        
        gradients = dict()
        gradients['dL_dZ'] = dL_dZ
        gradients['dL_dW'] = dL_dW
        gradients['dL_dword_vec'] = dL_dword_vec
        
        return gradients

    def update_parameters(self, parameters, caches, gradients, learning_rate):
        vocab_size, emb_size = parameters['WRD_EMB'].shape
        inds = caches['inds']
        WRD_EMB = parameters['WRD_EMB']
        dL_dword_vec = gradients['dL_dword_vec']
        m = inds.shape[-1]
        
        WRD_EMB[inds.flatten(), :] -= dL_dword_vec.T * learning_rate

        parameters['W'] -= learning_rate * gradients['dL_dW']

    def skipgram_model_training(self):
        """
        X: Input word indices. shape: (1, m)
        Y: One-hot encodeing of output word indices. shape: (vocab_size, m)
        vocab_size: vocabulary size of your corpus or training data
        emb_size: word embedding size. How many dimensions to represent each vocabulary
        learning_rate: alaph in the weight update formula
        epochs: how many epochs to train the model
        batch_size: size of mini batch
        parameters: pre-trained or pre-initialized parameters
        print_cost: whether or not to print costs during the training process
        """
        costs = []
        m = self.X.shape[1]
        
        if self.parameters is None:
            self.parameters = self.initialize_parameters(self.vocab_size, self.emb_size)
        
        for epoch in range(self.epochs):
            epoch_cost = 0
            batch_inds = list(range(0, m, self.batch_size))
            np.random.shuffle(batch_inds)
            for i in batch_inds:
                X_batch = self.X[:, i:i+self.batch_size]
                Y_batch = self.Y[:, i:i+self.batch_size]

                softmax_out, caches = self.forward_propagation(X_batch, self.parameters)
                gradients = self.backward_propagation(Y_batch, softmax_out, caches)
                self.update_parameters(self.parameters, caches, gradients, self.learning_rate)
                cost = self.cross_entropy(Y_batch, softmax_out)
                epoch_cost += np.squeeze(cost)
                
            costs.append(epoch_cost)
            if self.print_cost and epoch % (self.epochs // 100) == 0:
                print("Cost after epoch {}: {}".format(epoch, epoch_cost))
            if epoch % (self.epochs // 50) == 0:
                self.learning_rate *= 0.98
                
        if self.plot_cost:
            plt.plot(np.arange(self.epochs), costs)
            plt.xlabel('# of epochs')
            plt.ylabel('cost')
        return self.parameters