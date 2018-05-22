import torch
from torch import nn
import pandas as pd
import torch.nn.functional as F
from torch.nn.init import kaiming_normal
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np
from DataAssignment.src.py.utils.utils import roc_charts, accuracy_score, plot_confusion_matrix, confusion_matrix


def create_embedding_layer(n_elements, embedding_dim):
    """
    Auxiliary function to create and initialize embedding layers

    Parameters
    ----------
    n_elements : int
        number of elements in entity
    embedding_dim: int
        embedding dimension

    Returns
    -------
    torch.nn.Embedding
        embedding layer of shape (#n_elements, #embedding_dim)
    """
    embedding = nn.Embedding(n_elements, embedding_dim)
    embedding.weight.data.uniform_(-0.01, 0.01)
    return embedding


class EmbeddingRecommenderProdCust(nn.Module):
    def __init__(self, n_customers, n_products, keys_embedding_dim,
                 categorical_embedding_dim, numerical_dim, hidden_dim,
                 embedding_drop, hidden_drop=None):
        """
        Recommender based on customer/product embeddings and additional customer/product features

        Parameters
        ----------
        n_customers: int
            number of customers in dataset. Sets customer-embedding layer
        n_products: int
            number of products in dataset. Sets product-embedding layer
        keys_embedding_dim: int
            dimension of the high-dimensional space to project both customers and products
        categorical_embedding_dim: list of Tuple(int, int)
            dimensions of the high-dimensional space to project the categorical features
        numerical_dim: int
            number of numerical features
        hidden_dim: list of int
            list of hidden layer dimensions
        embedding_drop: float
            dropout to apply to the embedding layer
        hidden_drop: list of float
            dropouts to apply to the hidden layers
        """
        # nn.Module initialization
        super().__init__()
        self.hidden_sizes = hidden_dim
        # customer / product embedding initialization
        (self.customer_embed, self.product_embed) = [
            create_embedding_layer(*i)
            for i in [(n_customers, keys_embedding_dim), (n_products, keys_embedding_dim)]
        ]

        # categorical embedding initialization
        self.categorical_embeddings = nn.ModuleList(
            [create_embedding_layer(*i)
             for i in categorical_embedding_dim]
        )

        # sum of categorical embedding dimensions
        categorical_embedding = sum(e.embedding_dim for e in self.categorical_embeddings)

        # fully connected hidden layers initialization
        self.linearOut = nn.Linear(self.hidden_sizes[-1], 1)
        kaiming_normal(self.linearOut.weight.data)
        if len(self.hidden_sizes) > 1:
            self.linears = nn.ModuleList(
                [nn.Linear(keys_embedding_dim*2 + categorical_embedding + numerical_dim, self.hidden_sizes[0])] +
                [nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i + 1])
                 for i in range(0, len(self.hidden_sizes) - 1)]
            )
        else:
            self.linears = [nn.Linear(keys_embedding_dim*2 + categorical_embedding + numerical_dim, self.hidden_sizes[0])]
        for o in self.linears:
            kaiming_normal(o.weight.data)

        # dropout layers initialization
        self.embedding_drop = nn.Dropout(embedding_drop)
        self.dropouts = nn.ModuleList([nn.Dropout(drop) for drop in hidden_drop])

        # batch-normalization layers initialization
        self.batchNorms = nn.ModuleList(
            [nn.BatchNorm1d(self.hidden_sizes[i]) for i in range(0, len(self.hidden_sizes))]
        )
        self.bn = nn.BatchNorm1d(numerical_dim)
    
    def forward(self, x_categorical, x_numerical):
        """
        Base method to do the net's forward pass

        Parameters
        ----------
        x_categorical: torch.Tensor
            tensor with the categorical features
        x_numerical: torch.Tensor
            tensor with the numerical features

        Returns
        ----------
        torch.Tensor
            [batch_size x 1] tensor with predictions

        """

        # customer, product embedding extraction
        customers, products = x_categorical[:, 0], x_categorical[:, 1]
        x = torch.cat([self.customer_embed(customers), self.product_embed(products)], dim=1)  # [bz x emb_custProd*2]

        # categorical embedding extraction
        x_cat = [emb(x_categorical[:, i+2]) for i, emb in enumerate(self.categorical_embeddings)]
        x_cat = torch.cat(x_cat, 1)  # [bz x sum(emb_cat)]

        # embedding concatenation
        x_emb = self.embedding_drop(torch.cat([x, x_cat], 1)) # [bz x sum(emb_cat) + emb_custProd*2]

        # numerical features concatenation
        x2 = self.bn(x_numerical)  # [bz x dim_num]
        x = torch.cat([x_emb, x2], 1)  # [bz x sum(emb_cat) + emb_custProd*2 + dim_num]

        # fully connected pass:
        for linear, drop, bn in zip(self.linears, self.dropouts, self.batchNorms):
            x = drop(bn(F.relu(linear(x))))  # [bz x dim_hl(i)]

        # last fully connected pass
        x = self.linearOut(x)  # [bz x dim_hl(-1)]

        # sigmoid activation (prediction probs in range [0,1]]
        result = F.sigmoid(x)  # [bz x 1]

        # eof
        return result

    def fit(self, x_train, y_train, idx_cat, n_epochs, batch_size, learning_rate, eval_set=None):
        """
        Network fitting method rutine

        Parameters
        ---------
        x_train: dataframe/numpy array
            train features
        y_train: numpy array
            train target
        idx_cat: list
            categorical features column index (needed to split the x_train)
        n_epochs: int
            number of epochs
        batch_size: int
            batch size to use during training
        learning_rate: float
            epochs learning rate
        eval_set: tuple of pandas DataFrame / numpy array
            validation dataset (x_val, y_val)

        Returns
        -------
        self
        """

        # categorical transformation to numeric values
        x_train.iloc[:, idx_cat] = x_train.iloc[:, idx_cat].apply(lambda x: x.astype(int))

        # numerical indexes
        idx_num = [i for i in range(x_train.shape[1]) if i not in idx_cat]

        # tensor transformation of features and target
        tx = torch.from_numpy(np.array(x_train))
        ty = torch.from_numpy(np.array(y_train)).float()

        # "mini-batch" object initialization
        train = Data.TensorDataset(tx, ty)

        # loader object initialization (needed for batch train)
        train_loader = Data.DataLoader(train, batch_size=batch_size, shuffle=True)

        # network optimizer initialization
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # loss funtion initialization
        loss_func = nn.BCELoss()

        # eval set check
        if eval_set is not None:
            do_eval = True
        else:
            do_eval = False

        # training loop
        for epoch in range(n_epochs):
            for data in train_loader:
                x, y = data  # minibatch x, y
                x_cat, x_num = x[:, idx_cat], x[:, idx_num]  # categorical/numerical split

                b_x_cat, b_x_num = Variable(x_cat.long()), Variable(x_num.float())  # batch x
                b_y = Variable(y.float())  # batch y, shape

                # minibatch prediction
                pred = self(b_x_cat, b_x_num)

                loss = loss_func(pred.view(-1), b_y)  # binary cross-entropy
                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # back-propagation, compute gradients
                optimizer.step()  # apply gradients

            # reporting step (useful to monitorize training)
            if epoch % 1 == 0:
                epoch_train_auc, _ = self.evaluate_epoch(x_train, y_train, idx_cat)
                if do_eval:
                    epoch_auc, epoch_acc = self.evaluate_epoch(eval_set[0], eval_set[1], idx_cat)
                    print(epoch, '  train AUC : %.4f || val AUC : %.4f || val ACC : %.4f' %
                          (epoch_train_auc, epoch_auc, epoch_acc))
                else:
                    print(epoch, ' train AUC  : %.4f' % epoch_train_auc)

    def evaluate_epoch(self, x_eval, y_eval, idx_cat):
        """
        Auxiliary method to evaluate network performance in AUC, Accuracy metrics

        Parameters
        -------
        x_eval : DataFrame/numpy array
            features to use for prediction
        y_eval : Series/numpy array
            real values of target variable
        idx_cat : list
            categorical features indexes

        """
        # same transformation than in train
        x_eval.iloc[:, idx_cat] = x_eval.iloc[:, idx_cat].apply(lambda x: x.astype(int))
        idx_num = [i for i in range(x_eval.shape[1]) if i not in idx_cat]
        tx = torch.from_numpy(np.array(x_eval))
        x_cat, x_num = tx[:, idx_cat], tx[:, idx_num]
        b_x_cat, b_x_num = Variable(x_cat.long()), Variable(x_num.float())

        # predictions
        prediction = self(b_x_cat, b_x_num)
        prediction = prediction.data.numpy()
        prediction = np.array([prediction[i][0] for i in range(0, prediction.shape[0])])

        # metric extraction (auc, accuracy, confusion matrix)
        auc_error, _ = roc_charts(y_eval, prediction, "intermediate_eval")
        prediction = np.where(prediction > 0.5, 1, 0)
        acc_error = accuracy_score(y_eval, prediction)
        # confusion matrix plot
        plot_confusion_matrix(confusion_matrix(y_eval, prediction), ['0', '1'])

        # eof
        return auc_error, acc_error

    def extract_product_embeddings(self, products):
        """
        Auxiliary function to extract embeddings for the list of products given

        Parameters
        -----------
        products: list, np.array, pd.Series
            List of products to extract embeddings

        Returns
        -----------
        pd.DataFrame
            product-embedding df
        """
        # products tensor conversion
        products = products.astype(int)
        tproducts = torch.from_numpy(np.array(products))
        tproducts = Variable(tproducts.long())

        # embedding extraction
        emb_products = self.product_embed(tproducts).data.numpy()

        # output DataFrame composition
        columns = ['prod'] + ['emb_%02i' % i for i in range(emb_products.shape[1])]
        df_emb_products = pd.DataFrame(np.column_stack((products, emb_products)), columns=columns)

        #eof
        return df_emb_products

