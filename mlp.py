import torch
import math

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features), # 20 x  2
            b1 = torch.randn(linear_1_out_features), ## 20
            W2 = torch.randn(linear_2_out_features, linear_2_in_features), # 5 x 20
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def forward(self, x): ## Will get one batch 
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        ## W1*x + b1
        W1 = self.parameters['W1']
        b1 = self.parameters['b1']
        s_temp = torch.matmul(x, W1.t()) ## 10 x 20
        s1 = s_temp + b1.unsqueeze(0).expand_as(s_temp)


        ## s1 = f_function :: pointwise (10 x 20)
        f = self.f_function
        if f == 'relu':
            a1 = torch.relu(s1)
        elif f == 'sigmoid':
            a1 = torch.sigmoid(s1)
        elif f == 'identity':
            a1 = s1


        ## W2*s + b2; W2 :: 5 x 20
        W2 = self.parameters['W2']
        b2 = self.parameters['b2']
        s2_temp = torch.matmul(a1, W2.t()) ## 10 x 5
        s2 = s2_temp + b2.unsqueeze(0).expand_as(s2_temp)


        ## y = g_function :: pointwise 10 x 5
        g = self.g_function
        if g == 'relu':
            y_hat = torch.relu(s2)
        if g == 'sigmoid':
            y_hat = torch.sigmoid(s2)
        if g == 'identity':
            y_hat = s2

        self.cache['y_hat'] = y_hat
        self.cache['s2'] = s2
        self.cache['s1'] = s1
        self.cache['x'] = x
        self.cache['a1'] = a1
        return y_hat


    
    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        
        W2 = self.parameters['W2']
        y_hat = self.cache['y_hat']
        g = self.g_function
        f = self.f_function 
        batch_size = self.cache['x'].size()[0]

        ## dJdW1
        dy_hatds2 = derivative(g, self.cache['s2'])
        ## dJdy_hat is 10 x 5 
        ## dy_hatds2 is same as s2 which is 10 x 5

        dJds2 = dJdy_hat*dy_hatds2 
        dJda1 = torch.matmul(dJds2, W2)  # dJds2 :: 10x5 and W2::5x20
        
        da1ds1 = derivative(f, self.cache['s1'])  # 10 x 20
        dJds1 = dJda1 * da1ds1 # dJda1::10 x 20
        dJdW1 = torch.matmul(dJds1.t(), self.cache['x']) # dJds1:: 10 x 20, x :: 10 x 2
        self.grads['dJdW1'] = dJdW1/batch_size

        # dJ/dW2
        dJdW2 = torch.matmul(dJds2.t(), self.cache['a1']) # 5 x 10; 10x20
        self.grads['dJdW2'] = dJdW2/batch_size


        # dJ/db1
        self.grads['dJdb1'] = dJds1.mean(axis = 0)
        # dJ/db2
        self.grads['dJdb2'] = dJds2.mean(axis = 0)


    
    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()


def derivative(activation, x):
    if activation == "relu":
        return (x > 0).float()
    elif activation == "sigmoid":
        return torch.sigmoid(x) * (1 - torch.sigmoid(x))
    elif activation == "identity":
        return torch.ones_like(x) 


def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # loss 
    """
    What are the dimensions of y and y_hat?
    y = 10 x 5 for 10 outputs only 
    thus the final output should be 1 x 5 or 5 x 1 averaged over the batch 
    y_hat too is 10 x 5
    """
    loss = (y - y_hat)**2
    loss_mean = loss.mean() # loss is calculated for each variable say price etc

    dJdy_hat = 2*(y_hat - y)/y.size()[1]

    return loss_mean, dJdy_hat

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    ## Make sure you iterate over this 
    loss = -1*y*torch.log(y_hat) + (1-y)*torch.log(1-y_hat)
    loss = loss.mean()
    dJdy_hat = -1*y/y_hat + (1-y)/(1 - y_hat)
    return loss, dJdy_hat/y.size()[1]











