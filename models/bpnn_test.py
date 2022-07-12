# -*- coding: utf-8 -*-
import random
import torch
from torch import nn
import math
import numpy as np
from sklearn.model_selection import train_test_split


def example():

    class DynamicNet(torch.nn.Module):

        def __init__(self):
            """
            In the constructor we instantiate five parameters and assign them as members.
            """
            super().__init__()
            self.a = torch.nn.Parameter(torch.randn(()))
            self.b = torch.nn.Parameter(torch.randn(()))
            self.c = torch.nn.Parameter(torch.randn(()))
            self.d = torch.nn.Parameter(torch.randn(()))
            self.e = torch.nn.Parameter(torch.randn(()))

        def forward(self, x):
            """
            For the forward pass of the model, we randomly choose either 4, 5
            and reuse the e parameter to compute the contribution of these orders.

            Since each forward pass builds a dynamic computation graph, we can use normal
            Python control-flow operators like loops or conditional statements when
            defining the forward pass of the model.

            Here we also see that it is perfectly safe to reuse the same parameter many
            times when defining a computational graph.
            """
            y = self.a + self.b * x + self.c * x**2 + self.d * x**3
            for exp in range(4, random.randint(4, 6)):
                y += self.e * x**exp
            return y

        def string(self):
            """
            Just like any class in Python, you can also define custom method on PyTorch modules
            """
            return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3 + {self.e.item()} x^4 ? + {self.e.item()} x^5 ?'

    # Create Tensors to hold input and outputs.
    x = torch.linspace(-math.pi, math.pi, 2000)
    y = torch.sin(x)

    # Construct our model by instantiating the class defined above
    model = DynamicNet()

    # Construct our loss function and an Optimizer. Training this strange model with
    # vanilla stochastic gradient descent is tough, so we use momentum
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)
    for t in range(30000):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        loss = criterion(y_pred, y)
        if t % 2000 == 1999:
            print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Result: {model.string()}')


def prepare_data(xs, ys, training):
    #train_x, test_x, train_y, test_y = train_test_split(data[:, :-1],
    #                                                    data[:, -1],
    #                                                    test_size=0.3,
    #                                                    random_state=124)


    return xs, ys


class atomicNet(torch.nn.Module):
    """
    Constructs NN for H, C, N, O, and F individually
    """

    def __init__(self, Gs_num: int):
        super().__init__()
        self.h = nn.Sequential(
            nn.Linear(Gs_num, 2 * Gs_num),
            nn.Linear(2 * Gs_num, Gs_num),
            nn.Linear(Gs_num, 1),
        )
        self.c = nn.Sequential(
            nn.Linear(Gs_num, 2 * Gs_num),
            nn.Linear(2 * Gs_num, Gs_num),
            nn.Linear(Gs_num, 1),
        )
        self.n = nn.Sequential(
            nn.Linear(Gs_num, 2 * Gs_num),
            nn.Linear(2 * Gs_num, Gs_num),
            nn.Linear(Gs_num, 1),
        )
        self.o = nn.Sequential(
            nn.Linear(Gs_num, 2 * Gs_num),
            nn.Linear(2 * Gs_num, Gs_num),
            nn.Linear(Gs_num, 1),
        )
        self.f = nn.Sequential(
            nn.Linear(Gs_num, 2 * Gs_num),
            nn.Linear(2 * Gs_num, Gs_num),
            nn.Linear(Gs_num, 1),
        )
        self.el = {1: self.h, 6: self.c, 7: self.n, 8: self.o, 9: self.f}

    def forward(self, x):
        k = int(x[0])
        v = self.el[k]
        v = v(x[1:])
        return v


def atomic_nn(xs: [], ys: []):
    prepare_data(xs, ys, 0.8)
    for n, i in enumerate(xs):
        i = np.float32(i)
        xs[n] = torch.from_numpy(i)
        ys[n] = torch.tensor(ys[n], dtype=torch.float32)
    Gs_num = xs[0].size()[1] - 1

    learning_rate = 1e-1
    epochs = 300
    model = atomicNet(Gs_num)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        for n, x in enumerate(xs):
            y = ys[n]
            E_tot = 0
            for atom in x:
                E_i = model(atom)
                E_tot += E_i
            E_tot = E_tot[0]
            loss = criterion(E_tot, y)
            if epoch % 50 == 0:
                print(epoch, loss.item())
                # print(E_tot.size(), y.size())
                print(E_tot, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return
