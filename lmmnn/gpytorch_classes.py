import torch
try:
    import gpytorch
except Exception:
    pass

class DKLModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, mlp):
        super(DKLModel, self).__init__(train_x, train_y, likelihood,)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
            num_dims=2, grid_size=100
        )
        self.mlp = mlp
    
    def forward(self, x_gp, x_mlp):
        projected_x = self.mlp(x_mlp)
        mean_x = self.mean_module(x_gp) + projected_x.flatten()
        covar_x = self.covar_module(x_gp)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SVDKLModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, mlp):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(SVDKLModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.mlp = mlp

    def forward(self, x_gp, x_mlp, n_inducing_points):
        projected_x = self.mlp(x_mlp)
        mean_x = self.mean_module(x_gp)
        mean_x2 = mean_x.clone()
        mean_x2[n_inducing_points:] += projected_x.flatten()
        covar_x = self.covar_module(x_gp)
        return gpytorch.distributions.MultivariateNormal(mean_x2, covar_x)


class MLP(torch.nn.Module):
    def __init__(self, data_dim, n_neurons, dropout, activation):
        super(MLP, self).__init__()
        self.layers = torch.nn.ModuleList()
        prev_n_neurons = data_dim
        if len(n_neurons) > 0:
            self.layers.append(torch.nn.Linear(prev_n_neurons, n_neurons[0]))
            self.layers.append(torch.nn.ReLU())
            prev_n_neurons = n_neurons[0]
            if dropout is not None and len(dropout) > 0:
                self.layers.append(torch.nn.Dropout(dropout[0]))
            for i in range(1, len(n_neurons) - 1):
                self.layers.append(torch.nn.Linear(prev_n_neurons, n_neurons[i]))
                self.layers.append(torch.nn.ReLU())
                if dropout is not None and len(dropout) > i:
                    self.layers.append(torch.nn.Dropout(dropout[i]))
                prev_n_neurons = n_neurons[i]
        if len(n_neurons) > 1:
            self.layers.append(torch.nn.Linear(prev_n_neurons, n_neurons[-1]))
            self.layers.append(torch.nn.ReLU())
            prev_n_neurons = n_neurons[-1]
        self.layers.append(torch.nn.Linear(prev_n_neurons, 1))

    def forward(self, x):
        y = x
        for i in range(len(self.layers)):
            y = self.layers[i](y)
        return y
