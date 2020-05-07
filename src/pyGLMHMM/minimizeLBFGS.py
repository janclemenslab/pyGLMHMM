import torch
import torch.optim
from LBFGS import FullBatchLBFGS

def _minimize_LBFGS(objective_function, x_initial, lr = 1, max_iter = 500, tol = 1e-5, line_search = 'Wolfe', interpolate = True, max_ls = 25, history_size = 100, out = True):
    
    model = ModelfromFunction(objective_function, x_initial)
    
    # Define optimizer
    optimizer = FullBatchLBFGS(model.parameters(), lr = lr, history_size = history_size, line_search = line_search, debug = False)
    
    # Main training loop
    if out == True:
        print('===================================================================================')
        print('Solving the Minimization Problem')
        print('===================================================================================')
        print('    Iter:    |     F       |    ||g||    | |x - y|/|x| |   F Evals   |    alpha    ')
        print('-----------------------------------------------------------------------------------')
    
    func_evals = 0
    
    optimizer.zero_grad()
    obj = model()
    obj.backward()
    grad = model.grad()
    func_evals = func_evals + 1
    
    x_old = model.x().clone()
    x_new = x_old.clone()
    f_old = obj
    
    # Main loop
    for n_iter in range(0, max_iter):
    
        # Define closure for line search
        def closure():
            optimizer.zero_grad()
            loss_fn = model()
            return loss_fn
    
        # Perform line search step
        options = {'closure': closure, 'current_loss': obj, 'eta': 2, 'max_ls': max_ls, 'interpolate': interpolate, 'inplace': False}
        if line_search == 'Armijo':
            obj, lr, backtracks, clos_evals, desc_dir, fail = optimizer.step(options = options)
    
            # Compute gradient at new iterate
            obj.backward()
            grad = optimizer._gather_flat_grad()
    
        elif (line_search == 'Wolfe'):
            obj, grad, lr, backtracks, clos_evals, grad_evals, desc_dir, fail = optimizer.step(options = options)
    
        x_new.copy_(model.x())
    
        func_evals = func_evals + clos_evals
    
        # Compute quantities for checking convergence
        grad_norm = torch.norm(grad)
        x_dist = torch.norm(x_new - x_old) / torch.norm(x_old)
        f_dist = torch.abs(obj - f_old) / torch.max(torch.tensor(1, dtype=torch.float), torch.abs(f_old))
    
        # Print data
        if out == True:
            print('  %.3e  |  %.3e  |  %.3e  |  %.3e  |  %.3e  |  %.3e  ' %(n_iter + 1, obj.item(), grad_norm.item(), x_dist.item(), clos_evals, lr))
    
        # Stopping criterion
        if fail == True or torch.isnan(obj) or n_iter == max_iter - 1:
            break
        elif torch.norm(grad) < tol or x_dist < 1e-5 or f_dist < 1e-9 or obj.item() == -float('inf'):
            break
    
        x_old.copy_(x_new)
        f_old.copy_(obj)
    
    # print summary
    print('==================================== Summary ======================================')
    print('Iterations:', n_iter + 1)
    print('Function Evaluations:', func_evals)
    print('F:', obj.item())
    print('||g||:', torch.norm(grad).item())
    print('===================================================================================')
    
    return x_new.clone().detach().numpy()

class ModelfromFunctionGrad(torch.autograd.Function):
    """
    Converts an objective function to PyTorch function.
    """

    @staticmethod
    def forward(ctx, input, problem):
        x = input.clone().detach().numpy()
        obj, grad = problem(x)
        ctx.save_for_backward(torch.tensor(grad, dtype=torch.float))
        return torch.tensor(obj, dtype=torch.float)

    @staticmethod
    def backward(ctx, grad_output):
        grad, = ctx.saved_tensors
        return grad, None

class ModelfromFunction(torch.nn.Module):
    """
    Converts an objective function to torch neural network module.
    """

    def __init__(self, problem, x0):
        super(ModelfromFunction, self).__init__()
        # Get initialization
        x = torch.tensor(x0, dtype=torch.float)
        x.requires_grad_()

        # Store variables and problem
        self.variables = torch.nn.Parameter(x)
        self.problem = problem

    def forward(self):
        model = ModelfromFunctionGrad.apply
        return model(self.variables, self.problem)

    def grad(self):
        return self.variables.grad

    def x(self):
        return self.variables