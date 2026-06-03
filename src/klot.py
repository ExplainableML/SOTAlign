import torch
import torch.nn as nn

class KLOT(nn.Module):
    '''
    KLOT objective from [1]
    
    KLOT(K, K^*) = KL(P^* || P) where P and P^* are the optimal transport plans for K and K^* respectively.
    
    Gradient is computed using theorem 5.1 from [1].
    
    [1] SOTAlign: Semi-Supervised Alignment of Unimodal Vision and Language Models via Optimal Transport. arXiv preprint arXiv:2602.23353.
    '''
    
    def __init__(self, epsilon_student=0.05, epsilon_teacher=0.005, n_iter=100, return_solver_log=False):
        super(KLOT, self).__init__()
        self.epsilon_student = epsilon_student
        self.epsilon_teacher = epsilon_teacher
        self.n_iter = n_iter
        self.return_solver_log = return_solver_log

    def forward(self, K_student, K_teacher):
        
        with torch.no_grad():
            P_student, log_P_student, log_solver_student = sinkhorn(-K_student, epsilon=self.epsilon_student, max_iter=self.n_iter)
            P_teacher, log_P_teacher, log_solver_teacher = sinkhorn(-K_teacher, epsilon=self.epsilon_teacher, max_iter=self.n_iter)

        # This expresion yields the correct gradient (wrong value) :
        kl_correct_grad = ((P_student - P_teacher).detach() * K_student).sum() / self.epsilon_student 
        # This expression yields the correct value (wrong gradient) :
        kl_correct_value = (P_teacher * (log_P_teacher - log_P_student)).sum()
        # This expression yields the correct value and gradient :
        kl = (kl_correct_value - kl_correct_grad).detach() + kl_correct_grad
        
        if self.return_solver_log:
            return kl, log_solver_student, log_solver_teacher
        else:
            return kl
        
def sinkhorn(cost, epsilon, max_iter, check_convergence_every=None, tol=1e-6, symmetric=True):

    K = -cost / epsilon
    
    n, m = K.shape
    a = torch.ones(n, device=K.device) / n
    b = torch.ones(m, device=K.device) / m
    u = torch.zeros((n), dtype=K.dtype, device=K.device)  # u = torch.log(a) - torch.logsumexp(K, dim=2).squeeze()
    v = torch.zeros((m), dtype=K.dtype, device=K.device)  # v = torch.log(b) - torch.logsumexp(K, dim=1).squeeze()

    n_iters = 0
    for n_iters in range(max_iter):
        u = torch.log(a) - torch.logsumexp(K + v[None, :], dim=1).squeeze()
        v = torch.log(b) - torch.logsumexp(K + u[:, None], dim=0).squeeze()

        # Check convergence once every {check_convergence_every} iterations
        if check_convergence_every is not None:
            if n_iters % check_convergence_every == 0:
                T = torch.exp(K + u[:, None] + v[None, :])
                marginal = torch.sum(T, dim=1)
                err = torch.mean(torch.abs(marginal - a))
                if err < tol:
                    break

    if symmetric: # Make it more symmetric, no marginals are exactly satisfied, both are approximately satisfied.
        u_extra = torch.log(a) - torch.logsumexp(K + v[None, :], dim=1).squeeze()
        v_extra = torch.log(b) - torch.logsumexp(K + u[:, None], dim=0).squeeze()
        u = 0.5 * (u + u_extra)
        v = 0.5 * (v + v_extra)
        
    log_T = K + u[:, None] + v[None, :] #    
    T = torch.exp(log_T)
    marginal = torch.sum(T, dim=1)
    err = torch.mean(torch.abs(marginal - a))
    
    log_solver = {"n_iters": n_iters, "err": err}

    return T, log_T, log_solver

if __name__ == "__main__":
    # Demo
    torch.set_printoptions(precision=2, sci_mode=False)
    
    n = 5
    lr = 1e-2
    n_grad_steps = 1000
    
    K_teacher = 10*torch.eye(n)
    K_student = torch.rand((n, n), requires_grad=True)
    klot = KLOT(epsilon_student=0.1, epsilon_teacher=0.1, n_iter=100)
    
    for iteration in range(n_grad_steps):
        kl = klot(K_student, K_teacher)
        kl.backward()
        with torch.no_grad():
            K_student -= lr * K_student.grad
            K_student.grad.zero_()
        if iteration % 10 == 0:
            print("KL:", kl.item())
    
    
    print("Final KL:", kl.item())
    print("Final P_student:")
    print(sinkhorn(-K_student, epsilon=0.1, max_iter=100)[0])
    print("Final P_teacher:")
    print(sinkhorn(-K_teacher, epsilon=0.1, max_iter=100)[0])