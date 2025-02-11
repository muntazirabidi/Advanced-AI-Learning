def write_latex_section(section):
    """Return LaTeX formatted theoretical explanations"""
    if section == "bnn":
        return r"""
        Bayesian Neural Networks extend traditional neural networks by treating weights as probability distributions rather than point estimates.
        
        The posterior distribution over weights is given by:
        
        $$p(w|D) = \frac{p(D|w)p(w)}{p(D)}$$
        
        where:
        - $w$ represents the network weights
        - $D$ is the observed data
        - $p(w)$ is the prior over weights
        - $p(D|w)$ is the likelihood
        - $p(D)$ is the evidence
        
        We typically use variational inference to approximate the posterior:
        
        $$q_\theta(w) \approx p(w|D)$$
        
        minimizing the KL divergence:
        
        $$KL(q_\theta(w)||p(w|D))$$
        """
    
    elif section == "gp":
        return r"""
        A Gaussian Process defines a distribution over functions, fully specified by its mean function $m(x)$ and covariance function $k(x,x')$:
        
        $$f(x) \sim \mathcal{GP}(m(x), k(x,x'))$$
        
        The posterior predictive distribution at a new point $x_*$ is:
        
        $$p(f_*|x_*, X, y) = \mathcal{N}(\mu_*, \sigma_*^2)$$
        
        where:
        
        $$\mu_* = k_*^T(K + \sigma_n^2I)^{-1}y$$
        $$\sigma_*^2 = k_{**} - k_*^T(K + \sigma_n^2I)^{-1}k_*$$
        
        Common kernel choices include:
        - RBF: $k(x,x') = \sigma^2\exp(-\frac{||x-x'||^2}{2l^2})$
        - Matérn: More flexible smoothness assumptions
        - Periodic: For cyclic patterns
        """
    
    return ""