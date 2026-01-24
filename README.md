This is a package with following subpackages:

1. Differential -> function diff to calculate discrete time derivative
  differentialial -> discrete -> diff (x,t) It does discrete time derivative of two vectors (list) x and t and returs a list v
2. Matrix-> 
function rowswap(matrix: torch.Tensor, i: int, j: int) -> torch.Tensor
    Swap row i and row j in a matrix.

    Args:
        matrix: Input matrix.
        i : Index of the first row.
        j : Index of the second row.

    Returns:
        torch.Tensor: matrix with row i,j swapped.


            function rowscale(matrix: torch.Tensor, i: int, s_f: float) -> torch.Tensor
     Scale row i by a factor s_f.

    Args:
        matrix: input matrix.
        i : Index of the row to scale.
        s_f (float): Scaling factor.

    Returns:
        torch.Tensor: matrix with the scaled row i by factor s_f


            function rowreplacement(matrix: torch.Tensor, i: int, j: int, sf_a: float, sf_b: float) -> torch.Tensor
    Replace the ith row with: iRi + jRj.

    Args:
        matrix (torch.Tensor): input matrix.
        i (int): Index of the row to be replaced.
        j (int): Index of the source row.
        sf_a (float): Scaling factor for row i.
        sf_b (float): Scaling factor for row j.

    Returns:
        torch.Tensor: Result Matrix


            function rref(matrix: torch.Tensor) -> torch.Tensor
    Performs Reduced Row Echelion Form or Gausian Elemination on Marix

    Args:
        matrix (torch.Tensor): input matrix
    Returns:
        torch.Tensor: Result matrix RREF of input

            function uniform(a: float = 0.0, b: float = 1.0)
        Returns uniform values between 0 and 1
        E.G Use uniform(0,1)

            function exponentialdist(lmbd: float)
        Returns exponential distribution
    Args:
        Lambda value for exponential distribution 
        E.G. Use exponentialdist(lmbd)

            function poissondist(lmbd: float):
        Returns poisson distribution
    Args:
        Lambda value for poisson distribution
        E.G. Use poissondist(lmbd)

3. Subpackage: distribution
        Class: cvdistribution

4. Subpackage: model
    Added class LinearRegression, CauchyRegression, LogisticRegression
    Added class TorchNet

5. Subpackage: deepl
    Function: binary_classification
    Performs binary classification using gradient descent and autograd.

    Args:
        d (int): number of features
        n (int): number of samples
        epochs (int): number of training epochs (default 10000)
        eta (float): learning rate (default 0.001)

    Returns:
        W1, W2, W3, W4: trained weight matrices
        loss_history: list containing loss value at each epoch
