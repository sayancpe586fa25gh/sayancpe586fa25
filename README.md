This is a package with two subpackages:

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
