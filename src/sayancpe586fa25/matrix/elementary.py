import torch
def rowswap(matrix: torch.Tensor, i: int, j: int) -> torch.Tensor:
    """
    Swap row i and row j in a matrix.

    Args:
        matrix: Input matrix.
        i : Index of the first row.
        j : Index of the second row.

    Returns:
        torch.Tensor: matrix with row i,j swapped.
    """
    result = matrix.clone()
    result[[i, j]] = result[[j, i]]
    return result

def rowscale(matrix: torch.Tensor, i: int, s_f: float) -> torch.Tensor:
    """
    Scale row i by a factor s_f.

    Args:
        matrix: input matrix.
        i : Index of the row to scale.
        s_f (float): Scaling factor.

    Returns:
        torch.Tensor: matrix with the scaled row i by factor s_f.
    """
    result = matrix.clone()
    result[i] = result[i] * s_f
    return result


def rowreplacement(matrix: torch.Tensor, i: int, j: int, sf_a: float, sf_b: float) -> torch.Tensor:
    """
    Replace the ith row with: iRi + jRj.

    Args:
        matrix (torch.Tensor): input matrix.
        i (int): Index of the row to be replaced.
        j (int): Index of the source row.
        sf_a (float): Scaling factor for row i.
        sf_b (float): Scaling factor for row j.

    Returns:
        torch.Tensor: Result Matrix
    """
    result = matrix.clone()
    result[i] = sf_a * result[i] + sf_b * result[j]
    return result

def rref(matrix: torch.Tensor) -> torch.Tensor:
    """
    Performs Reduced Row Echelion Form or Gausian Elemination on Marix

    Args:
        matrix (torch.Tensor): input matrix
    Returns:
        torch.Tensor: Result matrix RREF of input
    """
    result = matrix.clone().float()
    size=result.shape
    if len(size) < 2:
        raise TypeError("Matrix Should be 2-D")
    else:
        print(size)
        current_row = 0
        for current_col in range(size[1]):
            if current_row >= size[0]:
                break;
            my_pivot = None
            for row in range(current_row,size[0]):
                if result[row,current_col] != 0:
                    my_pivot = row
                    break
            if my_pivot==None:
                continue
            if my_pivot != current_row:
                result = rowswap(result, current_row, my_pivot)
            scale = result[current_row,current_col]
            result = rowscale(result, current_row, 1/scale)
            for row in range(current_row+1,size[0]):
                if result[row,current_col] != 0:
                    scale = result[row,current_col]
                    result = rowreplacement(result,row,current_row,1.0,-scale)
            current_row += 1
        return result
def main():
    print(rref(torch.Tensor([[1, 2, 3],[4,5,6],[7,8,9]])))
    print(rref(torch.Tensor([[1, 3, 0],[1, 0 ,9],[0, -1,4]])))
    print(rref(torch.Tensor([[1, 3, 0, 0, 3],[0, 0, 1, 0 ,9],[0,0,0,-1,4]])))
if __name__ == "__main__":
    main()
