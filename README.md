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
    
    Class: SimpleNN usage: deepl.SimpleNN(in_features=num_features, num_classes=num_classes)
           ClassTrainer usage: deepl.ClassTrainer(
                                X_train=X_train,
                                Y_train=y_train,
                                model=training model,
                                eta=learning_rate,
                                epochs=epochs,
                                loss_fn=loss_function,
                                optimizer_cls=optimizer
                                )

6. Subpackage: animation
    Class: WeightMatrixAnime, LargeWeightMatrixAnime
    Function: animate_weight_heatmap (uses class WeightMatrixAnime), animate_large_heatmap (uses class LargeWeightMatrixAnime)
        animation.animate_large_heatmap(
            3d_torch_tensor),
            dt=0.04,
            file_name="fileName",
            title_str="Title String"
            )
## HW02Q7
  Here we are doing binary classification using deepl.binary_classification and creating an animation of how the weight matrices evolve per epochs using animation.animate_large_heatmap. Example implementation provided in scripts/binaryclassification_animate_impl.py. The generated media mp4 for every weight matrix will be present inside media/ directory. A wrapper to run this is provided in scripts/binaryclassification_animate_impl.sh. The whole wrapper is also automated by scripts/run_binary_training.sh and logs are saved to scripts/training_log.out. 
    
## HW02Q8
  Here we doing multiclass classification with android malware dataset from https://github.com/rahulbhadani/CPE487587_
SP26/releases/download/android_malware/Android_Malware.csv. This dataset can be downloaded using scripts/malwaredatadownload.sh. The complete implementation is provided in scripts/multiclass_impl.py. This implementation is using deepl.SimpleNN model and deepl.ClassTrainer to achieve this. After each run performance metrics are saved to result/ directory in csv format. Another script, scripts/multiclass_eval.py then takes all the .csv from the result/ directory and generate box plots for all metrics and saves it to result/ directory. An automated implementation of the whole scheme is presented in scripts/multiclass_impl.sh. A wrapper for the whole pipeline presented in scripts/run_multiclass.sh and logs are saved to scripts/multiclass_log.out.   
