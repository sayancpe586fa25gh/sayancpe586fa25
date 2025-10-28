import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, Optional
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict

class LinearRegression:
    """
    A PyTorch-based Linear Regression implementation for one variable.
    
    Model: y = w_1 * x + w_0
    Loss: Mean Squared Error
    
    Features:
    - Gradient-based optimization using PyTorch
    - Confidence intervals for parameters w_1 and w_0
    - Visualization with confidence bands
    """
    
    def __init__(self, learning_rate: float = 0.01, max_epochs: int = 1000, 
                 tolerance: float = 1e-6):
        """
        Initialize the Linear Regression model.
        
        Args:
            learning_rate: Learning rate for gradient descent
            max_epochs: Maximum number of training epochs
            tolerance: Convergence tolerance for early stopping
        """
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        
        # Model parameters
        self.w_1 = nn.Parameter(torch.randn(1, requires_grad=True))  # slope
        self.w_0 = nn.Parameter(torch.randn(1, requires_grad=True))  # intercept
        
        # Training data storage
        self.X_train = None
        self.y_train = None
        
        # Model statistics for confidence intervals
        self.n_samples = None
        self.residual_sum_squares = None
        self.X_mean = None
        self.X_var = None
        self.fitted = False
        
        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD([self.w_1, self.w_0], lr=self.learning_rate)
        
        # Training history
        self.loss_history = []
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the linear model.
        
        Args:
            X: Input tensor of shape (n_samples,)
            
        Returns:
            Predictions tensor of shape (n_samples,)
        """
        return self.w_1 * X + self.w_0
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit the linear regression model to the training data.
        
        Args:
            X: Input features of shape (n_samples,)
            y: Target values of shape (n_samples,)
            
        Returns:
            self: Returns the fitted model instance
        """
        # Convert to PyTorch tensors
        self.X_train = torch.tensor(X, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.float32)
        self.n_samples = len(X)
        
        # Store statistics for confidence intervals
        self.X_mean = float(np.mean(X))
        self.X_var = float(np.var(X, ddof=1))  # Sample variance
       
        # Initialize tracking lists
        self.loss_history = []
        self.w0_history = []
        self.w1_history = []
        # Training loop
        prev_loss = float('inf')
        
        for epoch in range(self.max_epochs):
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            y_pred = self.forward(self.X_train)
            
            # Compute loss
            loss = self.criterion(y_pred, self.y_train)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            self.optimizer.step()
            
            # Store loss history
            current_loss = loss.item()
            self.loss_history.append(current_loss)
            self.w0_history.append(self.w_0.item())
            self.w1_history.append(self.w_1.item())   
            # Check for convergence
            if abs(prev_loss - current_loss) < self.tolerance:
                print(f"Converged after {epoch + 1} epochs")
                break
            
            prev_loss = current_loss
        
        # Compute residual sum of squares for confidence intervals
        with torch.no_grad():
            y_pred = self.forward(self.X_train)
            residuals = self.y_train - y_pred
            self.residual_sum_squares = float(torch.sum(residuals ** 2))
        
        self.fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features of shape (n_samples,)
            
        Returns:
            Predictions as numpy array
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        with torch.no_grad():
            predictions = self.forward(X_tensor)
        
        return predictions.numpy()
    
    def get_parameters(self) -> Tuple[float, float]:
        """
        Get the fitted parameters.
        
        Returns:
            Tuple of (w_1, w_0) - slope and intercept
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before accessing parameters")
        
        return float(self.w_1.item()), float(self.w_0.item())
    
    def parameter_confidence_intervals(self, confidence_level: float = 0.95) -> dict:
        """
        Compute confidence intervals for parameters w_1 and w_0.
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            Dictionary containing confidence intervals for both parameters
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before computing confidence intervals")
        
        # Degrees of freedom
        df = self.n_samples - 2
        
        # Critical t-value
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        # Standard error of regression
        mse = self.residual_sum_squares / df
        se_regression = np.sqrt(mse)
        
        # Standard error for w_1 (slope)
        se_w1 = se_regression / np.sqrt(self.n_samples * self.X_var)
        
        # Standard error for w_0 (intercept)
        se_w0 = se_regression * np.sqrt(1/self.n_samples + self.X_mean**2 / (self.n_samples * self.X_var))
        
        # Get current parameter values
        w_1_val, w_0_val = self.get_parameters()
        
        # Compute confidence intervals
        w_1_ci = (
            w_1_val - t_critical * se_w1,
            w_1_val + t_critical * se_w1
        )
        
        w_0_ci = (
            w_0_val - t_critical * se_w0,
            w_0_val + t_critical * se_w0
        )
        
        return {
            'w_1_confidence_interval': w_1_ci,
            'w_0_confidence_interval': w_0_ci,
            'confidence_level': confidence_level,
            'standard_errors': {
                'se_w1': se_w1,
                'se_w0': se_w0,
                'se_regression': se_regression
            }
        }
    
    def plot_regression_with_confidence_band(self, confidence_level: float = 0.95, 
                                           figsize: Tuple[int, int] = (10, 6),
                                           title: Optional[str] = None) -> plt.Figure:
        """
        Plot the fitted regression line with confidence band.
        
        Args:
            confidence_level: Confidence level for the band
            figsize: Figure size tuple
            title: Optional plot title
            
        Returns:
            matplotlib Figure object
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Convert training data to numpy for plotting
        X_np = self.X_train.numpy()
        y_np = self.y_train.numpy()
        
        # Create prediction range
        X_range = np.linspace(X_np.min(), X_np.max(), 100)
        y_pred_range = self.predict(X_range)
        
        # Compute confidence band
        df = self.n_samples - 2
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        mse = self.residual_sum_squares / df
        se_regression = np.sqrt(mse)
        
        # Standard error for predictions (confidence band)
        X_centered = X_range - self.X_mean
        se_pred = se_regression * np.sqrt(1/self.n_samples + X_centered**2 / (self.n_samples * self.X_var))
        
        # Confidence band bounds
        margin_of_error = t_critical * se_pred
        y_upper = y_pred_range + margin_of_error
        y_lower = y_pred_range - margin_of_error
        
        # Plot data points
        ax.scatter(X_np, y_np, alpha=0.6, color='blue', label='Data points')
        
        # Plot regression line
        ax.plot(X_range, y_pred_range, 'r-', linewidth=2, label='Fitted line')
        
        # Plot confidence band
        ax.fill_between(X_range, y_lower, y_upper, alpha=0.3, color='red', 
                       label=f'{int(confidence_level*100)}% Confidence band')
        
        # Get parameter values for display
        w_1_val, w_0_val = self.get_parameters()
        
        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        if title is None:
            title = f'Linear Regression: y = {w_1_val:.3f}x + {w_0_val:.3f}'
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def summary(self) -> dict:
        """
        Provide a summary of the fitted model.
        
        Returns:
            Dictionary containing model summary statistics
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before generating summary")
        
        w_1_val, w_0_val = self.get_parameters()
        
        # R-squared calculation
        y_mean = float(torch.mean(self.y_train))
        ss_tot = float(torch.sum((self.y_train - y_mean) ** 2))
        r_squared = 1 - (self.residual_sum_squares / ss_tot)
        
        # Adjusted R-squared
        adj_r_squared = 1 - ((1 - r_squared) * (self.n_samples - 1) / (self.n_samples - 2))
        
        # RMSE
        rmse = np.sqrt(self.residual_sum_squares / self.n_samples)
        
        return {
            'parameters': {
                'w_1 (slope)': w_1_val,
                'w_0 (intercept)': w_0_val
            },
            'model_fit': {
                'r_squared': r_squared,
                'adjusted_r_squared': adj_r_squared,
                'rmse': rmse,
                'residual_sum_squares': self.residual_sum_squares
            },
            'training_info': {
                'n_samples': self.n_samples,
                'epochs_trained': len(self.loss_history),
                'final_loss': self.loss_history[-1] if self.loss_history else None
            }
        }

    def analysis_plots(self, confidence_level: float = 0.95,
                   figsize: Tuple[int, int] = (12, 10),
                   show_confidence_band: bool = True) -> plt.Figure:
        """
        Create a 3x1 subplot figure showing:
        1. Original data and fitted regression line
        2. Evolution of parameters w_0 and w_1 over epochs
        3. Loss curve over epochs

        Args:
            confidence_level: Confidence level for regression confidence band
            figsize: Size of the figure
            show_confidence_band: Whether to include confidence interval band

        Returns:
            matplotlib Figure object
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting analysis")

        # Create subplots: 3 rows, 1 column
        fig, ax = plt.subplots(3, 1, figsize=figsize)

        # --- (1) Data + Regression Line ---
        X_np = self.X_train.numpy()
        y_np = self.y_train.numpy()
        X_range = np.linspace(X_np.min(), X_np.max(), 100)
        y_pred_range = self.predict(X_range)

        ax[0].scatter(X_np, y_np, color='blue', alpha=0.6, label='Data points')
        ax[0].plot(X_range, y_pred_range, 'r-', linewidth=2, label='Fitted line')

        if show_confidence_band:
            # Compute confidence band
            df = self.n_samples - 2
            alpha = 1 - confidence_level
            t_critical = stats.t.ppf(1 - alpha/2, df)

            mse = self.residual_sum_squares / df
            se_regression = np.sqrt(mse)
            X_centered = X_range - self.X_mean
            se_pred = se_regression * np.sqrt(1/self.n_samples + X_centered**2 / (self.n_samples * self.X_var))
            margin = t_critical * se_pred
            ax[0].fill_between(X_range, y_pred_range - margin, y_pred_range + margin,
                               color='red', alpha=0.2, label=f'{int(confidence_level*100)}% CI')

        ax[0].set_title("Fitted Regression Line")
        ax[0].set_xlabel("X")
        ax[0].set_ylabel("y")
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)

        # --- (2) Parameter evolution over epochs ---
        epochs = np.arange(len(self.w0_history))
        ax[1].plot(epochs, self.w0_history, label='w₀ (intercept)', color='green')
        ax[1].plot(epochs, self.w1_history, label='w₁ (slope)', color='orange')
        ax[1].set_title("Parameter Evolution")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Parameter value")
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)

        # --- (3) Loss curve ---
        ax[2].plot(self.loss_history, color='purple', linewidth=2)
        ax[2].set_title("Loss vs. Epochs")
        ax[2].set_xlabel("Epoch")
        ax[2].set_ylabel("MSE Loss")
        ax[2].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
class CauchyLoss(nn.Module):
    def __init__(self, c: float = 1.0):
        super().__init__()
        self.c = c
    def forward(self, y_pred, y_true):
        residual = (y_true - y_pred) / self.c
        loss = (self.c ** 2 / 2) * torch.log1p(residual ** 2)
        return torch.mean(loss)

class CauchyRegression:
    def __init__(self, n_features,
                 learning_rate=1e-3,
                 max_epochs=2000,
                 tolerance=1e-6,
                 c=1.0,
                 standardize=True,
                 weight_init='zeros',
                 grad_clip=1.0,
                 verbose=True):
        self.n_features = n_features
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.c = c
        self.standardize = standardize
        self.grad_clip = grad_clip
        self.verbose = verbose

        # parameters: weight vector (n_features,) and intercept (scalar)
        self.w = nn.Parameter(torch.zeros(n_features, dtype=torch.float32, requires_grad=True))
        self.b = nn.Parameter(torch.zeros(1, dtype=torch.float32, requires_grad=True))

        if weight_init == 'random':
            with torch.no_grad():
                self.w.uniform_(-0.01, 0.01)
                self.b.uniform_(-0.01, 0.01)

        self.criterion = CauchyLoss(c=self.c)
        self.optimizer = optim.SGD([self.w, self.b], lr=self.lr)

        # storage
        self.loss_history = []
        self.param_history = []
        self.fitted = False
        self.scaler_X = None
        self.y_mean = None
        self.y_std = None

    def forward(self, X):
        # X: (n_samples, n_features)
        return X @ self.w + self.b  # shape (n_samples,)

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Basic shape checks
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        n_samples, n_feat = X.shape
        assert n_feat == self.n_features, f"Expected {self.n_features} features, got {n_feat}"

        # Optionally standardize X and center/scale y for stability
        if self.standardize:
            self.scaler_X = StandardScaler().fit(X)
            Xs = self.scaler_X.transform(X).astype(np.float32)
            self.y_mean = float(y.mean())
            self.y_std = float(y.std()) + 1e-12
            ys = ((y - self.y_mean) / self.y_std).astype(np.float32)
        else:
            Xs = X
            ys = y

        X_tensor = torch.tensor(Xs, dtype=torch.float32)
        y_tensor = torch.tensor(ys, dtype=torch.float32)

        prev_loss = float('inf')
        self.loss_history = []
        self.param_history = []

        for epoch in range(self.max_epochs):
            self.optimizer.zero_grad()
            y_pred = self.forward(X_tensor).view(-1)
            loss = self.criterion(y_pred, y_tensor)
            loss.backward()
            # gradient clipping
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_([self.w, self.b], max_norm=self.grad_clip)
            self.optimizer.step()

            loss_val = float(loss.item())
            self.loss_history.append(loss_val)
            with torch.no_grad():
                # convert weights back to original scale for logging:
                if self.standardize:
                    w_orig = (self.w.detach().numpy() / (self.scaler_X.scale_ + 1e-12)) * self.y_std
                    b_orig = (self.b.detach().numpy() * self.y_std) + self.y_mean - np.dot(w_orig, self.scaler_X.mean_)
                else:
                    w_orig = self.w.detach().numpy().copy()
                    b_orig = self.b.detach().numpy().copy()
                self.param_history.append(np.concatenate([w_orig.ravel(), np.array(b_orig).ravel()]))

            if epoch % 100 == 0 or epoch == self.max_epochs - 1:
                if self.verbose:
                    print(f"Epoch {epoch+1}/{self.max_epochs}, loss={loss_val:.6f}")
                    # print small sample of preds in original scale
                    with torch.no_grad():
                        sample_pred = self.predict(X[:5])
                        print(" sample y:", y[:5])
                        print(" sample y_pred:", np.round(sample_pred, 4))

            if abs(prev_loss - loss_val) < self.tolerance:
                if self.verbose:
                    print(f"Converged after {epoch+1} epochs (delta loss < tolerance).")
                break

            prev_loss = loss_val

        self.fitted = True
        self.X_train = Xs
        self.y_train = ys
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if self.standardize and self.scaler_X is not None:
            Xs = self.scaler_X.transform(X).astype(np.float32)
        else:
            Xs = X
        Xt = torch.tensor(Xs, dtype=torch.float32)
        with torch.no_grad():
            y_pred_scaled = self.forward(Xt).numpy().ravel()
        if self.standardize and self.y_mean is not None:
            return y_pred_scaled * self.y_std + self.y_mean
        else:
            return y_pred_scaled

    def get_parameters(self):
        if not self.fitted:
            raise ValueError("Model must be fitted before getting parameters.")
        last = self.param_history[-1]
        y_pred = self.predict(X)
        residuals = y - y_pred
        return {
            **{f"w{i+1}": float(last[i]) for i in range(self.n_features)},
            "w0 (intercept)": float(last[self.n_features]), "Mean residual": float(residuals.mean()),
            "R² (approx)": float(1 - np.var(residuals) / np.var(y))
        }

    
    def analysis_plots(self, X: np.ndarray, y: np.ndarray, figsize=(12,10)):
        if not self.fitted:
            raise ValueError("Model must be fitted before plotting.")
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        y_pred = self.predict(X)
        residuals = y - y_pred

        fig, ax = plt.subplots(7, 1, figsize=figsize)
        ax = ax.ravel()

        # Actual vs Predicted
        ax[0].scatter(y, y_pred, alpha=0.6)
        ax[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        ax[0].set_title("Actual vs Predicted")
        ax[0].set_xlabel("Actual y")
        ax[0].set_ylabel("Predicted y")

        # Loss
        ax[1].plot(self.loss_history)
        ax[1].set_title("Cauchy Loss")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Loss")

        # Parameter evolution
        param_arr = np.array(self.param_history)
        labels = [f"w{i+1}" for i in range(self.n_features)] + ["w0 (intercept)"]
        for i in range(param_arr.shape[1]):
            ax[2].plot(param_arr[:, i], label=labels[i])
        ax[2].legend()
        ax[2].set_title("Parameter evolution")

        # residuals vs each feature
        for i in range(4):
          ax[3 + i].scatter(X[:, i], residuals, alpha=0.6)
          ax[3 + i].axhline(0, color='red', linestyle='--')
          ax[3 + i].set_title(f"Residuals vs X{i+1}")
          ax[3 + i].set_xlabel(f"X{i+1}")
          ax[3 + i].set_ylabel("Residuals")
          ax[3 + i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        return fig

class LogisticRegression:
    """
    Multiclass Logistic Regression using PyTorch.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        learning_rate: float = 0.01,
        max_epochs: int = 1000,
        use_class_weights: bool = True
    ):
        self.n_features = n_features
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.use_class_weights = use_class_weights

        # Model
        self.model = nn.Linear(n_features, n_classes)
        self.criterion = None  
        self.optimizer = None  
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        # Compute class weights for biased datasets
        if self.use_class_weights:
            classes = np.unique(y)
            weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
            class_weights = torch.tensor(weights, dtype=torch.float32)
            print(f"Using class weights: {weights}")
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer: SGD
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        for epoch in range(self.max_epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 100 == 0 or epoch == self.max_epochs - 1:
                with torch.no_grad():
                    preds = torch.argmax(outputs, dim=1)
                    acc = (preds == y_tensor).float().mean().item()
                print(f"Epoch [{epoch+1}/{self.max_epochs}] | Loss: {loss.item():.4f} | Accuracy: {acc:.4f}")

        self.fitted = True
        self.final_loss = loss.item()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model not fitted yet.")

        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(X_tensor)
            predictions = torch.argmax(logits, dim=1)
        return predictions.numpy()

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict:
        if not self.fitted:
            raise ValueError("Model not fitted yet.")

        y_pred = self.predict(X)
        cm = confusion_matrix(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')

        print("\nConfusion Matrix:\n", cm)
        print(f"\nF1 Score (weighted): {f1:.4f}")
        print(f"Final Loss: {self.final_loss:.4f}")

        # Compute TP, TN, FP, FN metric per class 
        metrics = {}
        for i in range(self.n_classes):
            TP = cm[i, i]
            FN = cm[i, :].sum() - TP
            FP = cm[:, i].sum() - TP
            TN = cm.sum() - (TP + FP + FN)
            metrics[i] = {"TP": TP, "TN": TN, "FP": FP, "FN": FN}

        print("\nPer-class Metrics:")
        for cls, m in metrics.items():
            print(f"Class {cls}: TP={m['TP']}, TN={m['TN']}, FP={m['FP']}, FN={m['FN']}")

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))

        return {"confusion_matrix": cm, "f1_score": f1, "metrics": metrics}

    def plot_roc_curves(self, X: np.ndarray, y_true: np.ndarray):
        if not self.fitted:
            raise ValueError("Model not fitted yet.")

        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1).numpy()

        y_true_bin = label_binarize(y_true, classes=list(range(self.n_classes)))

        plt.figure(figsize=(8, 6))
        for i in range(self.n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.title("ROC Curves")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
