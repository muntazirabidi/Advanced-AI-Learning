from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.decomposition import PCA

class GPWrapper:
  def __init__(self, n_components = 10):
    self.pca = PCA(n_components)
    self.gp() = GaussianProcessRegressor(
      kernel = RBF() + WhiteKernel(),
      normalize_y = True
    )
    
    def fit(self, x, y):
      X_redcued = self.pca.fit_transformed(X)
      self.gp.fit(X_reduced, y)
      
    def predict(self, X, return_std = True):
      X_reduced = self.pca.tramsform(X)
      return self.gp.predict(X_reduced, return_std = return_std)
    
  