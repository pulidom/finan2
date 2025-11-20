import torch
import torch.nn as nn

def NLL_loss(in_var, labels, eps=1e-6):
    ''' Negative log likelihood '''
    mu, sigma = in_var
    sigma = torch.clamp(sigma, min=eps)
    distribution = torch.distributions.normal.Normal(mu, sigma)
    return -torch.mean(distribution.log_prob(labels))

def eMSE_loss( in_var,labels,alpha=0.5):
    """
    Extended MSE using estimated variance
    """
    mu,sigma = in_var
    error = (mu-labels)**2
    return (1-alpha) * F.mse_loss(sigma, var)+alpha*F.mse_loss(mu, labels)

def NNmodel(layers,activation_fn,activation_output=None):
    """ 
    Generate an NN with linear transformations and activation functions 
    """
    # Parameter Initialization with .apply
    modules = []
    for i in range(len(layers)-2):
        modules.append(nn.Linear(layers[i], layers[i+1]))
        modules.append(activation_fn)
    #output layer
    modules.append(nn.Linear(layers[len(layers)-2], layers[len(layers)-1]))
    if activation_output is not None:
        modules.append(activation_output)

    NNmdl = nn.Sequential(*modules)
    # aca puedo inicializar la red
    #NNmdl.apply(init)
    
    return NNmdl

class NNModelMeanLogVar(nn.Module):
    """
    Red neuronal con dos salidas: media y log(varianza) para regresión
    """
    def __init__(self, layers, activation_fn, init_logvar_bias=-1.0):
        """
        Args:
            layers: lista con dimensiones [input_dim, hidden1, hidden2, ..., output_dim]
            activation_fn: función de activación (ej: nn.ReLU())
            init_logvar_bias: valor inicial del bias para log-varianza (default: -1.0)
        """
        super(NNModelMeanLogVar, self).__init__()
        
        # Capas compartidas (backbone)
        modules = []
        for i in range(len(layers)-2):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            modules.append(activation_fn)
        
        self.shared_layers = nn.Sequential(*modules)
        
        # Dimensión de la última capa oculta y dimensión de salida
        hidden_dim = layers[-2]
        output_dim = layers[-1]
        
        # Dos cabezas de salida separadas
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        self.logvar_head = nn.Linear(hidden_dim, output_dim)
        
        # Inicialización personalizada
        self._initialize_weights(init_logvar_bias)
    
    def _initialize_weights(self, init_logvar_bias):
        """Inicializa los pesos de las capas"""
        # Inicialización Xavier para todas las capas
        def init_xavier(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        
        self.shared_layers.apply(init_xavier)
        
        # Cabeza de media: inicialización estándar
        nn.init.xavier_uniform_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        
        # Cabeza de log-varianza: bias negativo para varianzas iniciales razonables
        nn.init.xavier_uniform_(self.logvar_head.weight)
        nn.init.constant_(self.logvar_head.bias, init_logvar_bias)
    
    def forward(self, x):
        """
        Forward pass
        Returns:
            mean: predicción de la media
            logvar: predicción del log de la varianza
        """
        # Capas compartidas
        features = self.shared_layers(x)
        
        # Dos salidas separadas
        mean = self.mean_head(features)
        logvar = self.logvar_head(features)
        
        return mean, logvar
    
    def predict_with_variance(self, x):
        """
        Retorna media y varianza (no log-varianza)
        """
        mean, logvar = self.forward(x)
        variance = torch.exp(logvar)
        return mean, variance
    
    def predict_with_std(self, x):
        """
        Retorna media y desviación estándar
        """
        mean, logvar = self.forward(x)
        std = torch.exp(0.5 * logvar)
        return mean, std


# Función de pérdida para entrenar (Negative Log-Likelihood Gaussiana)
def gaussian_nll_loss(mean, logvar, target):
    """
    Negative Log-Likelihood para distribución Gaussiana
    
    Args:
        mean: media predicha
        logvar: log-varianza predicha
        target: valores objetivo
    """
    # NLL = 0.5 * [log(2π) + log(var) + (y - μ)²/var]
    # Simplificado: 0.5 * [log(var) + (y - μ)²/var]
    loss = 0.5 * (logvar + ((target - mean) ** 2) / torch.exp(logvar))
    return loss.mean()


# Ejemplo de uso
if __name__ == "__main__":
    # Definir arquitectura: [input_dim, hidden1, hidden2, output_dim]
    layers = [10, 64, 32, 1]  # 10 inputs, 2 capas ocultas, 1 output
    
    # Crear modelo
    model = NNModelMeanLogVar(
        layers=layers,
        activation_fn=nn.ReLU(),
        init_logvar_bias=-1.0  # exp(-1) ≈ 0.37 varianza inicial
    )
    
    # Datos de ejemplo
    x = torch.randn(100, 10)  # 100 muestras, 10 features
    y = torch.randn(100, 1)   # 100 targets
    
    # Forward pass
    mean_pred, logvar_pred = model(x)
    print(f"Mean shape: {mean_pred.shape}")
    print(f"Log-var shape: {logvar_pred.shape}")
    
    # Calcular pérdida
    loss = gaussian_nll_loss(mean_pred, logvar_pred, y)
    print(f"Loss: {loss.item():.4f}")
    
    # Predicción con varianza
    mean, variance = model.predict_with_variance(x[:5])
    print(f"\nPrimeras 5 predicciones:")
    print(f"Media: {mean.squeeze()}")
    print(f"Varianza: {variance.squeeze()}")
    
