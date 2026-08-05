### Preliminary Notes
* JAX precision - float32 applicable for NN training
* use float64 for verifying closed form/ analytical solutions numerically - float32 has a floor of 1e-7, float64 has a floor of 1e-15. Also use for long sequential computations
* @dataclass is a python decorator that automatically generates special methods like __init__() and __repr__() for classes. It is used to create classes that primarily store data and provides a concise way to define them. - freeze makes it immutable after creation
* reparam - allows you to choose the weight factorisation function for the dense layer
* check whether reparam is called 

### Learned Params
* risk free rate is held constant - we have the rate data in the WTI dataset - check whether it is used downstream
* psi = learned params have two transforms applied to them - softplus and inverse soft plus
* softplus ensures learned params are positive
* softplus = log(1 + exp(x)) - smooth approximation to ReLU
* unconstrained raw nums -> softplus -> constrained learned params in valid range
* inverse softplus = log(exp(x) - 1) - used to initialise learned params in valid range to test initialising params above/below the true value



* psi learned params have two transforms applied to them - softplus and inverse soft plus 
* alpha_q derived analytically from alpha p now 
* learned params initialised 