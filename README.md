# Logistic regression algorithm
```python
import numpy as np
```

## sigmoid
$$g(z) = \frac{1}{1 + e^{-z}}$$

## cost funtion
$$J(f_{\mathbf{w},b}(\mathbf{x}^{(i)}),y^{(i)}) =
\frac{1}{m} \sum_{i=1}^{m}[loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}),y^{(i)})]$$

$$loss(f_{\mathbf{w},b}(x^{(i)}, y^{(i)}) = -y^{(i)}\log(g(z)) - (1-y^{(i)})\log(1 - g(z)) $$
> $\log$ is $\ln$
  

