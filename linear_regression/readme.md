## linear regression 
1. data:
`../input/USA_Housing.csv`
选取csv文件最后一列为标签，前面的列为特征
2. `lr_normal_equation.py`
mean idea:$\mu = (X^T X)^{-1}X^Ty$
3. `lr_sklearn.py` 使用`sklearn`自带的库进行线性回归,使用类`sklearn.inear_model.LinearRegression`
4. `lr_gradient_descent.py`梯度下降法
$$
   \begin{aligned}
\frac{\partial}{\partial \theta_j} J(\theta) &= \frac{\partial}{\partial \theta_j} \frac{1}{2} (h_\theta(x) - y)^2\\
 &= 2 \cdot \frac{1}{2} (h_\theta(x) - y) \cdot \frac{\partial}{\partial \theta_j} (h_\theta(x) - y) \\
 &= (h_\theta(x) - y) \cdot \frac{\partial}{\partial \theta_j} (\sum_{i=0}^n \theta_i x_i - y) \\
 &= (h_\theta(x) - y) x_j
\end{aligned}
$$
