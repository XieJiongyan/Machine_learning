## Machine_learning-liner_regression guide
线性回归模型：
$$
\bm y = \bm w ^T \bm x + b = \bm w^* \bm x^*
$$
这个文件夹用来实践线性回归，里面包含三个文件`lr_gradient_descent.py`, `lr_normal_equation.py`, `lr_sklearn.py`。这三个文件的共同点是：
- 都是线性回归模型
- 都处理了同样的数据 `input/USA_Housing.csv`。选取`csv`文件最后一列为标签，前面的列为特征.


这三个文件的不同点是：
1. `lr_gradient_descent.py`通过梯度下降法学习线性回归的参数$\bm w^*$
2. `lr_normal_equation.py`用公式直接给出了最优参数$\bm w^*$，方法是$\bm \mu = (\bm X^T \bm X)^{-1}\bm X^T \bm y$
3. `lr_sklearn.py`使用`sklearn`自带的库进行学习

### `lr_gradient_descent.py`的梯度下降法
$$
   \begin{aligned}
\frac{\partial}{\partial \theta_j} J(\theta) &= \frac{\partial}{\partial \theta_j} \frac{1}{2} (h_\theta(x) - y)^2\\
 &= 2 \cdot \frac{1}{2} (h_\theta(x) - y) \cdot \frac{\partial}{\partial \theta_j} (h_\theta(x) - y) \\
 &= (h_\theta(x) - y) \cdot \frac{\partial}{\partial \theta_j} (\sum_{i=0}^n \theta_i x_i - y) \\
 &= (h_\theta(x) - y) x_j
\end{aligned}
$$
