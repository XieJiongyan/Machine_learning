# Machine_learning
Excellent starter machine learning projects. 精彩的机器学习项目集。

***全面，细致，容易上手的机器学习项目集，真是赞不绝口***  
***What a masterpiece! --- XieJiongyan***

## table of Contents（目录）
*下述的每一个子项目对应着一个文件夹，具体内容可在文件夹中寻找并运行*

1. `linear_regression`线性回归模型
2. `logistic_regression`逻辑回归模型
3. `svm`svm分类器
4. `backpropagation`反向传播算法


## liner_regression
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

## logistic_regression
这个文件夹中包含了逻辑回归的两个`.py`文件：`lr_gradient_descent.py`和`lr_sklearn.py`. 他们的共同点是：  
- 都使用`input/logisticR_data.csv`作为数据输入

他们的区别是：
1. `lr_gradient_descent.py`使用**梯度下降法**进行逻辑回归
2. `lr_sklearn.py`使用`scikit-learn`自带的库进行逻辑回归

## Machine_learning-svm guide
本文件夹中包含如下四个文件：`svm_dataplot.py`, `svm_linear.py`, `svm_scikitlearn.py`, `svm_spiraldata.py`。其中，`svm_linear.py`, `svm_scikitlearn.py`, `svm_spiraldata.py`三个文件是使用**svm方法**进行二分类。`svm_dataplot.py`则是在打印了数据集。

### `svm_linear.py`
首先，调用`simple_synthetic_data`函数，生成了可以线性分类的数据集。  
然后使用线性分类核函数`default_ker(x, z)`: $rev = \bm x^T \bm z$, 在`svm_smo`分类器下进行分类

### `svm_spiraldata.py`
这里使用的数据集是`input/spiral.txt`中的螺旋形状的数据集，显然，不适合使用线性分类器。  
![avatar](fig/Figure_1.png)  
采用两种核函数`default_ker(x, z)`和`rbf_ker(x, z)`，使用`svm_smo`分类器下进行分类。  
- 核函数`default_ker(x, z)`: $rev = \bm x^T \bm z$
- 核函数`rbf_ker(x, z)`: $rev = \exp (- (\bm x - \bm z) ^T (\bm x - \bm z)) / (2 \sigma ^2)$

### `svm_scikitlearn.py`
这里使用的数据集是`input/spiral.txt`中的螺旋形状的数据集。并且采用`scikit-learn`中的`sklearn.svm.SVC`进行分类
## 集成学习 Ensemble learning
### 项目结构：
- `ensemble_learning/el_datagenerate.py`是数据生成的代码。
- `ensemble_learing/el_reandomforest.py`是依据`sklearn.tree.DecisionTreeClassifier`封装的随机森林。
- -`ensemble_learing/el_sklearn.py`是`sklearn.ensemble.BaggingClassifier`中的**bagging算法**实现