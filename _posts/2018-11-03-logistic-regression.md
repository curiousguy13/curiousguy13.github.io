---
layout: post
comments: false
title: "Math of Intelligence : Logistic Regression"
date: 2018-11-03
tags: math-of-intelligence, machine-learning, artificial-intelligence
---


# Math Of Intelligence : Logistic Regression

Here, we will be figuring out the math for a binary logistic classifier.

Logistic Regression is similar to Linear Regression but instead of a real valued output $$y$$, it will be either 0 or 1 since we need to classify into one of 2 categories.

In the linear regression post, we have defined our hypothesis function as:


$$
\begin{equation}
h_\theta(x) = \theta_0 + \theta_1x
\end{equation}
$$


Now, we can also have multiple input features i.e $$x_1, x_2, x_3...$$ and so on, so in that case our hypothesis function becomes:

$$
\begin{equation}
h_\theta(x) = \theta_0x_0 + \theta_1x_1 + \theta_2x_2 + \theta_1x_3 ....
\end{equation}
$$


We have added $$x_0=1$$ with $$\theta_0$$ for simplification. Now, the hypothesis function can be expressed as a combination of just 2 vectors: $$X=[x_0, x_1, x_2, x_3, ...]$$ and $$\theta = [\theta_0, \theta_1, \theta_2, ...]$$

$$
\begin{equation}
h_\theta(x) = \theta^TX
\end{equation}
$$

Still, the output of this function will be a real value, so we'll apply an activation function to convert the output to 0 or 1. We'll use the [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) $$g(z)$$ for this purpose. **TODO: Explore other activation functions**



\begin{equation}
g(z) = \frac{1}{1+e^{-z}}
\end{equation}

\begin{equation}
h(X) = g(\theta^TX) = \frac{1}{1+e^{-\theta^TX}}
\end{equation}



The most commonly used loss function for logistic regression is log-loss (or cross-entropy) **TODO: Why log-loss? Explore other loss functions.**

So, the loss function $$l(\theta)$$ for $$m$$ training examples is:

$$
\begin{equation}
l(\theta) = -\frac{1}{m}(\sum_{i=1}^m y^{(i)}log(h(x^{(i)}) + (1-y^{(i)})log(1-h(x^{(i)}))
\end{equation}
$$

which can also be represented as:

$$
\begin{equation}
l(\theta) = -(\sum_{i=1}^m y^{(i)}log(g(\theta^T x^{(i)})) + (1-y^{(i)})log(1-g(\theta^T x^{(i)}))
\end{equation}
$$




Now, similar to linear regression, we need to find out the value of $\theta$ that minimizes the loss. We can again use gradient descent for that. **TODO: Explore other methods to minimize the loss function.**

$$
\begin{equation}
\theta_j = \theta_j - \alpha \frac{\partial}{\partial \theta_j} l(\theta)
\end{equation}
$$

where $$\alpha$$ is the learning rate.

From (8), we get that we need to find out $$\frac{\partial}{\partial \theta_j} l(\theta)$$ to derive the gradient descent rule. Lets start by working with just one training example.

$$\frac{\partial}{\partial \theta_j} l(\theta)$$ can be broken down as follows:

$$
\begin{equation}
\frac{\partial}{\partial \theta} l(\theta) = \frac{\partial}{\partial h(x)}l(\theta).\frac{\partial}{\partial \theta}h(x)
\end{equation}
$$

$$
\begin{equation}
\frac{\partial}{\partial \theta} l(\theta) = \frac{\partial}{\partial g(\theta^T x)}l(\theta).\frac{\partial}{\partial \theta}g(\theta^Tx)
\end{equation}
$$

Calculating $$\frac{\partial}{\partial \theta}g(\theta^Tx)$$ first:

$$
\frac{\partial}{\partial \theta}g(\theta^Tx)  = \frac{\partial}{\partial \theta} \left(\frac{1}{1+e^{-\theta^T x}}\right)
$$

$$
= \frac{\partial}{\partial \theta}({1+e^{-\theta^T x}})^{-1}
$$

Using the chain rule of derivatives,

$$
=-({1+e^{-\theta^T x}})^{-2}.(e^{-\theta^T x}).(-x)
$$

$$
=\frac{e^{-\theta^T x}}{(1+e^{-\theta^T x})^2}.(x)
$$

$$
=\frac{1+e^{-\theta^T x}-1}{(1+e^{-\theta^T x})^2}.(x)
$$

$$
=\left(\frac{1+e^{-\theta^T x}}{(1+e^{-\theta^T x})^2}-\frac{1}{(1+e^{-\theta^T x})^2}\right).(x)
$$

$$
=\left(\frac{1}{(1+e^{-\theta^T x})}-\frac{1}{(1+e^{-\theta^T x})^2}\right).(x)
$$

$$
=(g(\theta^T x)-g(\theta^T x)^2).(x)
$$

$$
\begin{equation}
\frac{\partial}{\partial \theta}g(\theta^Tx) =g(\theta^T x)(1-g(\theta^T x).x
\end{equation}
$$

Now, calculating $$\frac{\partial}{\partial g(\theta^T x)}l(\theta)$$,

$$
\frac{\partial}{\partial g(\theta^T x)}l(\theta) = \frac{\partial}{\partial g(\theta^T x)}.(-(y.log(g(\theta^T x) + (1-y)log(1-g(\theta^T x)))
$$

Again, using the chain rule,

$$
= -\left(\frac{y}{g(\theta^T x)} + \frac{1-y}{1-g(\theta^T x)}.(-1)\right)
$$

$$
= -\left(\frac{y-y.g(\theta^T x)-g(\theta^T x)+y.g(\theta^T x)}{g(\theta^T x).(1-g(\theta^T x)}\right)
$$

$$
= -\left(\frac{y-g(\theta^T x)}{g(\theta^T x).(1-g(\theta^T x)}\right)
$$

$$
\begin{equation}
\frac{\partial}{\partial g(\theta^T x)}l(\theta) = -\left(\frac{y-g(\theta^T x)}{g(\theta^T x).(1-g(\theta^T x)}\right)
\end{equation}
$$

Finally, combining (10),(11),(12), we get

$$
\frac{\partial}{\partial \theta} l(\theta) = -\left(\frac{y-g(\theta^T x)}{g(\theta^T x).(1-g(\theta^T x)}\right).g(\theta^T x)(1-g(\theta^T x).x
$$

$$
\frac{\partial}{\partial \theta} l(\theta) = -(y-g(\theta^T x)).x
$$

$$
\begin{equation}
\frac{\partial}{\partial \theta} l(\theta) = -(y-h(x)).x
\end{equation}  
$$

Plugging this back in (8),

$$
\begin{equation}
\theta_j = \theta_j + \alpha(y-h(x)).x
\end{equation}
$$
