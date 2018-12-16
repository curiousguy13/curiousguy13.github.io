---
layout: post
comments: false
title: "Math of Intelligence : Linear Regression"
date: 2018-11-02
tags: math-of-intelligence, machine-learning, artificial-intelligence
---


# Math Of Intelligence : Linear Regression

Let $$x$$ be the input feature and $$y$$ is the output that we are interested in.

For linear regression, we need a hypothesis function that predicts y, given the input feature x.

Let us assume that y is linearly dependent on x, so our hypothesis function is:


$$
\begin{equation}
h_\theta(x) = \theta_0 + \theta_1x
\end{equation}
$$


Here $$\theta_i$$'s are the parameters(or weights). To simplify the notation, we will drop the $$\theta$$ in the subscript of $$h_\theta(x)$$ and mention it simply as $$h(x)$$.

Now, we need to find a way to measure the error between our predicted output $h(x)$ and the actual value y for all our training examples.

One way to measure this error is the [ordinary least squared](https://en.wikipedia.org/wiki/Ordinary_least_squares) method. **TODO: Explore other cost functions**

So, the cost function(or loss function)* $$J(\theta)$$ according to the ordinary least square method will be as follows:

*there's some debate about whether they are the same or not but for now we'll assume they are the same

$$
\begin{equation}
J(\theta) = \frac{1}{2}(h(x)-y)^2
\end{equation}
$$

On expanding $$h(x)$$, we get

$$
\begin{equation}
J(\theta) = \frac{1}{2}(\theta_0 + \theta_1 x-y)^2
\end{equation}
$$

Our objective is to find the values of $$\theta_0$$ and $$\theta_1$$ that minimize the loss function.

One way to do this is by using the [Gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) method. **TODO:Explore other methods to find the global minima of a function**

$$
\begin{equation}
\theta_j = \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)
\end{equation}
$$

In this method, we first initialize $$\theta_j$$ randomly and then update it according to the above rule to come closer the minima with each update.

Here, $$\alpha$$ is the learning rate.

Hence, in order to update $$\theta_j$$, we need to find out the partial derivative of $$J(\theta)$$ w.r.t. $$\theta_j$$. In our case j = 0 and 1

w.r.t. $$\theta_0$$

$$
\begin{equation}
\frac{\partial}{\partial \theta_0} =  2*\frac{1}{2}(\theta_0 + \theta_1 x-y).(1)
\end{equation}
$$

$$
\begin{equation}
\frac{\partial}{\partial \theta_0} =  \theta_0 + \theta_1 x-y
\end{equation}
$$

w.r.t. $$\theta_1$$

$$
\begin{equation}
\frac{\partial}{\partial \theta_1} =  2*\frac{1}{2}(\theta_0 + \theta_1 x-y).(x)
\end{equation}
$$

$$
\begin{equation}
\frac{\partial}{\partial \theta_1} =  (\theta_0 + \theta_1 x-y)x
\end{equation}
$$

Combining equations (4) and (6) as well as (4) and (8) we get:

$$
\begin{equation}
\theta_0 = \theta_0 - \alpha(\theta_0 + \theta_1 x-y)
\end{equation}
$$

$$
\begin{equation}
\theta_1 = \theta_1 - \alpha(\theta_0 + \theta_1 x-y)(x)
\end{equation}
$$

The above equations can be used to update the weights and hence improve the hypothesis function with every training example.

### References:
1. https://see.stanford.edu/materials/aimlcs229/cs229-notes1.pdf
