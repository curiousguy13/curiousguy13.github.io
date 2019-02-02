---
layout: post
comments: false
title: "Math of Intelligence : Dynamic Programming for Markov Decision Process"
date: 2018-11-14
tags: math-of-intelligence, reinforcement-learning, artificial-intelligence
---

# Math Of Intelligence : Dynamic Programming

For a random policy $$ \pi $$

$$
\begin{equation}
V_{\pi}(s) = E_{\pi} [r+ \gamma V_{\pi} | S_{t} = s]
= \sum_{a \in A} \pi (a|s) \sum_{s', r} P(s', r | s,a)(r+\gamma V_{\pi}(s'))
\end{equation}
$$

$$
\begin{equation}
Q_{\pi}(s,a) = E[R_{t+1} + \gamma V_{\pi}(S_{t+1} | S_{t} = s, A_{t} = a]
= \sum_{s', r} P(s', r | s,a)(r+\gamma V_{\pi}(s'))
\end{equation}
$$

Since this is a model based algorithm, we know the value of $$ $$ using which we can calculate first state value function $$V_{\pi}(s)$$ & then action value function $$ Q_{\pi}(s,a) \forall s \in S, a \in A$$ Then we can update our policy $$\pi$$ to be actions that maximize the state value of that state.

$$
\pi'(s) \leftarrow argmax a \in A(s) Q_{\pi}(s,a)
$$

Continue iterating for better policies.
