---
layout: post
comments: false
title: "Math of Intelligence : Temporal Difference Learning"
date: 2018-11-16
tags: math-of-intelligence, reinforcement-learning, artificial-intelligence
---

# Math Of Intelligence : Temporal Difference Learning

Monte Carlo prediction must wait until the end of an episode to update the value function. Temporal Difference methods update after every time step.

$$
\begin{equation}
V(S_{t}) = R_{t+1} + \gamma V(S_{t+1})
\end{equation}
$$

$$
\begin{equation}
V(S_t) \leftarrow V(S_t) + \alpha[G_t - V(S_t)]
\end{equation}
$$

$$
\begin{equation}
V(S_t) \leftarrow V(S_t) + \alpha[R_{t+1} + \gamma V(S_{t+1})-V(S_t)]
\end{equation}
$$

After taking an action at time t, we know the value of $$ R_{t+1} $$ as the reward received, so we can update the state value of the state $$ S_t $$ based on the above equation.

This is called $$ TD(0) $$ or one step TD. It is called so because it is a special case of $$ TD(\lambda) $$ where $$ \lambda=0 $$

Similarly for action value function:

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha(R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t))
$$

In TD method, the value in brackets is called TD-error $$(\delta_{t})
\delta_t = G_t - V(S_t)
\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)
$$

SARSA : On policy TD Control

The next action is picked based on current policy + $$\epsilon$$ - greedy

Q-Learning: Off policy TD control
We pick next action based on max Q-values + $$\epsilon$$ - greedy
