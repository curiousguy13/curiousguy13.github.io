---
layout: post
comments: false
title: "Math of Intelligence : Temporal Difference Learning"
date: 2019-03-09
tags: math-of-intelligence, reinforcement-learning, artificial-intelligence
---

# Math Of Intelligence : Temporal Difference Learning

#### Monte Carlo:

$$
\begin{equation}
V(S_{t}) \leftarrow V(S_{t}) + \alpha (G_t - V(S_{t+1}))
\end{equation}
$$


Monte Carlo methods wait until the end of episode to update the state value function $$V(S_t)$$ for a state $$S_t$$ where $$G_t$$ is the actual return at time t.

#### TD Learning:
Also, 
$$
\begin{equation}
v_\pi(s) = E_\pi[G_t | S_t = s] \\
               = E_\pi[R_{t+1} + \gamma G_{t+1} | S_t = s]
\end{equation}
$$

So, eq. (1) becomes 

$$
\begin{equation}
V(S_t) \leftarrow V(S_t) + \alpha[R_{t+1} + \gamma V(S_{t+1})-V(S_t)]
\end{equation}
$$ 

With this equation, $$V(S_t)$$ can be updated as soon as $$R_{t+1}$$ is received. This is called TD(0) or one-step TD Learning.



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

In TD method, the value in brackets is called TD-error $$(\delta_{t})$$
$$
\delta_t = G_t - V(S_t)  \\
         = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)
$$

TD method convergence proof? - TODO
Which one converges faster? TD or MC? How do we formalize this question? - TODO
Similarly for action value function:
$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha(R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t))
$$

#### SARSA : 
On policy TD Control
The next action is picked based on current policy + $$\epsilon$$ - greedy.
$$Q(s,a)$$ is learned from actions taken from the current policy $$\pi$$

#### Q-Learning: 
Off policy TD control
We pick next action based on max Q-values + $$\epsilon$$ - greedy.
$$Q(s,a)$$ value does not depend on policy $$\pi$$

So,
$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha(R_{t+1} + \gamma max_a Q(S_{t+1},a) - Q(S_t, A_t))
$$

### References:
Richard S. Sutton, Andrew G. Barto  - Reinforcement Learning: An Introduction 
