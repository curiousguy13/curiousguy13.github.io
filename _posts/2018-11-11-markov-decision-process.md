---
layout: post
comments: false
title: "Math of Intelligence : Markov Decision Process"
date: 2018-11-11
tags: math-of-intelligence, reinforcement-learning, artificial-intelligence
---


# Math Of Intelligence : Markov Decision Process

### What is a Markov Decision Process?

A Markov Decision Process consists of 5 elements: S, A, P, R, $$\gamma$$

S $$ \rightarrow $$ set of states 



A $$\rightarrow$$ set of actions

R $$\rightarrow$$ reward function

P $$\rightarrow$$ transition probability function:  P(s',r | s,a)


$$\gamma$$ $$\rightarrow$$ discounting factor

The states of an MDP have a property that:

$$P[S_{t+1} \space | \space S_t]=P[S_{t+1}\space | \space S_1, S_2, .... S_t] $$

It means that the future depends on the current state and not on the history of all previous states.

### Bellman Equations

$$V(s)$$ is the state value function. It describes the expected return given the current state s and Q(s,a) is the action value function which describes the expected return given the current state s and the action a, that the agent takes from state s.

$$
\begin{equation}
V(s) = E[G_t \space | \space S_t=s]
\end{equation}
$$

Here, $$G_t$$ is the expected return at time t. That is the expected sum of rewards that we will get after time t. So, $$G_t$$ can be represented as $$R_{t+1} + \gamma R_{t+2} + \gamma ^2 R_{t+3} ...$$ where $$\gamma$$ is the [discount factor](https://en.wikipedia.org/wiki/Discounting).

Now, Eq. (1) becomes:

$$ V(s) = E[R_{t+1} + \gamma R_{t+2} + \gamma ^2 R_{t+3} ... \space | \space S_t=s]$$

$$  = E[R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+3} ...) \space | \space S_t=s]$$

$$  = E[R_{t+1} + \gamma (G_{t+1}) \space | \space S_t=s]$$

$$  
\begin{equation}
= E[R_{t+1} + \gamma (V(S_{t+1})) \space | \space S_t=s]
\end{equation}$$

Similarly, for Q-value,

$$
\begin{equation}
Q(s,a)=E[R_{t+1}+\gamma Q(S_{t+1}, A_{t+1})\space | \space S_t=s, A_t=a]
\end{equation}
$$

### Bellman Expectation Equations:

$$
\begin{equation}
V_\pi(s) = \sum_{a \in A}^{} \pi (a | s) Q_\pi(s,a)
\end{equation}
$$

$$
\begin{equation}
Q_\pi(s,a) = R(s,a) + \gamma \sum_{s' \in S}^{} P_{ss'}^{a}V_{\pi}(s')
\end{equation}
$$

$$
\begin{equation}
V_\pi(s) = \sum_{a \in A}^{} \pi (a | s)(R(s,a) + \gamma \sum_{s' \in S}^{} P_{ss'}^{a}V_{\pi}(s'))
\end{equation}
$$

$$
\begin{equation}
Q_\pi(s,a) = R(s,a) + \gamma \sum_{s' \in S}^{} P_{ss'}^{a} \sum_{a' \in A}^{} \pi (a' | s') Q_\pi(s',a')
\end{equation}
$$

### Bellman Optimality Equations
Lets find out the optimal values for state value and action value functions:

$$
\begin{equation}
V_{*}(s) = argmax_{a \in A} Q_{*}(s, a)
\end{equation}
$$

$$
\begin{equation}
Q_{*}(s,a) = R(s, a) + \gamma \sum_{s' \in S} P_{ss'}^{a}V_{*}(s')
\end{equation}
$$

$$
\begin{equation}
V_{*}(s) = argmax_{a \in A}(R(s, a) + \gamma \sum_{s' \in S} P_{ss'}^{a}V_{*}(s'))
\end{equation}
$$

$$
\begin{equation}
Q_{*}(s,a) = R(s, a) + \gamma \sum_{s' \in S} P_{ss'}^{a} argmax_{a' \in A} Q_{*}(s', a')
\end{equation}
$$
