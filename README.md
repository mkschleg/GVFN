# GVFN

This repo contains the code used for the majority of results in "General Value Function Networks". This was accepted to JAIR and is currently in the production phase.

## Authors

- [Matthew Schlegel](https://mkschleg.github.io)
- Andrew Jacobsen
- Zaheer Abbas
- Andrew Patterson
- [Adam White](https://sites.ualberta.ca/~amw8/)
- [Martha White](https://webdocs.cs.ualberta.ca/~whitem/)


## Abstract
State construction is important for learning in partially observable environments. A general purpose strategy for state construction is to learn the state update using a Recurrent Neural Network (RNN), which updates the internal state using the current internal state and the most recent observation. This internal state provides a summary of the observed sequence, to facilitate accurate predictions and decision-making. At the same time, specifying and training RNNs is notoriously tricky, particularly as the common strategy to approximate gradients back in time, called truncated Back-prop Through Time (BPTT), can be sensitive to the truncation window. Further, domain-expertise---which can usually help constrain the function class and so improve trainability---can be difficult to incorporate into complex recurrent units used within RNNs. In this work, we explore how to use multi-step predictions to constrain the RNN and incorporate prior knowledge. In particular, we revisit the idea of using predictions to construct state and ask: does constraining (parts of) the state to consist of predictions about the future improve RNN trainability? We formulate a novel RNN architecture, called a General Value Function Network (GVFN), where each internal state component corresponds to a prediction about the future represented as a value function. We first provide an objective for optimizing GVFNs, and derive several algorithms to optimize this objective. We then show that GVFNs are more robust to the truncation level, in many cases only requiring one-step gradient updates.


## Acknowledgments

We would like to thank the Alberta Machine Intelligence Institute, IVADO, NSERC and the Canada CIFAR AI Chairs Program for the funding for this research, as well as Compute Canada for the computing resources used for this work. We would also like to thank Marc Bellemare for helpful comments about extensions to infinite sets for the set of histories. 




