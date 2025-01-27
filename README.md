# NDMA library

The goal of this library is twofold: on one hand, we want to propose a sleak, efficient and effective numerical library for the computation of dynamically interesting elements in system sof interest in biological applications, in particular looking at ODEs of the form

$$\dot x = - \gamma x + \prod \sum \sigma(x;p)$$

where we are particularly interested in the evolution of the phase space w.r.t. to the parameter given by $p$.
On the other hand, we also want to offer a smooth integration with DSGRN (https://github.com/marciogameiro/DSGRN) - a library that allows the cimputation of dynamical sygnatures on switching systems, that is ODEs such as (1) where the non-linearity $\sigma$ is a step function (additional infomation in: https://www.frontiersin.org/articles/10.3389/fphys.2018.00549/full and many others since).

The central class of this library is thus the Model class, where the ODE can be instantiated. 

Functionalities offered by this library are:
- interplay between NDMA and DSGRN mainly at the parameter level
- equilibrium detection
- saddle node detection
- integration
- ...

For additional information, please refer to our paper : Kepley, Shane, Konstantin Mischaikow, and Elena Queirolo. "Global analysis of regulatory network dynamics: equilibria and saddle-node bifurcations." arXiv preprint arXiv:2204.13739 (2022).

Small test codes are available in the \test folder, additonal tutorials are a work in progress.
