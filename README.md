# Prediction of wing changes in the Lorenz63
Drawing inspiration from [E. Brugnago's paper](http://inaesp.org/PublicJG/brugnago_etal_predict_regime_change-durations_lorenz_CHAOS2020.pdf)
on regime changes in Lorenz's model, we trained an RNN to predict the number of time steps before the next wing change using the angles between the Lypunov covariant vectors. 

</br>

More precisely, if we denote $\theta_{ij}^{t}$ the angle between CLVs $i$ and $j$ at time $t$,
we trained the model to predict one of the following three categories using the matrix of angles between CLVs:

```math
\begin{equation}
    \begin{pmatrix}
        \theta_{12}^{t-199} & \cdots & \cdots & \theta_{12}^{t}\\
        \theta_{13}^{t-199} & \cdots & \cdots & \theta_{13}^{t}\\
        \theta_{23}^{t-199} & \cdots & \cdots & \theta_{23}^{t}
    \end{pmatrix}
    \longrightarrow \left\{
    \begin{array}{ll}
        0: \text{next change in less than 50 time steps} \\
        1: \text{next change between 50 and 200 time steps} \\
        2: \text{next change in more then 200 time steps} 
    \end{array}
\right.
\end{equation}
```

Most projects are developed in Python: the environment can then be copied using the following command

```console
conda env create --name ENVNAME --file environment.yml
```

For projects containing a demonstration, this can be launched with the following command

```console
bash run.sh
```
