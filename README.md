# pyGLMHMM

## What Is It?
pyGLMHMM is a pure Python implementation of the GLM-HMM model of this [repository](https://github.com/murthylab/GLMHMM) implemented in MATLAB. It follows the general framework of a [scikit-learn estimator](https://scikit-learn.org/stable/developers/develop.html) while being faithful to the original implementation.

This GLM-HMM model has been developed in ([Calhoun et al., 2019](https://www.nature.com/articles/s41593-019-0533-x)) as a method to infer internal states of an animal based on sensory environment and produced behavior. This technique makes use of a regression method, Generalized Linear Models ([GLMs](https://en.wikipedia.org/wiki/Generalized_linear_model)), that identify a 'filter' that describes how a given sensory cue is integrated over time. Then, it combines it with a hidden state model, Hidden Markov Models ([HMMs](https://en.wikipedia.org/wiki/Hidden_Markov_model)), to identify whether the behavior of an animal can be explained by some underlying state. The end goal of this GLM-HMM model is to best predict the acoustic behaviors of the vinegar fly D. melanogaster. The GLM–HMM model allows each state to have an associated multinomial GLM to describe the mapping from feedback cues to the probability of emitting a particular type of song. Each state also has a multinomial GLM that produces a mapping from feedback cues to the transition probabilities from the current state to the next state. This allows the probabilities to change from moment to moment in a manner that depends on the sensory feedback that the fly receives and to determine which feedback cues affect the probabilities at each moment. This model was inspired by a previous work that modeled neural activity ([Escola et al., 2011](https://www.mitpressjournals.org/doi/abs/10.1162/NECO_a_00118)), but instead uses multinomial categorical outputs to account for the discrete nature of singing behavior.

## Getting Started
### Installation
`pip install pyGLMHMM`

### Instructions on using pyGLMHMM:
The main module is [`GLMHMM`](https://github.com/aslansd/pyGLMHMM/blob/master/src/pyGLMHMM/GLMHMM.py) which follows generally the [scikit-learn estimator](https://scikit-learn.org/stable/developers/develop.html) framework:

- First, an instance of `GLMHMMEstimator` class must be generated with its different parameters. The most important ones are:
  - `num_samples`: the number of distinct samples in the input data
  - `num_states`: the number of hidden internal states
  - `num_emissions`: the number of emitted behaviors or actions (like song types)
  - `num_feedbacks`: the number of sensory feedback cues
  - `num_filter_bins`: the number of bins to discretize the filters of sensory feedback cues
  - `num_steps`: the number of steps taken in the maximization step of the EM algorithm for calculating the emission matrix
  - `filter_offset`: the number of bias terms added to the sensory feedback cues

- Second, the `fit(X, y, [])` method of `GLMHMMEstimator` class must be run on the input data:
  - `X` (stim): A list with the length of `num_samples`. Each element of the list is a two-dimensional numpy array with the size of ((`num_feedbacks` * `num_filter_bins` + `filter_offset`) * (number of time points)). They represent the filtered sensory feedback cues across time and they match up with the corresponding elements of input `y`.
  - `y` (symb): A list with the length of `num_samples`. Each element of the list is a one-dimensional numpy array with the size of ((number of time points)). They represent the emitted bahaviors or actions (like song types) by integer numbers across time and they match up with the corresponding elements of input `X`.
  
- Third, the output of the `fit` method is a dictionary which has the emission and transition matrices of all EM iterations of the `fit` method and also some other attributes of `GLMHMMEstimator` class.

Here is a sample code:

```python
from GLMHMM import GLMHMMEstimator
estimator = GLMHMMEstimator(num_samples = 5, num_states = 2, num_emissions = 2, num_feedbacks = 3, num_filter_bins = 30, num_steps = 1, filter_offset = 1)
output = estimator.fit(X, y, [])
```

## The GLM-HMM Model
![Schematic illustrating the GLM–HMM](https://github.com/aslansd/pyGLMHMM/blob/master/fig/GLM-HMM.jpg)

This is a schematic figure (Fig. 1 (d) of [Calhoun et al., 2019](https://www.nature.com/articles/s41593-019-0533-x)) showing the GLM-HMM model for 17 sensory feedback cues, 4 song modes, and 3 internal states. In the paper, they showed that their model allows experimenters to identify, in an unsupervised manner, dynamically changing internal states that influence decision-making and, ultimately, behavior. Using this model, they found that during courtship, Drosophila males utilize three distinct sensorimotor strategies (the three states of the model). Each strategy corresponded to a different relationship between inputs (17 sensory feedback cues that affect male singing behavior) and outputs (three types of song and no song). While previous work had revealed that fly feedback cues predict song-patterning decisions, the discovery of distinct state-dependent sensorimotor strategies was only possible with the GLM–HMM. The GLM-HMM model shows that there is substantial overlap in the distribution of feedback cues that describe each state, and also there is not a simple one-to-one mapping between states and song outputs. In conclusion, in comparison to classical descriptions of behavior as fixed action patterns, even instinctive behaviors such as courtship displays are continuously modulated by feedback signals. Moreover, the relationship between feedback signals and behavior is not fixed, but varies continuously as animals switch between strategies. Instead, just as feedback signals vary over time, so too do the algorithms that convert these feedback cues into behavior outputs.

## Minimization Method for EM Algorithm
To find the emission and transition matrices of the GLM-HMM model, the [Expectation-Maximization (EM)](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm) algorithm is used. In the M step of this algorithm, the negative expected complete-data log-likelihood is minimized. To this end, the [LBFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS), a popular quasi-Newton method, is used. While [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) has a native implementation of LBFGS, its implementation is somewhat different from MATLAB LBFGS implementation in ['minFunc'](https://github.com/murthylab/GLMHMM/tree/master/matlab_code/minFunc) function and also has a lower performance. As a result, we decided to use the [PyTorch implementation of LBFGS](https://github.com/hjmshi/PyTorch-LBFGS). For this purpose, the objective function is converted to a Torch neural network module through a [wrapper function](https://github.com/aslansd/pyGLMHMM/blob/master/src/pyGLMHMM/minimizeLBFGS.py). Also, the following modifications were performed on the main [LBFGS module](https://github.com/aslansd/pyGLMHMM/blob/master/src/pyGLMHMM/LBFGS.py) to make it more similar to the MATLAB implementation of [line search methods](https://optimization.mccormick.northwestern.edu/index.php/Line_search_methods):
- The Armijo backtracking line search was directly translated from [here](https://github.com/murthylab/GLMHMM/blob/master/matlab_code/minFunc/ArmijoBacktrack.m).
- The strong Wolfe line search was directly translated from [here](https://github.com/murthylab/GLMHMM/blob/master/matlab_code/minFunc/WolfeLineSearch.m).

## To Do
### Implementation
- [ ] So far the code was tested and compared with the results of the MATLAB code considering the default options. However, it must be tested and compared with the results of the MATLAB code running with the non-default options too in near future.
- [ ] Since the code was translated from MATLAB, it is not totally [Pythonic](https://docs.python-guide.org/writing/style/), and this somewhat degrades its efficiency. So one major improvement would be re-writing the code in a more Pythonic way.
### Extension
- [ ] The framework presented here can be extended to include continuous internal states with state-dependent dynamics.
- [ ] In principle, states themselves may operate along multiple timescales that necessitate hierarchical models in which higher-order internal states modulate lower-order internal states. The method presented here can also be extended to include this feature.

## References
1. Calhoun, A. J., Pillow, J. W., & Murthy, M. (2019). Unsupervised identification of the internal states that shape natural behavior. Nature neuroscience, 22(12), 2040-2049.
2. Escola, S., Fontanini, A., Katz, D., & Paninski, L. (2011). Hidden Markov models for the stimulus-response relationships of multistate neural systems. Neural computation, 23(5), 1071-1132.
3. Schmidt, M. (2005). minFunc: Unconstrained Differentiable Multivariate Optimization in MATLAB. Software available [here](https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html).
4. Michael-Shi, H. J., & Mudigere, D. (2018). A PyTorch implementation of L-BFGS. Software available [here](https://github.com/hjmshi/PyTorch-LBFGS)
