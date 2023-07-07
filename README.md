# ProjectionPursuitAdaptation
Data-driven method to learn the optimal polynomial chaos expansion on the optimal low-dimensional space

To use the code, please cite the following paper:
Zeng, Xiaoshu, and Roger Ghanem. "Projection pursuit adaptation on polynomial chaos expansions." Computer Methods in Applied Mechanics and Engineering 405 (2023): 115845.

Some of the parameters that are specific to the problem at hand are as follows:

PCE order: The appropriate PCE order needs to be determined based on the application requirements and characteristics.

tol_pce: This parameter represents the acceptable tolerance for the PCE model within the PPA method. Varying the value of tol_pce can lead to different adapted dimensions.

PPA_dim: The pre-defined adapted dimension for the PPA method. If a value is specified for PPA_dim, the method will consistently search for the optimal adapted PCE with that specific dimensional value.
