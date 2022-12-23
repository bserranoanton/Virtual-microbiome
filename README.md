# The virtual microbiome: a computational framework to evaluate microbiome analyses

This repository contains the code needed to generate realistic abundance matrices for microbiome simulation. 

The algorithm (ecologyGeneration.py) takes as input the relevant parameters of the problem (i.e. the number of populations and species and the shape parameters of the distributions, mu, sigma, shape and scale) and outputs an abundance matrix that complies with the macroecological laws followed by real-world microbial populations (as defined in [1]). It should be noted that abundance matrices are generated at family, genus and species level.

The use of ecologyGeneration.py is fairly straightforward. Just set the desired parameters from the main function (or change the parameters of the ecologyGeneration_Family class) to generate the matrices, which are saved as text files in the same folder where the code is executed. The rows represent each community and the columns represent the species. In this way, the value of the abundance matrix [i,j] indicates the abundance of species j in community i.

In addition, in the folder GAIA_configuration, you can see the parameter settings used for the generation of the scenarios described in CITA PAPER

[1] Grilli, J. Macroecological laws describe variation and diversity in microbial communities. Nature Communications 11 (2020).
