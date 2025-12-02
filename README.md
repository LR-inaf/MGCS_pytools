# Python tools used for the Missing Globular Cluster Survey [D. Massari et al. 2024](https://doi.org/10.1051/0004-6361/202554007)

## Installation
You can install this module from PyPi repository using `pip install mgcs-pytools`

## Brief description
This package provides three modules:
1. `mcmc`: this module provide a inference-based Guassian Mixture Model (GMM). this module has been designed to study the distribution of the Proper Motions but it can be used for any 2D distribution.
2. `statistical_membership`: you can use this module to perform a statistical decontamination of your CMD. This approach implements an adaptive CMD grid based on the Voronoi tassellation, and it combines the statistical decontamination with the differential reddening correction. the output is your photometric catalog with the membership probability for each star, the corrected magnitude and the delta_ebv to construct the reddening map
3. `utils`: this package contain usefull plotting routin for both outcomes of the `mcmc` and `statistical_memberhsip` modules.

Further details on the code and it performance on real data will be published in an upcoming paper(s).

You can find a boilerplate in the main folder.

## Contributing
__!!Contributions are super welcome!!__
If you wanna develop this project just clone this repo, make your branch with your branch and start developing.

## Issue reporting
Also reporting bugs is important. If you find some bugs please open a ticket in the issue tab and I will more than happy to try to fix it.

For any information please reach me at _luca.rosignoli@inaf.it_