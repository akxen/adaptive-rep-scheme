# Adaptive Recalibration Strategies for Refunded Emissions Payment Schemes
Code in this repository implements an agent-based model to investigate the design of adaptive recalibration strategies for Refunded Emissions Payment (REP) schemes. This model is implemented in the context of a power system, and uses data describing Australia's largest electricity transmission network, which are obtained from the following link: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1326942.svg)](https://doi.org/10.5281/zenodo.1326942).

Two approaches are employed to update scheme parameters. The first looks one period into the future and seeks to update an emissions intensity baseline such that any imbalance between current scheme revenue and some target is eliminated in the next period. The second employs a model predicitive control framework which forecasts several periods into the future, identifying a path of emissions intensity baselines that achieves the revenue target, with the objective of minimising changes to the baseline between successive intervals.


