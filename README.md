# Paper code

_NOTE: The anonymized supplementary material does not contain model weights or training data, due to lack of space. We provide all the code used to run the experiments included in the paper and appendix, as well as instructions on how to reproduce our data from scratch.
Weights and data will be made public in a URL for the final version, allowing reproduction of all results together with the code available here._

Install all dependencies and run:
```
python generate_all_plots.py
```
to get all plots in the paper saved to `plots`. This will use saved data to generate all results.
To run all experiments from scratch, change the flag `load_data` to `False`. Doing so can take several hours to generate all plots.

## Hardware requirements

All experiments ran on a single node with 64 CPU cores and 125Gi RAM.

## Generating all results from scratch

Some experiments require importing or installing 3rd party data and software manually:
- Almost all experiments require agent training/inference data, and the trained agents. 
    These can be generated by OpenSpiel (Apache License 2.0) with the code available [here](https://github.com/OrenNeumann/AlphaZero-scaling-laws) (Apache License 2.0). Some of the model checkpoints used in this work were taken from the latter repository.
- Experiments calculating agent Elo require installing the [Bayeselo tool](https://www.remi-coulom.fr/Bayesian-Elo/) (GNU General Public License v3.0), and the Python API available [here](https://github.com/yytdfc/Bayesian-Elo) (GNU General Public License v3.0).
- Experiments using the Connect Four solver (`connect4_loss_plots` and `plot_policy_degradation`) require installing the solver available [here](https://github.com/PascalPons/connect4) (GNU Affero General Public License v3.0). The opening book, available as a release in the github repo, should be saved in the running directory.
- The alpha-beta pruning experiment in `connect4_loss_plots` requires that the solver opening book is NOT present in the 
    parent dir (forcing the solver to perform a full search without prior data).

