# Paper code

Install all dependencies and run:
```
python generate_all_plots.py
```
to get all plots in the paper saved to `plots`. This will use saved data to generate all results.
To run all experiments from scratch, change the flag `load_data` to `False`. Doing so can take several hours to generate all plots.

## Hardware requirements

All experiments ran on 64 CPU cores with 125Gi RAM.

## Generating all results from scratch

NOTE: This anonymized repo does not contain model weights or training data, due to lack of space. Weights and data will be made public in the final version.

Some experiments require importing or installing 3rd party data and software manually:
- Almost all experiments require agent training/inference data, and the trained agents. 
    These can be generated with OpenSpiel by the code available here:
    https://github.com/OrenNeumann/AlphaZero-scaling-laws
- Experiments using the Connect Four solver (`connect4_loss_plots` and `plot_policy_degradation`) require installing the solver available here:
    https://github.com/PascalPons/connect4
    The opening book, available as a release in the github repo, should be saved in the running directory.
- The alpha-beta pruning experiment in `connect4_loss_plots` requires that the solver opening book is NOT present in the 
    parent dir (forcing the solver to perform a full search without prior data).

