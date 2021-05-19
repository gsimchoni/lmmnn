# LMMNN

## Using Random Effects to Account for High-Cardinality Categorical Features and Repeated Measures in Deep Neural Networks

This is the working directory for our Neurips 2021 submission.

For full details see the paper and lmmnn_neurips2021_additional_material.pdf.

For running the simulations use the `simulate.py` file, like so:

```
python simulate.py --conf conf.yaml --out res.csv
```

The `--conf` attribute accepts a yaml file such as `conf.yaml` which you can change.

To run various real data experiments see the jupyter notebooks in the notebooks folder. We cannot unfortunately attach the actual datasets, see paper for details.

For using LMMNN with your own data use the `NLL` loss layer as shown in notebooks and simulation.
