# LMMNN

## Integrating Random Effects in Deep Neural Networks

This is the working directory for our [JMLR 2023](https://jmlr.org/papers/v24/22-0501.html) and [NeurIPS 2021](https://papers.nips.cc/paper/2021/hash/d35b05a832e2bb91f110d54e34e2da79-Abstract.html) papaers.

For full implementation details see the papers and supplemental material.

For running the simulations use the `simulate.py` file, like so:

```
python simulate.py --conf conf_files/conf_random_intercepts.yaml --out res.csv
```

The `--conf` attribute accepts a yaml file such as `conf_random_intercepts.yaml` which you can change.

To run various real data experiments see the jupyter notebooks in the notebooks folder. We cannot unfortunately attach the actual datasets, see papers for details.

For using LMMNN with your own data use the `NLL` loss layer as shown in notebooks and simulation.
