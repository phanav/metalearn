Implementation of Reptile in PyTorch.

Meta learning can be understood as finding an efficient model training procedure.
For this purpose, the Reptile algorithm attempts to find good model parameter initilization.
A good parameter initialization here means the parameters can be easily adapted for many tasks, after a few update steps.
Reptile is a variant of fo-MAML (1st order Model Agnostic Meta Learning)

## Basic usage

Critical input variable are at the top of the Jupyter notebook or python script.
By default, expected input are images and a label file.
`datadir` contains an `images` folder with all images in flat structure, i.e. as direct children.
`datadir` also contains a metadata file, having at least 2 columns: filename and label.
The column names are specified by 2 variables: 
`filecolumn, labelcolumn`

`resultdir`: location to save result

`dataname`: result are saved in this folder inside the resultdir

`resultprefix`: prefix for output file
 
 `random_seed`: can be set to None

## References

Reptile:
https://arxiv.org/abs/1803.02999

MAML:
http://proceedings.mlr.press/v70/finn17a
