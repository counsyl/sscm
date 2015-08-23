sscm
=======

SSCM is the name for the genome-wide mutation score that Counsyl has been developing.

Getting Started
------------
First clone this repo, create a `virtualenv` for this repository, and jump into the `virtualenv`.
```
$ git clone git@github.com:counsyl/sscm.git
$ virtualenv venv
$ source venv/bin/activate
```

Then, install the requirements.
```
$ pip install -r requirements.txt
```

You're now ready to start training models!

Input Data and Features
------------
The input(s) to the training should be *tab-delimited files*, where each row represents a mutation and columns are features for that mutation. If a feature isn't present for the mutation, leave it as 'NA'. Typically, one TSV file will be for clustering (the simulated data, that is), and the other will be the known benign data.


The first step is to create a JSON file specifying exactly what features exist in your TSV files and how the algorithm should treat them. Let's call this `features.json`.
```
{
}
```

The first field to add is `columns`, which is just a list of the names you want to give each of the columns in your TSV file. The length of this list should be the same as the number of columns in your files.
```
{
    "columns": [ "verPhyloP", "verPhCons", ... ]
}
```

The names that you put in the `columns` attribute will be used in the next attribute you add to `features.json`, which is `features`.
`features` is a mapping between the *name* you give a feature that you are interested in using while training, and the following information:

* `"feature"`: the type of feature it is ("scalar", "vector")
* `"column"` or `"columns"`: the name of the column(s) in the file it corresponds to
* `"type"`: the type of values you will see for that feature("float", "string", or "int"),
* `"distribution"`: the distribution you want the model to assume that feature has

Feature types supported:

* `"scalar"`
* `"vector"`

Feature values supported:

* `"float"`
* `"string"`
* `"int"`

Feature distributions supported:

* `"Gaussian"`
* `"Multinomial"`
* `"MultivariateGaussian"` (for vector only)
*

An example `features.json` would look like this:
```
{
    "columns": [ "verPhyloP", "verPhCons", ... ],
    "features": {
        "verPhyloP": {
            "feature": "scalar",
            "column": "verPhyloP",
            "type": "float",
            "distribution": "Gaussian"
        },
        "Consequence": {
            "feature": "scalar",
            "column": "Consequence",
            "type": "str",
            "distribution": "Multinomial"
        },
        "Conservation": {
            "feature": "vector",
            "columns": ["GerpS", "verPhCons", "priPhCons"],
            "type": "float",
            "distribution": "MultivariateGaussian"
        }
    }
}
```
Every name in the `column` and `columns` property of each feature needs to correspond to name in the global `columns` attribute.

For a full blown `features.json` look at `sscm/data/features.json`.

Training
----------------------------
To train you want to use the `bin/train.py` script. 

### Basic usage
The script takes in the two TSV files (simulated and known benign) in the `--train` and the `--fit` arguments respectively. You'll also need
to specify the features you want this model to train on in the `--features` argument, which should be a list of keys from the `features` in your JSON file. Finally, you need to specify the name of the model you're training. Make sure `features.json` is in your current directory, or you can override the `--feature_file` argument to point to where it is.
 
Example usage:
```
$ bin/train.py first_model --train sim.raw --fit benign.raw --features verPhyloP verPhCons Consequence
```

After training, the model will be saved by default in the `models/` directory but this can also be overridden by specifying `--models_dir`.

Please look at the script's usage for a more complete description:
```
$ bin/train.py -h
```

Prediction
-----------------------------
To generate scores for mutations, use `bin/predict.py`. 

### Basic usage
This script takes in rows from a TSV file from stdin that should contain the exact same features as your training files. It'll add a column to the end that contains the particular model's score for that mutation and write the annotated file to stdout. All you need to do is enter the model's name and it'll find the model from the `models/` directory.

Example usage:
```
$ bin/predict.py first_model < test.raw > test-annotated.raw
```
Please look at it's usage for a more complete description.
```
$ bin/predict.py -h
```
