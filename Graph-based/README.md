
## Requirements

Use the following command:

```
    conda env create -f graphtsr.yml
```

You have to get the license of [Gurobi optimizer](https://www.gurobi.com/downloads/).

## Data

Document data used in paper are stored in data folder.


## Execution

### Train

inside codes folder,

```
    python code_training/main.py --mode train --batch_size 6 --experiment_name model --data_dir ../data/ctdar19_B2_m/train/ --NUM_STACKS 2 --num_epochs EPOCHS --gpu GPU
```

### Test

inside codes folder,

```
    python code_training/main.py --mode test --batch_size 1 --experiment_name bordered --test_data_dir ../data/ctdar19_B2_m/test/SCAN/img/ --test_name test --test_scan True
```

