# Data

The full course dataset is not included in this repository.

Expected raw file:

```text
data/data.txt
```

The committed `sample.txt` file is only for format inspection and smoke tests.

Generate processed public-state data with:

```bash
DATA_DIR=data_public python preprocess.py
```

The generated `.npz` files and `count.json` are ignored by git.
