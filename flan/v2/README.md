# Flan v2 training data (writing in progress)
This repository contains seqio tasks and mixtures used in Flan v2.

## List of mixtures
mixtures.py contains the following mixture names:
```
{flan,t0,cot,dialog,niv2}_{zsopt,zsnoopt,fsopt,fsnoopt}
```

## How to use
You can import mixtures.py and directly use the mixtures, or combine all the mixtures into a new mixture:
```
seqio.MixtureRegistry.add(
    'flan_zs_fs_opt',
    tasks=[
        ('flan_zsopt', 50),  # mixing weight = 50
        ('flan_fsopt', 50),  # mixing weight = 50
    ])
```
