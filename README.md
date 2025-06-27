# aims_dgs4abm

## Installation
I highly recommend that you use [UV](https://docs.astral.sh/uv/):
```
cd aims_dgs4abm
uv venv
uv sync
```
If you use UV you won't need to activate the virtual environment before running scripts. Simply use ```uv run``` instead of ```python```.

Otherwise, you can use pip. First make sure that you're using Python >= 3.12. Next:
```
pip install -r requirements.txt
```



## ABMs

### Dataset Generation
```create_si_dataset.py``` and ```create_sir_dataset.py``` in ```abm/spatial_compartmental``` are used to generate datasets
from the SI and SIR ABMs respectively. SI datasets are generated via grid sampling over the parameter ranges whilst the SIR datasets are generated via Latin Hypercube Sampling over the parameter ranges.

For example:
```
python -m abm.spatial_compartmental.create_si_dataset --config-name SI_diag_grid_10_steps_10.yaml
```

SIR Image datasets are saved as many ```.npy``` files to avoid running out of GPU VRAM. They are made contiguous ```.npy``` files with ```fuse_all.py```
## Surrogate Models

### Training
```train.py``` is used to train the MCMLP and PriorCVAE surrogates and the final configurations for these surrogates are found in ```configs/train```.  The model's checkpoints will be saved inside the folder ```trained_models/<date>/<time>/```

For example:

```
python -m train.py --config-path configs/train/SIR/MCMLP --config-name 
```


and ```diffusion_train.py``` is used for the diffusion surrogate. It is configured via environment variables rather than Hydra.
To train the diffusion surrogate analysed in the project:

```
DATASET=data/NO_MOVE_SIR_vonn_image_grid_10_steps_20 NUM_ABM_PARAMS=4 MODEL_SIZE=2 NUM_COND=1 DATA_ON_HOST=1 N_EPOCHS=20 BATCH_SIZE=125 AUTOREG_STEPS=5 WARMUP_STEPS=125 python -m diffusion_train.py
```


## Parameter Inference
MCMC parameter inference can be performed in bulk for the SI ABM and surrogates is performed via ```si.py``` in ```inference/spatial_compartmental```. This will compute both posterior and posterior predictive samples. These samples and other metrics related to inference will be saved inside ```inference_ouputs/<date>/time/```. Example usage:
```
python -m inference.spatial_compartmental.si --config-name SI_diag_grid_10_steps_10
```
Parameter inference for the SIR ABM and surrogates is performed with ```sir.py``` in ```inference/spatial_compartmental```. Due to the massive number of MCMC samples, posterior predictive samples are obtained after inference with ```get_sir_predictive_samples``` which subsamples the MCMC samples. Both scripts use the same config file:
```
python -m inference.spatial_compartmental.sir --config-name full_cov_no_move_abm_grid_10_steps_30_A
python -m inference.spatial_compartmental.get_sir_predictive_samples --config-name full_cov_no_move_abm_grid_10_steps_30_A
```




## Reproducing figures from the project
The .ipynb notebooks in this repository are used to reproduce figures from the project. These notebooks require previously trained models and outputs of MCMC inference. This data can be downloaded here: [Gdrive Link](https://drive.google.com/file/d/1SUc2SjwoXHg36co9avRcIUxpdl_apnCw/view?usp=sharing) (Note that its ~20gb uncompressed). This is a zip file containing the folders ```inference_outputs```, ```trained_models``` and ```data``` which contains training data. These folders should be individually placed inside ```aims_dgs4abm```. 