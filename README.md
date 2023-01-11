# OCTAve: 2D en face Optical Coherence Tomography Angiography Vessel Segmentation in Weakly-Supervised Learning with Locality Augmentation

[![DOI](https://img.shields.io/badge/DOI-10.1109%2FTBME.2022.3232102-blue)](https://doi.org/10.1109/TBME.2022.3232102)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This is an `experimentation` branch -- an experimentation setting and implementation that we used to produce the result in the manuscript.

## Project Structure

- `architectures`:
This directory contains PyTorch model (`nn.module`) implementation of models used in this work.

- `evaluation`:
This directory contains metrics calculation function and helper class that is used for calculating various performance metrics.

- `experiments`:
This directory contains PyTorch Lightning's `LightningModule` implementation that dictate model training and experiments.

- `interfaces`:
This directory contains utility functions and classes that allowing the `experiments` module interact with datasets.

- `runsets`:
This directory dictates how we run the experiments like a parameter configuration 
and etc.

- `utils`:
This directory contains utility functions such as oveeriden SLURMExecutor class that was modified to be compatible with our cluster `sbatch` policy and etc.

---
## Usage

### Prepare dataset

- create a directory `data`

#### ROSE

- The dataset can be directly extract from zipfile and ready for use.

#### OCTA-500

- Extract OCTA-500's zipfile into `data` directory.
- For any `train_xxxx.py` script, you can execute `python train_xxxx.py run-octa500 --prepare-data --seed [SEED]`. This should only be done once.

### Training

Example of the training command can be found in `train_xxxx.sh`, but our experiment used these script for the sake of convenience.

```bash
# OCTAve(ResNeStUNet)
./train_octanetaag.sh
# OCTAve(UNet)
./train_scribble.sh
```
---

## Citation
```bibtex
@ARTICLE{9999313,
    author={Chinkamol, Amrest and Kanjaras, Vetit and Sawangjai, Phattarapong and Zhao, Yitian and Sudhawiyangkul, Thapanun and Chantrapornchai, Chantana and Guan, Cuntai and Wilaiprasitporn, Theerawit},
    journal={IEEE Transactions on Biomedical Engineering},
    title={OCTAve: 2D en face Optical Coherence Tomography Angiography Vessel Segmentation in Weakly-Supervised Learning with Locality Augmentation},
    year={2022},
    volume={},
    number={},
    pages={1-12},
    doi={10.1109/TBME.2022.3232102}
}
```