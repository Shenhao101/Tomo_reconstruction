# Tomo_reconstruction
This repository includes the codes for reconstructing a global tomography model from space domain to temporal domain, and for calculating subducted slab and carbon fluxes based reconstructed tomography model and plate motion model. 

---
* All datasets required to run these codes are availabl from the author (shenhao@mail.iggcas.ac.cn) on reasonable request.
* If you encounter any issues, have questions, or wish to contribute, please open a [GitHub Issue](https://github.com/Shenhao101/Tomo_reconstruction/issues) or start a [Discussion](https://github.com/Shenhao101/Tomo_reconstruction/discussions).

## Installation
We recommend using Conda to manage the Python enviroment.

### Create the environment
```bash
conda env create -f environment.yml
```
### Activate the environment

```bash
conda activate pygplates_py310
```
### Caveat

If creating the environment using “environment.yml” fails, we recommend manually creating a new Conda environment and installing gplately first, followed by the remaining packages listed in “environment.yml”.

For example:
```bash
conda create -n pygplates_py310 python=3.10
conda activate pygplates_py310
conda install -c conda-forge gplately
```
**Note:** The installation of gplately may be stuck at the *"Solving environment"* stage for several minutes (sometimes around 10 minutes) due to its complex dependencies. If the installation still cannot be completed, please try using **micromamba** instead. For details, please refer to the official GPlately installation guide (https://gplates.github.io/gplately/latest/sphinx/html/installation.html).

## Workflow
### 1. Calculate carbon volume density at subduction zones

```bash
python Calculate_Carbon_SubductionZone.py
```

This script converts the carbon area density dataset from Müller et al. (2022) into carbon volume density using the half-space cooling model and extracts carbon density distributions at subduction zones.

Input directory:
- `./Muller_etal_2019_Tectonics_v2.0_netCDF`
- `./Data_carbon_Muller2022`
- `./Muller_etal_2019_PlateMotionModel_v2.0_Tectonics_Updated`

Output directory: 
- `./Carbon_VolumeDensity_SubductionZone`

---

### 2. Time-domain reconstruction of tomography models

```bash
python Tomography_reconstruction.py
```

This script reconstructs the tomography model from the spatial domain into the geological time domain based on slab sinking rates.

Input directory: 
- `./Original_TomographyModel`
- `./Carbon_VolumeDensity_SubductionZone`

Output directory: 
- `./Reconstructed_TomographyModel`

---

### 3. Global subducted carbon flux calculation

```bash
python Calculate_subducted_Carbonflux.py
```

This script calculates the global carbon flux subducted into the mantle throughout geological history.

Output directory: Carbon_flux

---

### 4. Longitudinal distribution of subducted carbon flux

```bash
python Calculate_subducted_Carbonflux_longitude.py
```

This script calculates the longitudinal distribution of subducted carbon flux to investigate spatial variations in carbon recycling.

Output directory: Carbon_flux

### 5. Long-term tectonic carbon cycle model
```bash
python ./Linear_regression/regression.py
```
This script establishes a tectonic forcing model through least-squares fitting to proxy-based CO2 records.

## Reference
If you use these codes in your own research, please cite the following papers:
* Shen, H., Zhao, L., Guo, Z., Yuan, H., Yang, J., Wang, X., Guo, Z., Deng, C., Wu, F., 2023. Dynamic link between Neo-Tethyan subduction and atmospheric CO2 changes: insights from seismic tomography reconstruction. Science Bulletin 68, 637-644.
