# wmpy_power

`wmpy_power` is a hydropower generation simulation model that uses a physically-based representation of hydropower generation. The model parameterizes production using physically-meaningful parameters that are calibrated using a reference generation dataset.

#### To Do List
* Write automated tests
* Example batch jobs for parallel BA

### Documentation
#### Introduction
`wmpy_power` simulates hydropower generation using a physically-based representation
of hydropower generation ([Zhou et al., 2018](https://doi.org/10.1088/1748-9326/aad19f)).
Hydropower generation is simulated using timeseries of inflow and storage, and plant
characteristics including nameplate capacity, average head, and reservoir storage capacity (where applicable). Model parameters are calibrated to a reference monthly hydropower
generation dataset - typically historical generation - using the shuffle complex
evolution algorithm (SCE; [Duan et al., 1993](https://doi.org/10.1007/BF00939380)).
A two-stage calibration is performed: first at the balancing authority (BA) scale, and
second at the facility scale.

The model is designed to work with inflow and storage simulated by the mosartwmpy
routing and water management model ([Thurber et al., 2021](https://doi.org/10.21105/joss.03221)),
however is agnostic to the source of these data.

#### Calculations
`wmpy_power` uses a simple hydropower generation formula to calculate hydropower generation:

$$ P=\rho ghQ \eta \ (1) $$

|     Variable    |     Variable in Code                                   |     Definition                           |     Units     |     Value                         |
|-----------------|--------------------------------------------------------|------------------------------------------|---------------|-----------------------------------|
|     ρ           |     lumped in with gravitational acceleration; 9800    |     density of water                     |     kg m-3    |     1000                          |
|     g           |     lumped in with density of water; 9800              |     gravitational acceleration           |     m3s-2     |     9.81                          |
|     h           |     plant_head_m                                       |     hydraulic head of the dam            |     m         |     plant-specific                |
|     Q           |     flow                                               |     turbine flow rate                    |     m3s-1     |     plant-specific timeseries     |
|     η           |     not directly used; see below                       |     non-dimensional efficiency   term    |     –         |     plant-specific                |

This general formulation (equation 1) is modified for use in the model to accommodate parameters being calibrated, and to accommodate two cases of hydropower generation, run-of-river (ROR) plants, and plants associated with reservoirs. For both cases of generation, the non-dimensional efficiency term (η) is replaced with a combined efficiency and bias correction factor f<sub>b</sub>:

$$ P=\rho ghQf_b \ (2) $$

| Variable | Variable in Code | Definition                                                   | Units | Value                                                   | Range    |
|----------|------------------|--------------------------------------------------------------|-------|---------------------------------------------------------|----------|
| fb       | efficiency_spill | non-dimensional efficiency term and bias correction factor | –     | balancing authority-specific; calibrated in step one  | 0.5-1.5  |


This efficiency term is calibrated at the BA level in step one of the calibration. NOTE: alternative groupings of plants can be used in place of BAs; the BA variable is used by the code, but values can be replaced with other grouping identifiers, for example HUC4 basins.

Q is adjusted to account for both plant-specific maximum flow limits and spill. Maximum flow limits are imposed by limiting Q to a maximum value using Q<sub>max</sub> where:

$$ Q_max =S_f \ (3) $$ \
$$ Q =min(Q, Q_max) \ (4) $$

|     Variable    |     Variable in Code                   |     Definition                        |     Units     |     Value                                       |     Range      |
|-----------------|----------------------------------------|---------------------------------------|---------------|-------------------------------------------------|----------------|
|     Qmax        |     max_discharge                      |     density of water                  |     kg m-3    |     1000                                        |                |
|     S           |     nameplate_capacity_MW              |     nameplate capacity                |     W         |     plant-specific; from PLEXOS                 |                |
|     fm          |     efficiency_penstock_flexibility    |     penstock   intake scale factor    |     –         |     plant-specific; calibrated   in step two    |     0.5-1.5    |
|     g           |                                        |     gravitational acceleration        |     m3s-2     |     9.81                                        |                |
|     h           |     head                               |     hydraulic head of the dam         |     m         |     plant-specific                              |                |

Q is adjusted for spill by month using plant-specific monthly spill correction factors developed in step two of the calibration. These spill correction factors are applied as:

$$ Q_sc,m =Q_m(1 -f_f,m); m = {1,2, ..., 12} \ (5) $$

| Variable | Variable in Code     | Definition                                  | Units | Value                                    | Range   |
|----------|----------------------|---------------------------------------------|-------|------------------------------------------|---------|
| fs,m     | monthly_spill        | monthly spill correction factors            | –     | plant-specific; calibrated in step   two | 0-1     |
| fp       | penstock_flexibility | penstock flexibility of handling   max flow | –     | plant-specific; calibrated in step   two | 0.5-1.5 |

##### Run-of-River (ROR) Facilities
Generation for ROR plants is calculated using the hydropower generation formula and setting the head (h) to a fixed value equal to the dam height.

##### Reservoir Facilities
Generation for hydropower plants with reservoirs is calculated using the hydropower generation formula, with the head estimated using simulated volumetric storage, total volumetric capacity, and assuming that the shape of the reservoir is a tetrahedron:

$$ h=H^3\sqrt{\frac{v}{v_max}} \ (6) $$

| Variable | Variable in Code    | Definition                    | Units | Value                      |
|----------|---------------------|-------------------------------|-------|----------------------------|
| h        | height              | hydraulic head of the dam     | m     | plant-specific             |
| H        | plant_head_m        | dam height                    | m     | plant-specific             |
| V        | storage             | reservoir storage             | m3    | plant-specific timeseries  |
| Vmax     | storage_capacity_m3 | reservoir volumetric capacity | m3    | plant-specific             |

##### Shuffled Complex Evolution (SCE) Implementation
SCE is used to implement a two-step multiscale calibration that produces the inputs required for the hydropower generation formula (equation 2). Step one of the calibration is to address the errors in annual hydro-meteorological biases at the scale of hydrologic regions. The objective function used in step one is to minimizes the mean absolute error between annual observed potential generation and annual simulated potential generation at the BA level:

$$ PG_{BA,sim} =  \sum_{i=1}^n \rho gh_i Q_i f_{b,i} \ (6) $$
$$ PG_{BA,obs} =  TL_{2010} \times 0.04 + \sum_{i=1}^n G_{obs,i} \ (7) $$
$$ MAE_{PG} =  \sum_{i=1}^n \lvert PG_{BA_{sim,i}} - PG_{BA_{obs,i}} \rvert \ (8) $$

Potential generation is computed as:

$$ PG = G + R \ (10) $$
$$ R = 0.04TL \ (11) $$

|     Variable    |     Definition              |     Units    |     Value                                            |
|-----------------|-----------------------------|--------------|------------------------------------------------------|
|     PG          |     potential generation    |     MWh      |     computed at the BA scale                         |
|     G           |     actual generation       |     MWh      |     input at the plant scale                         |
|     R           |     operating reserve       |     MWh      |     calculated as 4% of the TL                       |
|     TL          |     total load              |     MWh      |     mean of annual generation   of years provided    |

The operating reserve percentage is set to as default to 4% of total load in model.py (operating_reserve_percent). This can be changed in model configuration.

Step two of the calibration seeks to reflect the complexity in monthly multi-objective reservoir operations and daily generation constraints. It works to improve seasonal variation for each power plant by calibrating a penstock flexibility factor (fp) and monthly spill correction factors (fs,1, fs,2,…, fs,12). The objective function used in step two is to minimize the Kling-Gupta Efficiency between simulated monthly power generation and observed monthly power generation at the plant level:

$$ KGE=1-\sqrt{(r-1)^2 + \left(\frac{sd(G_{sim})}{sd(G_{obs})}\right)^2 + \left(\frac{mean(G_{sim})}{mean(G_{obs})}\right)^2}\ (12) $$
$$ r = cor(G_{sim}, G_{obs}) \ (13) $$


|     Variable    |     Definition                      |     Units    |     Value                          |
|-----------------|-------------------------------------|--------------|------------------------------------|
|     Gsim        |     simulated monthly generation    |     MW       |     computed at the plant scale    |
|     Gobs        |     observed monthly generation     |     MW       |     input at the plant scale       |

#### Model Setup
The model repository contains documentation and example code for setting up and running the model. It is built as a Python package and the scripts can be imported into a Python environment.

##### Model Inputs
Model inputs are specified in the run script.

| Input                                    |                                        |       Description                                                                                        |     Format      |
|------------------------------------------|----------------------------------------|----------------------------------------------------------------------------------------------------------|-----------------|
|     daily simulated flow and storage     |     simulated_flow_and_storage_glob    |     daily simulated flow and storage, typically from a MOSART   simulation, for each hydropower plant    |     .parquet    |
|     observed monthly power generation    |     observed_hydropower_glob           |     observed monthly hydropower generation for each hydropower   plant                                   |     .parquet    |
|     reservoir and plant parameters       |     reservoir_parameter_glob           |     reservoir and hydropower plant static parameters                                                     |     .parquet    |



Many configuration options are available for running the model. These options can be specified in a configuration yaml
file or passed directly to the model initialization method, or both (the latter takes precedence). Most options have
sensible defaults, but a few are required as discussed below. For the full list of configuration options, see the
`config.yaml` file in the repository root.

Dependencies are listed in the `setup.cfg` file.
To install dependencies, run `pip install -e .` from the root directory of this repository.

There are two ways to run the model. The simplest is to provide a configuration file and run the model directly:

config.yaml
```yaml
# first year of calibration data
calibration_start_year: 1984
# last year of calibration data
calibration_end_year: 2008
# balancing authority or list of balancing authorities to calibrate
balancing_authority:
  - WAPA
# glob to files with simulated daily flow and storage for plants in the balancing authorities
simulated_flow_and_storage_glob: ./inputs/**/*flow*storage*.parquet
# glob to files with observed monthly hydropower for plants in the balancing authorities
observed_hydropower_glob: ./inputs/**/*monthly*obs*.parquet
# glob to files with reservoir/plant parameters for plants in the balancing authorities
reservoir_parameter_glob: ./inputs/**/*PLEXOS*.parquet
```

```commandline
python wmpy_power/model.py config.yaml
```

Alternatively, the model can be invoked from a python script or console:
```python
from wmpy_power import Model

# you can initialize the model using the configuration file:
model = Model('config.yaml')
# or directly:
model = Model(
  calibration_start_year=1984,
  calibration_end_year=2008,
  balancing_authority=['WAPA'],
  simulated_flow_and_storage_glob='./inputs/**/*daily*flow*storage*.parquet',
  observed_hydropower_glob='./inputs/**/*monthly*obs*.parquet',
  reservoir_parameter_glob='./inputs/**/*PLEXOS*.parquet',
)
# or both (keyword arguments take precedence)
model = Model(
  'config.yaml',
  balancing_authority=['CAISO'],
  output_type='csv',
)

# run the model (this will write the calibrations to file but also return a DataFrame
calibrations = model.run()

# plot each plant's modeled hydropower versus observed hydropower
model.plot(calibrations)

# get modeled generation for the calibrations and write to file
generation = Model.get_generation(
  './**/*_plant_calibrations.csv',
  './inputs/*reservoir_parameter*.parquet',
  './inputs/**/*daily*flow*storage*.parquet',
)
```

By default, the model writes the calibrated parameters to a parquet file per balancing  in the current working
directory, but this behavior can be overridden using the `output_path` and `output_type` configuration options.
So far only "csv" and "parquet" formats are supported.

#### Input Files

Daily flow and storage parquet files are expected to have these columns:

column       | example     | units   |
-------------|-------------|---------|
date         | 1980-01-01  | date    |
eia_plant_id | 153         | integer |
flow         | 461.003906  | m^3 / s |
storage      | 1.126790e10 | m^3     |

Monthly observed hydropower parquet files are expected to have these columns:

column         | example   | units   |
---------------|-----------|---------|
year           | 1980      | integer |
month          | 1         | integer |
eia_plant_id   | 153       | integer |
generation_MWh | 38221.193 | MWh     |

Reservoir/plant parameter parquet files are expected to have these columns:

column                | example | units   |
----------------------|---------|---------|
eia_plant_id          | 153     | integer |
balancing_authority   | WAPA    | string  |
name                  | 1980    | integer |
nameplate_capacity_MW | 1       | MW      |
plant_head_m          | 5123.3  | m       |
storage_capacity_m3   | 1.5e10  | m^3     |
use_run_of_river      | True    | boolean |


#### Notebook

The `tutorial.ipynb` file provides a Jupyter notebook illustration of running the model and plotting results.

#### Working with Parquet files

It's easy to read and write parquet files from pandas, just `pip install pyarrow` or `conda install pyarrow`, then:
```python
import pandas as pd
df = pd.read_parquet('/path/to/parquet/file.parquet')
df['my_new_column'] = 42
df.to_parquet('/path/to/new/parquet/file.parquet')
```
#### Legacy Files
wmpy_power was originally developed in MATLAB. The model provides a utility to convert Excel and MATLAB files to parquet files. This functionality can also be used to build inputs in Excel and convert them into parquet.

```python
from wmpy_power import Model

Model.update_legacy_input_files(
  daily_flow_storage_path='/path/to/flow/storage/matlab/file.mat',
  reservoir_parameters_path='/path/to/plexos/parameters/excel/file.xlsx',
  monthly_observed_generation_path='/path/to/observed/generation/excel/file.xlsx',
  daily_start_date='1980-01-01', # for daily flow/storage, since this isn't mentioned in the file itself, have to assume a start date
  monthly_start_date='1980-01-01', # for monthly observed hydro, since this isn't mentioned in the file itself, have to assume a start month
  output_path=None, # defaults to the same place as the input files, but with the .parquet extension
  includes_leap_days=False, # whether or not the daily data includes entries for leap days (Tian's files don't)
)
```
