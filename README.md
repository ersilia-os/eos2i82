# Pharmacokinetics with PKSmart

PKSmart contains a suite of ML models to predict clearance (CL), volume of distribution (VDss), fraction unbound in plasma (fup), mean residence time (MRT) and half-life (thalf) of small molecules in the human body. It has been built using a surrogate modelling approach. Models developed for  monkey, rat and dog were trained and then used to predict >1000 molecules with human PK parameters



## Information
### Identifiers
- **Ersilia Identifier:** `eos2i82`
- **Slug:** `pksmart`

### Domain
- **Task:** `Annotation`
- **Subtask:** `Property calculation or prediction`
- **Biomedical Area:** `Any`
- **Target Organism:** `Any`
- **Tags:** `ADME`

### Input
- **Input:** `Compound`
- **Input Dimension:** `1`

### Output
- **Output Dimension:** `14`
- **Output Consistency:** `Fixed`
- **Interpretation:** Pk measurements predictions for human, dog, monkey and rat

Below are the **Output Columns** of the model:
| Name | Type | Direction | Description |
|------|------|-----------|-------------|
| vdss_l_kg | float | high | Predicted value for human volume of distribution (L/kg) |
| cl_ml_min_kg | float | low | Predicted value for human clearance (mL/min/kg) |
| fup | float | high | Predicted value for human fraction unbound in plasma |
| mrt_hr | float | high | Predicted value for human mean residence time (min) |
| thalf_hr | float | high | Predicted value for human half-life (min) |
| dog_vdss_l_kg | float | high | Volume of distribution at steady state for dog (L/kg) |
| dog_cl_ml_min_kg | float | low | Clearance rate for dog (mL/min/kg) |
| dog_fup | float | high | Fraction unbound in plasma for dog |
| monkey_vdss_l_kg | float | high | Volume of distribution at steady state for monkey (L/kg) |
| monkey_cl_ml_min_kg | float | low | Clearance rate for monkey (mL/min/kg) |

_10 of 14 columns are shown_
### Source and Deployment
- **Source:** `Local`
- **Source Type:** `External`

### Resource Consumption


### References
- **Source Code**: [https://github.com/Manas02/pksmart-pip](https://github.com/Manas02/pksmart-pip)
- **Publication**: [https://link.springer.com/article/10.1186/s13321-025-01066-5#Sec19](https://link.springer.com/article/10.1186/s13321-025-01066-5#Sec19)
- **Publication Type:** `Peer reviewed`
- **Publication Year:** `2025`
- **Ersilia Contributor:** [GemmaTuron](https://github.com/GemmaTuron)

### License
This package is licensed under a [GPL-3.0](https://github.com/ersilia-os/ersilia/blob/master/LICENSE) license. The model contained within this package is licensed under a [None](LICENSE) license.

**Notice**: Ersilia grants access to models _as is_, directly from the original authors, please refer to the original code repository and/or publication if you use the model in your research.


## Use
To use this model locally, you need to have the [Ersilia CLI](https://github.com/ersilia-os/ersilia) installed.
The model can be **fetched** using the following command:
```bash
# fetch model from the Ersilia Model Hub
ersilia fetch eos2i82
```
Then, you can **serve**, **run** and **close** the model as follows:
```bash
# serve the model
ersilia serve eos2i82
# generate an example file
ersilia example -n 3 -f my_input.csv
# run the model
ersilia run -i my_input.csv -o my_output.csv
# close the model
ersilia close
```

## About Ersilia
The [Ersilia Open Source Initiative](https://ersilia.io) is a tech non-profit organization fueling sustainable research in the Global South.
Please [cite](https://github.com/ersilia-os/ersilia/blob/master/CITATION.cff) the Ersilia Model Hub if you've found this model to be useful. Always [let us know](https://github.com/ersilia-os/ersilia/issues) if you experience any issues while trying to run it.
If you want to contribute to our mission, consider [donating](https://www.ersilia.io/donate) to Ersilia!
