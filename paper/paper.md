---
title: 'wmpy-power: A Python package for process-based regional hydropower simulation'
tags:
  - Python
  - hydropower
  - water management
authors:
  - name: Travis B. Thurber
    orcid: 0000-0002-4370-9971
    corresponding: true
    affiliation: 1
  - name: Daniel Broman
    orcid: 0000-0001-8281-3299
    affiliation: 1
  - name: Tian Zhou
    orcid: 0000-0003-1582-4005
    affiliation: 1
  - name: Nathalie Voisin
    orcid: 0000-0002-6848-449X
    affiliation: "1, 2"
affiliations:
  - name: Pacific Northwest National Laboratory, Richland, WA., USA
    index: 1
  - name: University of Washington, Seattle, WA., USA
    index: 2
date: 15 March 2024
bibliography: paper.bib
---

# Summary

Hydropower is an important source of energy in many parts of the world. The generation potential for a hydropower facility can vary greatly due to fluctuations in precipitation and snowmelt patterns impacting streamflow and reservoir storage. Human activities such as irrigation, manufacturing, and hydration can also influence water availability at nearby and downstream facilities. Climate change and human adaptive behaviors can thus make the forecasting of hydropower availability very challenging for long-term planning and resource-adequacy considerations.

# Statement of need

`wmpy-power` (Water Management Python - Hydropower) is a Python package for hydropower simulation developed to support long-term electricity grid infrastructure planning and climate impacts studies. The model simulates hydropower production at the facility scale using a minimal set of physical characteristics for each facility, timeseries of daily streamflow and reservoir storage, and historical observations of monthly hydropower production. With this data, the model performs a two-step calibration process using the Shuffled Complex Evolution (SCE) algorithm [@Duan1993] to optimize a set of facility and regional efficiency and bias-correction factors. Once calibrated, the model can then be used to forecast regional and facility-scale hydropower production for arbitrary timeseries of streamflow and reservoir storage. See \autoref{fig:generation} for an example of `wmpy-power` modeled generation compared to observed generation at the regional and facility scales.

`wmpy-power` accounts for the non-stationarity in hydropower generation due to changes in hydrology [citation?] and the non-linearity in the effect of climate change on water management [@Zhou2018]. The model is designed to simulate an entire region of hydropower facilities in bulk where the details required to accurately simulate each facility are potentially incomplete, and accounts for biases in the input timeseries by calibrating against hydropower observations. `wmpy-power` is unique as a hydropower simulation model in that it explicitly simulates individual facilities using a process-based approach with less of a data requirement than other process-based models such as [what other models??]. The tradeoff is a decrease in accuracy at the facility scale, but the model is suitable at the regional scale to support long-term infrastructure planning. Alternative approaches use statistical methods that relate runoff directly to hydropower generation, such as WRES [@Kao2016]. This type of approach risks missing the complex interactions that arise from human management of water availability and hydropower production [citation??].

![Example model output of simulated hydropower at the regional and facility scales compared with example observations. The regional signal exhibits high fidelity despite the noise and missing data points in the certain facility signals. In this example, the calibration period was 1995-2006 and the simulation period was 2007-2010.\label{fig:generation}](figure1.png)

# Ongoing research

Active research is underway utilizing `wmpy-power`. @Zhou2023 investigates the compounding effects of climate and model uncertainty in multi-model assessments of hydropower (where WMP is a previous working name for `wmpy-power`). @Broman2024 examines regional changes in forecasted hydropower availability across the United States for a high-population, high-warming socio-economic climate scenario. @Aremu2024 studies the potential for replacing gas generators with hydrogen batteries charged using seasonal excess hydropower generation in the Niger River Basin.

# Acknowledgements

This study was supported by the US Department of Energy (DOE) Water Power Technologies Office as a part of the SECURE Water Act Section 9505 Assessment. This paper was authored by employees of Pacific Northwest National Laboratory, managed by Battelle under contract DE-AC05-76RL01830 with the US DOE.

# References