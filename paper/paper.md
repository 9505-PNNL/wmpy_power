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
date: 1 May 2024
bibliography: paper.bib
---

# Summary

Hydropower is an important source of renewable energy in many parts of the world. The generation potential for a hydropower facility can vary greatly due to fluctuations in precipitation and snowmelt patterns impacting streamflow and reservoir storage. Human activities such as irrigation, manufacturing, and hydration can also influence water availability at nearby and downstream facilities. `wmpy-power`--the hydropower model described in this work--is process-based, leveraging explicit reservoir storage and release data to address impacts on hydropower from climate change and human adaptive behaviors to inform long-term planning and resource-adequacy considerations.

# Statement of need

`wmpy-power` (Water Management Python - Hydropower) is a Python implementation of the WMP algorithm [@Zhou2018] for hydropower simulation developed to support long-term electricity grid infrastructure planning and climate impacts studies. The model simulates hydropower production at the facility scale using a minimal set of physical characteristics for each facility, timeseries of daily streamflow and reservoir storage, and historical observations of monthly hydropower production. With this data, the model performs a two-step calibration process using the Shuffled Complex Evolution (SCE) algorithm [@Duan1993] to optimize a set of facility and regional efficiency and bias-correction factors. Once calibrated, the model can then be used to simulate regional and facility-scale hydropower production for arbitrary timeseries of streamflow and reservoir storage. See \autoref{fig:generation} for an example of `wmpy-power` modeled generation compared to observed generation at the facility and regional scale.

![Example model output of simulated hydropower at the regional and facility scales compared with example observations. The regional signal exhibits high fidelity despite the noise and missing data points in the certain facility signals. In this example, the calibration period was 1995-2006 and the simulation period was 2007-2010.\label{fig:generation}](figure1.png)

As a process-based model, `wmpy-power` utilizes time series of channel flow and reservoir storage to account for the non-stationarity of hydropower generation arising from uncertainties in hydrology and the non-linear effect of climate change on water management [@Zhou2018]. The model is designed to simulate an entire region of hydropower facilities in bulk where the details required to accurately simulate each facility are potentially incomplete, and accounts for biases in the input timeseries by calibrating against hydropower observations. Although it was designed for regional scale prediction with a focus on long-term infrastructure planning, it also demonstrates commendable accuracy at the facility scale despite a tradeoff in precision when compared to facility-specific models. 

@Turner2022 provides a review of the landscape of hydropower models used at large spatial scales. Physics-based models such as Hydrogenerate [@Mitra2021] require more specific details on turbine characteristics and plant design to achieve high accuracy, which are not always widely available. Statistical models such as WRES [@Kao2016] directly correlate runoff with hydropower generation but may overlook the complex interactions arising from human adaptive management of water availability and and hydropower production. `wmpy-power` fills this gap between the physics-based and statistical paradigms. 

# Ongoing research

Active research is underway utilizing `wmpy-power` as part of a one-way coupled modeling chain from hydrology to river routing to reservoir operations to hydropower. The `mosartwmpy` model [@Thurber 2021, @Voisin2013] routes runoff through a river network with detailed water management rules [@Turner2021], providing flow and storage information to `wmpy-power`. @Zhou2023 investigates the compounding effects of climate and model uncertainty in multi-model assessments of hydropower. @Kao2022 provides monthly hydropower projections for hydropower facilities in the United States under climate and model uncertainties as part of the USDOE Secure Water Act. @Broman2024 examines regional changes in projected hydropower availability across the United States for a high-population, high-warming socio-economic climate scenario. 

# Acknowledgements

This study was supported by the US Department of Energy (DOE) Water Power Technologies Office as a part of the SECURE Water Act Section 9505 Assessment. This paper was authored by employees of Pacific Northwest National Laboratory, managed by Battelle under contract DE-AC05-76RL01830 with the US DOE.

# References