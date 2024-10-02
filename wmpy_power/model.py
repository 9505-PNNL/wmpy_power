from datetime import datetime
from dateutil.parser import parse
import duckdb
from glob import glob
import logging
import matplotlib.pyplot as plt
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import numpy as np
from os import linesep
import pandas as pd
from pathlib import Path
from psutil import cpu_count
from scipy.io import loadmat
import sys
from typing import Iterable, Tuple, Union

from .sce_ua import shuffled_complex_evolution_algorithm
from .utilities import timing

from yaml import load

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import warnings

class Model:

    config_defaults = dict(
        output_path=Path("."),
        output_type="parquet",
        operating_reserve_percent=0.04,
        lower_bound_efficiency=0.9,
        upper_bound_efficiency=2.0,
        lower_bound_percentile_factor=0.2,
        upper_bound_percentile_factor=0.8,
        lower_bound_spill=0.0,
        upper_bound_spill=1.0,
        lower_bound_penstock=0.5,
        upper_bound_penstock=1.5,
        efficiency_penstock_flexibility=1.1,
        efficiency_spill=0.0,
        efficiency_number_of_complexes=3,
        efficiency_maximum_trials=10000,
        efficiency_maximum_evolution_loops=4,
        efficiency_minimum_change_criteria=2.0,
        efficiency_minimum_geometric_range=0.001,
        efficiency_include_initial_point=False,
        efficiency_alpha=1.0,
        efficiency_beta=0.5,
        generation_number_of_complexes=3,
        generation_maximum_trials=5000,
        generation_maximum_evolution_loops=4,
        generation_minimum_change_criteria=1.0,
        generation_minimum_geometric_range=0.00001,
        generation_include_initial_point=False,
        generation_alpha=1.0,
        generation_beta=0.5,
        seed=None,
        log_to_file=True,
        log_to_stdout=True,
        parallel_tasks=cpu_count(logical=False),
    )

    def __init__(
        self,
        configuration_file: Union[str, Path] = None,
        *,
        output_path: Union[str, Path] = None,
        output_type: str = None,
        calibration_start_year: int = None,
        calibration_end_year: int = None,
        balancing_authority: Union[str, Iterable[str]] = None,
        simulated_flow_and_storage_glob: str = None,
        observed_hydropower_glob: str = None,
        reservoir_parameter_glob: str = None,
        operating_reserve_percent: float = None,
        lower_bound_efficiency: float = None,
        upper_bound_efficiency: float = None,
        lower_bound_percentile_factor: float = None,
        upper_bound_percentile_factor: float = None,
        lower_bound_spill: float = None,
        upper_bound_spill: float = None,
        lower_bound_penstock: float = None,
        upper_bound_penstock: float = None,
        efficiency_penstock_flexibility: float = None,
        efficiency_spill: float = None,
        efficiency_number_of_complexes: int = None,
        efficiency_maximum_trials: int = None,
        efficiency_maximum_evolution_loops: int = None,
        efficiency_minimum_change_criteria: float = None,
        efficiency_minimum_geometric_range: float = None,
        efficiency_include_initial_point: bool = None,
        efficiency_alpha: float = None,
        efficiency_beta: float = None,
        generation_number_of_complexes: int = None,
        generation_maximum_trials: int = None,
        generation_maximum_evolution_loops: int = None,
        generation_minimum_change_criteria: float = None,
        generation_minimum_geometric_range: float = None,
        generation_include_initial_point: bool = None,
        generation_alpha: float = None,
        generation_beta: float = None,
        seed: int = None,
        log_to_file: bool = None,
        log_to_stdout: bool = None,
        parallel_tasks: int = None,
    ):

        config = {}
        if configuration_file is not None:
            with open(configuration_file, "r") as f:
                config = load(f, Loader=Loader)
                config = {**self.config_defaults, **config}

        self.output_path = output_path or config.get("output_path", None)
        if self.output_path is None:
            self.output_path = self.config_defaults.get("output_path")

        self.output_type = output_type or config.get("output_type", None)
        if self.output_type not in ["parquet", "csv"]:
            self.output_type = self.config_defaults.get("output_type")

        self.calibration_start_year = calibration_start_year or config.get(
            "calibration_start_year", None
        )
        if self.calibration_start_year is None:
            raise ValueError("calibration_start_year is required")

        self.calibration_end_year = calibration_end_year or config.get(
            "calibration_end_year", None
        )
        if self.calibration_end_year is None:
            raise ValueError("calibration_end_year is required")

        self.balancing_authority = balancing_authority or config.get(
            "balancing_authority", None
        )
        if self.balancing_authority is None:
            raise ValueError("balancing_authority is required")

        self.simulated_flow_and_storage_glob = (
            simulated_flow_and_storage_glob
            or config.get("simulated_flow_and_storage_glob", None)
        )
        if self.simulated_flow_and_storage_glob is None:
            raise ValueError("simulated_flow_and_storage_glob is required")

        self.observed_hydropower_glob = observed_hydropower_glob or config.get(
            "observed_hydropower_glob", None
        )
        if self.observed_hydropower_glob is None:
            raise ValueError("observed_hydropower_glob is required")

        self.reservoir_parameter_glob = reservoir_parameter_glob or config.get(
            "reservoir_parameter_glob", None
        )
        if self.reservoir_parameter_glob is None:
            raise ValueError("reservoir_parameter_glob is required")

        self.operating_reserve_percent = operating_reserve_percent or config.get(
            "operating_reserve_percent", None
        )
        if self.operating_reserve_percent is None:
            self.operating_reserve_percent = self.config_defaults.get(
                "operating_reserve_percent"
            )

        self.lower_bound_efficiency = lower_bound_efficiency or config.get(
            "lower_bound_efficiency", None
        )
        if self.lower_bound_efficiency is None:
            self.lower_bound_efficiency = self.config_defaults.get(
                "lower_bound_efficiency"
            )

        self.upper_bound_efficiency = upper_bound_efficiency or config.get(
            "upper_bound_efficiency",
            None,
        )
        if self.upper_bound_efficiency is None:
            self.upper_bound_efficiency = self.config_defaults.get(
                "upper_bound_efficiency"
            )

        self.lower_bound_percentile_factor = (
            lower_bound_percentile_factor
            or config.get("lower_bound_percentile_factor", None)
        )
        if self.lower_bound_percentile_factor is None:
            self.lower_bound_percentile_factor = self.config_defaults.get(
                "lower_bound_percentile_factor"
            )

        self.upper_bound_percentile_factor = (
            upper_bound_percentile_factor
            or config.get("upper_bound_percentile_factor", None)
        )
        if self.upper_bound_percentile_factor is None:
            self.upper_bound_percentile_factor = self.config_defaults.get(
                "upper_bound_percentile_factor"
            )

        self.lower_bound_spill = lower_bound_spill or config.get(
            "lower_bound_spill", None
        )
        if self.lower_bound_spill is None:
            self.lower_bound_spill = self.config_defaults.get("lower_bound_spill")

        self.upper_bound_spill = upper_bound_spill or config.get(
            "upper_bound_spill", None
        )
        if self.upper_bound_spill is None:
            self.upper_bound_spill = self.config_defaults.get("upper_bound_spill")

        self.lower_bound_penstock = lower_bound_penstock or config.get(
            "lower_bound_penstock", None
        )
        if self.lower_bound_penstock is None:
            self.lower_bound_penstock = self.config_defaults.get("lower_bound_penstock")

        self.upper_bound_penstock = upper_bound_penstock or config.get(
            "upper_bound_penstock", None
        )
        if self.upper_bound_penstock is None:
            self.upper_bound_penstock = self.config_defaults.get("upper_bound_penstock")

        self.efficiency_penstock_flexibility = (
            efficiency_penstock_flexibility
            or config.get("efficiency_penstock_flexibility", None)
        )
        if self.efficiency_penstock_flexibility is None:
            self.efficiency_penstock_flexibility = self.config_defaults.get(
                "efficiency_penstock_flexibility"
            )

        self.efficiency_spill = efficiency_spill or config.get("efficiency_spill", None)
        if self.efficiency_spill is None:
            self.efficiency_spill = self.config_defaults.get("efficiency_spill")

        self.efficiency_number_of_complexes = (
            efficiency_number_of_complexes
            or config.get("efficiency_number_of_complexes", None)
        )
        if self.efficiency_number_of_complexes is None:
            self.efficiency_number_of_complexes = self.config_defaults.get(
                "efficiency_number_of_complexes"
            )

        self.efficiency_maximum_trials = efficiency_maximum_trials or config.get(
            "efficiency_maximum_trials", None
        )
        if self.efficiency_maximum_trials is None:
            self.efficiency_maximum_trials = self.config_defaults.get(
                "efficiency_maximum_trials"
            )

        self.efficiency_maximum_evolution_loops = (
            efficiency_maximum_evolution_loops
            or config.get("efficiency_maximum_evolution_loops", None)
        )
        if self.efficiency_maximum_evolution_loops is None:
            self.efficiency_maximum_evolution_loops = self.config_defaults.get(
                "efficiency_maximum_evolution_loops"
            )

        self.efficiency_minimum_change_criteria = (
            efficiency_minimum_change_criteria
            or config.get("efficiency_minimum_change_criteria", None)
        )
        if self.efficiency_minimum_change_criteria is None:
            self.efficiency_minimum_change_criteria = self.config_defaults.get(
                "efficiency_minimum_change_criteria", None
            )

        self.efficiency_minimum_geometric_range = (
            efficiency_minimum_geometric_range
            or config.get("efficiency_minimum_geometric_range", None)
        )
        if self.efficiency_minimum_geometric_range is None:
            self.efficiency_minimum_geometric_range = self.config_defaults.get(
                "efficiency_minimum_geometric_range"
            )

        self.efficiency_include_initial_point = (
            efficiency_include_initial_point
            if isinstance(efficiency_include_initial_point, bool)
            else config.get("efficiency_include_initial_point", None)
        )
        if self.efficiency_include_initial_point is None:
            self.efficiency_include_initial_point = self.config_defaults.get(
                "efficiency_include_initial_point"
            )

        self.efficiency_alpha = efficiency_alpha or config.get("efficiency_alpha", None)
        if self.efficiency_alpha is None:
            self.efficiency_alpha = self.config_defaults.get("efficiency_alpha")

        self.efficiency_beta = efficiency_beta or config.get("efficiency_beta", None)
        if self.efficiency_beta is None:
            self.efficiency_beta = self.config_defaults.get("efficiency_beta")

        self.generation_number_of_complexes = (
            generation_number_of_complexes
            or config.get("generation_number_of_complexes", None)
        )
        if self.generation_number_of_complexes is None:
            self.generation_number_of_complexes = self.config_defaults.get(
                "generation_number_of_complexes"
            )

        self.generation_maximum_trials = generation_maximum_trials or config.get(
            "generation_maximum_trials", None
        )
        if self.generation_maximum_trials is None:
            self.generation_maximum_trials = self.config_defaults.get(
                "generation_maximum_trials"
            )

        self.generation_maximum_evolution_loops = (
            generation_maximum_evolution_loops
            or config.get("generation_maximum_evolution_loops", None)
        )
        if self.generation_maximum_evolution_loops is None:
            self.generation_maximum_evolution_loops = self.config_defaults.get(
                "generation_maximum_evolution_loops"
            )

        self.generation_minimum_change_criteria = (
            generation_minimum_change_criteria
            or config.get("generation_minimum_change_criteria", None)
        )
        if self.generation_minimum_change_criteria is None:
            self.generation_minimum_change_criteria = self.config_defaults.get(
                "generation_minimum_change_criteria"
            )

        self.generation_minimum_geometric_range = (
            generation_minimum_geometric_range
            or config.get("generation_minimum_geometric_range", None)
        )
        if self.generation_minimum_geometric_range is None:
            self.generation_minimum_geometric_range = self.config_defaults.get(
                "generation_minimum_geometric_range"
            )

        self.generation_include_initial_point = (
            generation_include_initial_point
            if isinstance(generation_include_initial_point, bool)
            else config.get("generation_include_initial_point", None)
        )
        if self.generation_include_initial_point is None:
            self.generation_include_initial_point = self.config_defaults.get(
                "generation_include_initial_point"
            )

        self.generation_alpha = generation_alpha or config.get("generation_alpha", None)
        if self.generation_alpha is None:
            self.generation_alpha = self.config_defaults.get("generation_alpha")

        self.generation_beta = generation_beta or config.get("generation_beta", None)
        if self.generation_beta is None:
            self.generation_beta = self.config_defaults.get("generation_beta")

        # seeded random number generator instance
        self.random_number_generator = np.random.default_rng(
            seed or config.get("seed", None)
        )

        # parallel tasks
        self.parallel_tasks = parallel_tasks or config.get("parallel_tasks", None)
        if self.parallel_tasks is None:
            self.parallel_tasks = self.config_defaults.get("parallel_tasks")

        # logger
        self.logger = logging.getLogger("wmpy_power")
        for h in self.logger.handlers:
            self.logger.removeHandler(h)
        self.logger.addHandler(logging.NullHandler())
        formatter = logging.Formatter(
            "%(asctime)s - wmpy_power: %(message)s", datefmt="%Y-%m-%d %I:%M:%S %p"
        )
        if (
            log_to_file
            if isinstance(log_to_file, bool)
            else config.get("log_to_file", self.config_defaults.get("log_to_file"))
        ):
            now = datetime.now()
            h = logging.FileHandler(
                f"./wmpy_power_{now.strftime('%Y-%m-%d_%I_%M_%S')}.log"
            )
            h.setFormatter(formatter)
            self.logger.addHandler(h)
        if (
            log_to_stdout
            if isinstance(log_to_stdout, bool)
            else config.get("log_to_stdout", self.config_defaults.get("log_to_stdout"))
        ):
            h = logging.StreamHandler(sys.stdout)
            h.setFormatter(formatter)
            self.logger.addHandler(h)
        self.logger.setLevel(logging.INFO)

    @staticmethod
    def _get_reservoir_parameters_for_balancing_authority(
        balancing_authority: str,
        reservoir_parameter_glob: str,
    ) -> pd.DataFrame:
        q = duckdb.query(
            f"""
            SELECT *
            from '{reservoir_parameter_glob}'
            WHERE balancing_authority = '{balancing_authority}'
        """
        )
        return q.df()

    def get_reservoir_parameters_for_balancing_authority(
        self, balancing_authority: str
    ) -> pd.DataFrame:
        """
        Queries a parquet database for reservoir/plant parameters related to the requested balancing authority
        :param balancing_authority: the balancing authority acronym
        :return: a DataFrame containing the reservoir/plant parameters within the balancing authority
        """
        return self._get_reservoir_parameters_for_balancing_authority(
            balancing_authority,
            self.reservoir_parameter_glob,
        )

    @staticmethod
    def _get_flow_storage_for_plant_ids(
        plant_ids: Iterable[int],
        simulated_flow_and_storage_glob: str,
        start_year: int = None,
        end_year: int = None,
    ) -> pd.DataFrame:
        start_clause = (
            f"and date_part('year', date) >= {start_year}"
            if start_year is not None
            else ""
        )
        end_clause = (
            f"and date_part('year', date) <= {end_year}" if end_year is not None else ""
        )
        q = duckdb.query(
            f"""
            SELECT *
            from '{simulated_flow_and_storage_glob}'
            WHERE eia_plant_id in ({', '.join([f"{plant_id}" for plant_id in plant_ids])})
            {start_clause}
            {end_clause}
        """
        )
        return q.df()

    def get_flow_storage_for_plant_ids(
        self, plant_ids: Iterable[int], start_year: int = None, end_year: int = None
    ) -> pd.DataFrame:
        """
        Queries a parquet database for simulated flow and storage for a given list of plant IDs
        :param plant_ids: a list of plant IDs
        :param start_year: the beginning year for which to query data; None for all
        :param end_year: the end year for which to query data; None for all
        :return: a DataFrame containing the simulated flow and storage values for the requested plants
        """
        return self._get_flow_storage_for_plant_ids(
            plant_ids,
            self.simulated_flow_and_storage_glob,
            start_year,
            end_year,
        )

    @staticmethod
    def _get_hydropower_for_plant_ids(
        plant_ids: Iterable[int],
        observed_hydropower_glob: str,
        start_year: int = None,
        end_year: int = None,
    ) -> pd.DataFrame:
        start_clause = f"and year >= {start_year}" if start_year is not None else ""
        end_clause = f"and year <= {end_year}" if end_year is not None else ""
        q = duckdb.query(
            f"""
            SELECT *
            from '{observed_hydropower_glob}'
            WHERE eia_plant_id in ({', '.join([f"{plant_id}" for plant_id in plant_ids])})
            {start_clause}
            {end_clause}
        """
        )
        return q.df()

    def get_hydropower_for_plant_ids(
        self, plant_ids: Iterable[int], start_year: int = None, end_year: int = None
    ) -> pd.DataFrame:
        """
        Query a parquet database for observed hydropower for a given list of plant IDs
        :param plant_ids: a list of plant IDs
        :param start_year: the beginning year for which to query data; None for all
        :param end_year: the end year for which to query data; None for all
        :return: a DataFrame containing the observed hydropower values for the requested plants
        """
        return self._get_hydropower_for_plant_ids(
            plant_ids,
            self.observed_hydropower_glob,
            start_year,
            end_year,
        )

    def get_annual_adjusted_hydropower(self, hydropower: pd.DataFrame) -> pd.Series:
        """
        Annualizes observed hydropower and adjusts for the operating reserve percentage
        :param hydropower: a DataFrame containing the year and generation_MWh columns for a single balancing authority
        :return: a Series representing the adjusted annualized hydropower
        """
        annual = hydropower.groupby("year").generation_MWh.sum()
        adjusted = annual + annual.mean() * self.operating_reserve_percent
        return adjusted

    @staticmethod
    def efficiency_scoring(
        estimate: np.ndarray,
        flow_and_storage: pd.DataFrame,
        observed_hydropower: pd.DataFrame,
        reservoir_parameters: pd.DataFrame,
        penstock_flexibility: float,
        spill: float,
    ) -> float:
        """
        Scores an estimate of plant efficiencies and percentile factor within a balancing authority using Mean Absolute Error (MAE)
        :param estimate: a Numpy array containing the efficiency estimates; the last entry is the percentile factor
        :param flow_and_storage: a DataFrame containing the simulated flow and storage for plants in the balancing authority
        :param observed_hydropower: a DataFrame containing the observed hydropower generation for plants in the balancing authority
        :param reservoir_parameters: a DataFrame containing the reservoir/plant parameters for plants in the balancing authority
        :param penstock_flexibility: the penstock flexibility to apply to the annual generation calculation
        :param spill: the spill factor to apply to the annual generation calculation
        :return: the Mean Absolute Error (MAE) for the given estimate
        """

        percentile_factor = estimate[-1]

        reservoir_parameters["maximum_discharge_m3_s"] = (
            1e6
            * reservoir_parameters.nameplate_capacity_MW
            * penstock_flexibility
            / (9800 * reservoir_parameters.plant_head_m)
        )

        reservoir_parameters["efficiency"] = estimate[:-1]

        flow_storage_parameters = flow_and_storage.merge(
            reservoir_parameters[
                [
                    "eia_plant_id",
                    "storage_capacity_m3",
                    "plant_head_m",
                    "nameplate_capacity_MW",
                    "maximum_discharge_m3_s",
                    "use_run_of_river",
                    "efficiency",
                ]
            ],
            on="eia_plant_id",
        )

        # account for spill percentage
        flow_storage_parameters["flow"] = flow_storage_parameters["flow"] * (1 - spill)

        # count flows below the maximum discharge and divide by number of flows recorded
        flow_storage_parameters["below_max"] = (
            flow_storage_parameters.flow
            < flow_storage_parameters.maximum_discharge_m3_s
        )

        eia_groups = flow_storage_parameters.groupby("eia_plant_id")

        flow_storage_parameters["percentile_factor"] = eia_groups.below_max.transform(
            "sum"
        ) / eia_groups.below_max.transform("count")

        flow_storage_parameters["modified_flow"] = np.where(
            flow_storage_parameters.percentile_factor < percentile_factor,
            flow_storage_parameters.flow
            / (
                eia_groups.flow.transform("mean")
                / flow_storage_parameters.maximum_discharge_m3_s
            ),
            np.where(
                flow_storage_parameters.flow
                > flow_storage_parameters.maximum_discharge_m3_s,
                flow_storage_parameters.maximum_discharge_m3_s,
                flow_storage_parameters.flow,
            ),
        )

        flow_storage_parameters["height"] = np.where(
            flow_storage_parameters.use_run_of_river,
            flow_storage_parameters.plant_head_m,
            flow_storage_parameters.plant_head_m
            * (
                (
                    flow_storage_parameters.storage
                    / flow_storage_parameters.storage_capacity_m3
                )
            )
            ** (1 / 3),
        )

        flow_storage_parameters["power"] = (
            9800
            * flow_storage_parameters.modified_flow
            * flow_storage_parameters.height
            * flow_storage_parameters.efficiency
        )

        flow_storage_parameters["generation"] = (
            flow_storage_parameters["power"] * 24 / 1e6
        )

        annual_generation = flow_storage_parameters.groupby(
            flow_storage_parameters.date.dt.year
        ).generation.sum()

        return (annual_generation - observed_hydropower).abs().mean()

    @staticmethod
    def generation_scoring(
        estimate: np.ndarray,
        flow_and_storage: pd.DataFrame,
        observed_hydropower: pd.DataFrame,
        reservoir_parameters: pd.Series,
        efficiency: float,
        percentile_factor: float,
        return_generation: bool = False,
    ) -> Union[float, pd.DataFrame]:
        """
        Scores an estimate of monthly spills and penstock flexibilities for a plant using Kling-Gupta Efficiency (KGE)
        :param estimate: a Numpy array containing the monthly spill estimates for the plant; the last entry is the penstock flexibility
        :param flow_and_storage: a DataFrame containing the simulated flow and storage for the plant
        :param observed_hydropower: a DataFrame containing the observed hydropower generation for the plant
        :param reservoir_parameters: a Series containing the reservoir/plant parameters for the plant
        :param efficiency: the plant's efficiency as determined in phase one efficiency calibration
        :param percentile_factor: the balancing authority's percentile factor as determined in phase one efficiency calibration
        :param return_generation: if True, returns a DataFrame containing the plant's modeled monthly generation, rather than the KGE; default False
        :return: the Kling-Gupta Efficiency (KGE) for the given estimate, or the timeseries of monthly generation if return_generation is True
        """

        max_discharge = 1e6 * (
            reservoir_parameters.nameplate_capacity_MW
            * estimate[-1]
            / (9800 * reservoir_parameters.plant_head_m)
        )

        flow_and_storage["spill"] = list(
            map(lambda t: estimate[t.month - 1], flow_and_storage.date)
        )

        flow_and_storage["modified_flow"] = flow_and_storage.flow * (
            1 - flow_and_storage.spill
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            if (
                flow_and_storage[
                    flow_and_storage.modified_flow < max_discharge
                ].modified_flow.sum()
                / len(flow_and_storage.modified_flow)
                < percentile_factor
            ):
                flow_and_storage["modified_flow"] = flow_and_storage.modified_flow / (
                    flow_and_storage.modified_flow.mean() / max_discharge
                )
            else:
                flow_and_storage["modified_flow"] = np.where(
                    flow_and_storage.modified_flow > max_discharge,
                    max_discharge,
                    flow_and_storage.modified_flow,
                )

        if reservoir_parameters.use_run_of_river:
            height = reservoir_parameters.plant_head_m
        else:
            height = reservoir_parameters.plant_head_m * (
                (flow_and_storage.storage / reservoir_parameters.storage_capacity_m3)
            ) ** (1 / 3)

        flow_and_storage["power"] = (
            9800 * flow_and_storage.modified_flow * height * efficiency
        )

        flow_and_storage["generation"] = flow_and_storage.power * 24 / 1e6

        flow_and_storage["year"] = flow_and_storage.date.dt.year
        flow_and_storage["month"] = flow_and_storage.date.dt.month

        monthly_generation_df = (
            flow_and_storage.sort_values(["year", "month"])
            .groupby(["year", "month"])[["generation"]]
            .sum()
        )

        if return_generation:
            return monthly_generation_df

        monthly_generation_df = monthly_generation_df.reset_index()

        # merge together observed and simulated generation and keep paired values
        generation_df = monthly_generation_df.merge(observed_hydropower, how = 'left')

        nyear_sim = generation_df.shape[0] / 12

        generation_df.dropna(inplace = True)

        nyear_eval = generation_df.shape[0] / 12

        monthly_generation = generation_df.generation.values
        observed_generation = generation_df.generation_MWh.values

        # if fewer years are used for evaluation, print warning
        # TODO: fix so warning is only printed once per site - pass in iteration counter or similar
        # if nyear_eval != nyear_sim:
        #     warnings.warn(str(nyear_sim) + ' years simulated, but only ' +
        #         str(nyear_eval) + ' used in evaluation for ' +
        #         str(reservoir_parameters.eia_plant_id), RuntimeWarning)
        with np.errstate(divide="ignore", invalid="ignore"):
            kge = np.sqrt(
                ((np.corrcoef(observed_generation, monthly_generation)[0, 1] - 1) ** 2)
                + (((np.std(monthly_generation) / np.std(observed_generation)) - 1) ** 2)
                + (((np.mean(monthly_generation) / np.mean(observed_generation)) - 1) ** 2)
            )

        return kge

    @staticmethod
    def get_generation(
        calibration_parameters_path: str,
        reservoir_parameters_path: str,
        flow_and_storage_path: str,
        run_name: str,
        start_year: int = -np.inf,
        end_year: int = np.inf,
        write_output: bool = True,
        output_csv: bool = True,
        output_path: str = '.',
        parallel_tasks: int = cpu_count(logical=False),
    ) -> pd.DataFrame:
        """
        Calculates the monthly generation by plant based on the provided parameters and flows
        :param calibration_parameters_path: glob to parquet or csv files containing the calibration parameters
        :param reservoir_parameters_path: glob to parquet or csv files containing the reservoir parameters
        :param flow_and_storage_path: glob to parquet or csv files containing the flow and storage at the plants for which to calculate generation
        :param run_name: name to append to output file to track simulations
        :param start_year: year in which to start the calculation, inclusive
        :param end_year: year in which to end the calculation, inclusive
        :param write_output: if True write monthly generation to a file per balancing authority, otherwise don't write a file just return dataframe
        :param output_csv: if True write output to CSV file, else write to parquet file
        :param output_path: path to directory in which to write the output files
        :param parallel_tasks: number of tasks to run in parallel
        """
        # load calibration params
        if calibration_parameters_path.lower().endswith('csv'):
            calibration_parameters = pd.concat([
              pd.read_csv(f, converters={"monthly_spill": lambda x: [float(i) for i in x.strip("[]").replace("'","").split()]}) for f in glob(calibration_parameters_path, recursive=True)
            ])
        else:
            calibration_parameters = pd.concat([pd.read_parquet(f) for f in glob(calibration_parameters_path, recursive=True)])

        # load reservoir params
        if reservoir_parameters_path.lower().endswith('csv'):
            # reservoir_parameters = pd.concat([pd.read_csv(f) for f in glob(reservoir_parameters_path, recursive=True)])
            reservoir_parameters = pd.read_csv(reservoir_parameters_path)
        else:
            # reservoir_parameters = pd.concat([pd.read_parquet(f) for f in glob(reservoir_parameters_path, recursive=True)])
            reservoir_parameters = pd.read_parquet(reservoir_parameters_path)

        # load flow and storage
        if flow_and_storage_path.lower().endswith('csv'):
            flow_and_storage = pd.concat([pd.read_csv(f) for f in glob(flow_and_storage_path, recursive=True)])
        else:
            flow_and_storage = pd.concat([pd.read_parquet(f) for f in glob(flow_and_storage_path, recursive=True)])

        # dataframe for storing output
        monthly_generation = pd.DataFrame()

        # iterate through each plant and calculate generation
        def _get_gen(row):
            monthly_generation_plant = Model.generation_scoring(
                np.append(row.monthly_spill, np.array([row.penstock_flexibility])),
                flow_and_storage=flow_and_storage[
                  (flow_and_storage.eia_plant_id == row.eia_plant_id) &
                  (flow_and_storage.date.dt.year >= start_year) &
                  (flow_and_storage.date.dt.year <= end_year)
                ].copy(),
                observed_hydropower=None,
                reservoir_parameters=reservoir_parameters[reservoir_parameters.eia_plant_id == row.eia_plant_id].iloc[0],
                efficiency=row.efficiency,
                percentile_factor=row.percentile_factor,
                return_generation=True,
            ).rename(columns={
                'generation': 'modeled_generation_MWh'
            })
            monthly_generation_plant['eia_plant_id'] = row.eia_plant_id
            monthly_generation_plant['balancing_authority'] = row.balancing_authority
            return monthly_generation_plant
        with ThreadPool(parallel_tasks) as pool:
            monthly_generation = pool.map(
                _get_gen,
                [row for i, row in calibration_parameters.iterrows()],
            )
        monthly_generation = pd.concat(monthly_generation)

        # iterate through each balancing authority and save output file
        for group, data in monthly_generation.groupby('balancing_authority'):
            out_path = Path(f"{output_path}/{group}_{run_name}_modeled_generation.{'csv' if output_csv else 'parquet'}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            data = data.reset_index()
            if output_csv:
                data.to_csv(out_path, index=False)
            else:
                data.to_parquet(out_path, index=False)

        monthly_generation = monthly_generation.reset_index()
        
        monthly_generation["date"] = pd.to_datetime(
            monthly_generation["year"].astype(str)
            + "-"
            + monthly_generation["month"].astype(str)
            + "-1"
        )

        return monthly_generation.set_index('date')

    def efficiency_calibration(
        self,
        initial_population: np.ndarray,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
        flow_and_storage: pd.DataFrame,
        observed_hydropower: pd.Series,
        reservoir_parameters: pd.DataFrame,
    ) -> Tuple[np.ndarray, float, str]:
        """
        Performs the first phase efficiency and percentile factor calibration for plants within a balancing authority
        :param initial_population: the initial estimate as a Numpy array
        :param lower_bound: the lower bound of allowed values as a Numpy array
        :param upper_bound: the upper bound of allowed values as a Numpy array
        :param flow_and_storage: a DataFrame containing the simulated flow and storage for plants within the balancing authority
        :param observed_hydropower: a DataFrame containing the observed hydropower for plants within the balancing authority
        :param reservoir_parameters: a DataFrame containing the reservoir/plant parameters for plants within the balancing authority
        :return: a Tuple containing the best estimate, the best score, and the log message
        """
        return shuffled_complex_evolution_algorithm(
            initial_population,
            Model.efficiency_scoring,
            lower_bound,
            upper_bound,
            self.efficiency_maximum_trials,
            self.efficiency_maximum_evolution_loops,
            self.efficiency_minimum_geometric_range,
            self.efficiency_minimum_change_criteria,
            self.efficiency_number_of_complexes,
            additional_function_arguments={
                "flow_and_storage": flow_and_storage,
                "observed_hydropower": observed_hydropower,
                "reservoir_parameters": reservoir_parameters,
                "penstock_flexibility": self.efficiency_penstock_flexibility,
                "spill": self.efficiency_spill,
            },
            include_initial_point=self.efficiency_include_initial_point,
            alpha=self.efficiency_alpha,
            beta=self.efficiency_beta,
            random_number_generator=self.random_number_generator,
        )

    def generation_calibration(
        self,
        eia_plant_id: int,
        initial_population: np.ndarray,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
        flow_and_storage: pd.DataFrame,
        observed_hydropower: pd.DataFrame,
        reservoir_parameters: pd.Series,
        efficiency: float,
        percentile_factor: float,
    ) -> Tuple[dict, str]:
        """
        Performs the second phase calibration for the monthly spill and penstock flexibility of a single plant
        :param eia_plant_id: the plant ID
        :param initial_population: the initial estimate as a Numpy array
        :param lower_bound: the lower bound of allowed values as a Numpy array
        :param upper_bound: the upper bound of allowed values as a Numpy array
        :param flow_and_storage: a DataFrame containing the simulated flow and storage for the plant
        :param observed_hydropower: a DataFrame containing the observed hydropower for the plant
        :param reservoir_parameters: a Series containing the reservoir/plant parameters for the plant
        :param efficiency: the plant's efficiency as determined in phase one calibration
        :param percentile_factor: the balancing authority's percentile factor as determined in phase one calibration
        :return: a Tuple of a dictionary with the calibrated values for the plant, and a string of log messages
        """
        (
            monthly_spill_and_penstock,
            spill_score,
            log_messages,
        ) = shuffled_complex_evolution_algorithm(
            initial_population,
            Model.generation_scoring,
            lower_bound,
            upper_bound,
            self.generation_maximum_trials,
            self.generation_maximum_evolution_loops,
            self.generation_minimum_geometric_range,
            self.generation_minimum_change_criteria,
            self.generation_number_of_complexes,
            additional_function_arguments={
                "flow_and_storage": flow_and_storage,
                "observed_hydropower": observed_hydropower,
                "reservoir_parameters": reservoir_parameters,
                "efficiency": efficiency,
                "percentile_factor": percentile_factor,
            },
            alpha=self.generation_alpha,
            beta=self.generation_beta,
            include_initial_point=self.generation_include_initial_point,
            random_number_generator=self.random_number_generator,
        )
        log_message = (
            f"EIA {eia_plant_id} generation calibration:{linesep}{log_messages}{linesep}\t"
            f"The best score was: {spill_score}"
        )
        return (
            dict(
                eia_plant_id=eia_plant_id,
                name=reservoir_parameters["name"],
                efficiency=efficiency,
                percentile_factor=percentile_factor,
                monthly_spill=monthly_spill_and_penstock[:-1],
                penstock_flexibility=monthly_spill_and_penstock[-1],
            ),
            log_message,
        )

    def calibrate_balancing_authority(
        self,
        balancing_authority: str,
    ) -> pd.DataFrame:
        """
        Run the first and second phase calibration for plants within the requested balancing authority
        :param balancing_authority: the balancing authority acronym
        :return: a DataFrame containing the calibrated values for plants within the balancing authority
        """

        reservoir_parameters = self.get_reservoir_parameters_for_balancing_authority(
            balancing_authority
        )
        eia_plant_ids = list(reservoir_parameters.eia_plant_id)
        flow_storage = self.get_flow_storage_for_plant_ids(
            eia_plant_ids, self.calibration_start_year, self.calibration_end_year
        )
        hydropower = self.get_hydropower_for_plant_ids(
            eia_plant_ids, self.calibration_start_year, self.calibration_end_year
        )
        annual_hydropower = self.get_annual_adjusted_hydropower(hydropower)

        lower_bound = np.full(len(eia_plant_ids) + 1, self.lower_bound_efficiency)
        lower_bound[-1] = self.lower_bound_percentile_factor

        upper_bound = np.full(len(eia_plant_ids) + 1, self.upper_bound_efficiency)
        upper_bound[-1] = self.upper_bound_percentile_factor

        initial_population = 0.5 * (lower_bound + upper_bound)

        # calibrate annual hydropower efficiencies
        efficiencies, efficiency_score, log_messages = self.efficiency_calibration(
            initial_population,
            lower_bound,
            upper_bound,
            flow_storage,
            annual_hydropower,
            reservoir_parameters,
        )

        self.logger.info(
            f"{balancing_authority} efficiency calibration:{linesep}{log_messages}{linesep}\t"
            f"The best score was: {efficiency_score}{linesep}"
        )

        # calibrate monthly generation spill and penstock flexibility
        lower_bound = np.full(13, self.lower_bound_spill)
        lower_bound[-1] = self.lower_bound_penstock
        upper_bound = np.full(13, self.upper_bound_spill)
        upper_bound[-1] = self.upper_bound_penstock
        initial_population = lower_bound + (upper_bound - lower_bound) / 2.0
        with Pool(self.parallel_tasks) as pool:
            plant_calibrations_and_logs = pool.starmap(
                self.generation_calibration,
                [
                    (
                        eia_plant_id,
                        initial_population,
                        lower_bound,
                        upper_bound,
                        flow_storage[flow_storage.eia_plant_id == eia_plant_id].copy(),
                        hydropower[hydropower.eia_plant_id == eia_plant_id].sort_values(
                            ["year", "month"]
                        ),
                        reservoir_parameters[
                            reservoir_parameters.eia_plant_id == eia_plant_id
                        ].iloc[0],
                        efficiencies[i],
                        efficiencies[-1],
                    )
                    for i, eia_plant_id in enumerate(
                        list(reservoir_parameters.eia_plant_id)
                    )
                ],
            )
        plant_calibrations = []
        for x in plant_calibrations_and_logs:
            plant_calibrations.append(x[0])
            self.logger.info(x[1])
        return pd.DataFrame(plant_calibrations)

    def plot(self, calibrations: pd.DataFrame, show_plots: bool = True):

        if isinstance(self.balancing_authority, str):
            balancing_authority = [self.balancing_authority]
        else:
            balancing_authority = self.balancing_authority

        parameters = []
        for ba in balancing_authority:
            parameters.append(self.get_reservoir_parameters_for_balancing_authority(ba))
        parameters = pd.concat(parameters, ignore_index=True)
        observed_hydropower = self.get_hydropower_for_plant_ids(
            parameters.eia_plant_id
        ).sort_values(["year", "month"])

        plants = []
        for i, row in calibrations.iterrows():
            flow_and_storage = self.get_flow_storage_for_plant_ids([row.eia_plant_id])
            monthly_generation = (
                Model.generation_scoring(
                    np.append(row.monthly_spill, np.array([row.penstock_flexibility])),
                    flow_and_storage=flow_and_storage,
                    observed_hydropower=observed_hydropower[
                        (observed_hydropower.eia_plant_id == row.eia_plant_id)
                        & observed_hydropower.year.isin(
                            flow_and_storage.date.dt.year.unique()
                        )
                    ],
                    reservoir_parameters=parameters[
                        parameters.eia_plant_id == row.eia_plant_id
                    ].iloc[0],
                    efficiency=row.efficiency,
                    percentile_factor=row.percentile_factor,
                    return_generation=True,
                )
                .rename(columns={"generation": "modeled_generation_MWh"})
                .join(
                    observed_hydropower[
                        observed_hydropower.eia_plant_id == row.eia_plant_id
                    ].set_index(["eia_plant_id", "year", "month"])
                )
                .rename(
                    columns={
                        "generation_MWh": "observed_generation_MWh",
                    }
                )
            ).reset_index()
            monthly_generation["date"] = pd.to_datetime(
                monthly_generation["year"].astype(str)
                + "-"
                + monthly_generation["month"].astype(str)
                + "-1"
            )
            monthly_generation = monthly_generation.set_index("date", drop=True).drop(
                columns=["year", "month"]
            )
            monthly_generation['balancing_authority'] = row.balancing_authority
            monthly_generation['name'] = row['name']
            plants.append(monthly_generation)

        plants = pd.concat(plants)

        figs = []
        for ba in plants.balancing_authority.unique():
            subset = plants[plants.balancing_authority==ba]
            cols = 4
            rows = int(np.ceil(len(subset.eia_plant_id.unique()) / cols) + 1)
            
            fig = plt.figure(dpi=300, figsize=(19.2, 19.2))
            gs = fig.add_gridspec(rows, cols)
        
            ba_ax = fig.add_subplot(gs[0, :])
            
            subset.groupby('date')[['modeled_generation_MWh', 'observed_generation_MWh']].sum().plot(
                ax=ba_ax, style=["m-", "k--"], linewidth=1, 
            )
            ba_ax.set_ylabel('Generation [MWh]')
            ba_ax.set_xlabel('')
            ba_ax.set_title(f'{ba} Total')
            
            for i, plant in enumerate(subset.eia_plant_id.unique()):
                
                plant_subset = subset[subset.eia_plant_id == plant]
        
                r = 1 + i//cols
                c = i%cols
                
                plant_ax = fig.add_subplot(gs[r, c])
                plant_subset[['modeled_generation_MWh', 'observed_generation_MWh']].plot(
                    ax=plant_ax, style=["m-", "k--"], linewidth=1, legend=False
                )
                plant_ax.set_ylabel('')
                plant_ax.set_xlabel('')
                plant_ax.set_title(f"{plant_subset.iloc[0]['name']} ({plant})")
        
            fig.tight_layout()
            figs.append(fig)

        if show_plots:
            try:
                for f in figs:
                    f.show(block=False)
            except:
                pass
        
        return figs

    def _run(self) -> pd.DataFrame:
        # solve each balancing authority
        if isinstance(self.balancing_authority, str):
            balancing_authority = [self.balancing_authority]
        else:
            balancing_authority = self.balancing_authority

        calibrations = []
        for ba in balancing_authority:
            # TODO check that number of plants is consistent?
            ba_calibration = timing(self.logger, f"{ba} calibrated in")(
                self.calibrate_balancing_authority
            )(ba)
            ba_calibration["balancing_authority"] = ba
            calibrations.append(ba_calibration)
            output_name = f"{self.output_path}/{ba}_plant_calibrations"
            Path(output_name).parent.mkdir(parents=True, exist_ok=True)
            if self.output_type == "parquet":
                ba_calibration.to_parquet(f"{output_name}.parquet", index=False)
            else:
                ba_calibration.to_csv(f"{output_name}.csv", index=False)

        return pd.concat(calibrations)

    def run(self) -> pd.DataFrame:
        """
        Run the first and second phase calibration for plants within the configured balancing authorities
        :return: a DataFrame containing the calibrated values for plants within the configured balancing authorities
        """
        return timing(self.logger, "Calibrations completed in")(self._run)()

    @staticmethod
    def update_legacy_input_files(
        daily_flow_storage_path: str,
        reservoir_parameters_path: str,
        monthly_observed_generation_path: str,
        daily_start_date: str,
        monthly_start_date: str,
        output_path: str = None,
        includes_leap_days: bool = False,
    ) -> None:
        """
        Converts Excel and Matlab files used in the original Matlab code to the expected Apache Parquet file format
        :param daily_flow_storage_path: path to a Matlab file (.mat) containing simulated daily flow and storage for plants
        :param reservoir_parameters_path: path to an Excel file (.xlsx) containing reservoir/plant parameters
        :param monthly_observed_generation_path: path to an Excel file (.xlsx) containing observed monthly hydropower for plants
        :param daily_start_date: start date for values in the daily_flow_storage_path file, formatted like "1980-01-01"
        :param monthly_start_date: start date for values in the monthly_observed_generation_path file, formatted like "1980-01-01"
        :param output_path: path to a directory in which to write the output files; defaults to same place as the input files
        :param includes_leap_days: whether or not values in the monthly_observed_generation_file include leap days; default False
        :return: Nothing, but new parquet files are written
        """

        # load the reservoir parameters Excel file
        reservoir_parameters = pd.read_excel(reservoir_parameters_path)

        # rename the columns and update boolean types
        reservoir_parameters = reservoir_parameters.rename(
            columns=dict(
                productionRank="production_rank",
                Name="name",
                Power_Model_1_true_0_false_="use_power_model",
                lat_1_8_="nldas_latitude_eighth_degree",
                lon_1_8_="nldas_longitude_eighth_degree",
                nameplate_MW_="nameplate_capacity_MW",
                annualProduction_MWh_="annual_production_MWh",
                lat="latitude",
                long="longitude",
                ROR="use_run_of_river",
                Units="number_of_generating_units",
                Pump="use_pumped_storage",
                Reservoir="has_reservoir",
                alpha="alpha",
                Slope_NID="slope_reported_by_nid",
                Slope_Grand="slope_reported_by_grand",
                Head_NID_m_="plant_head_m",
                Storage_m3_="storage_capacity_m3",
                ID="eia_plant_id",
                BA="balancing_authority",
                PMA_region="power_management_administration_region",
            )
        ).astype(
            dict(
                use_power_model=bool,
                use_run_of_river=bool,
                use_pumped_storage=bool,
                has_reservoir=bool,
            )
        )

        # load the daily flow/storage matlab file
        daily_flow_storage = loadmat(daily_flow_storage_path)

        # get a pandas date range of the days
        if includes_leap_days:
            dates = pd.date_range(
                start=parse(daily_start_date).date(),
                periods=daily_flow_storage["flow_a"].shape[0],
                freq="D",
            )
        else:
            d = parse(daily_start_date).date()
            dates = pd.date_range(
                start=d,
                end=d.replace(
                    year=d.year + daily_flow_storage["flow_a"].shape[0] // 365 - 1,
                    month=12,
                    day=31,
                ),
                freq="D",
            )
            dates = dates[~((dates.month == 2) & (dates.day == 29))]

        # use the eia plant id for the plant index
        plants = reservoir_parameters.eia_plant_id.values

        # create a dataframe for flow and storage from the mat file
        daily_flow_storage = pd.DataFrame(
            dict(
                flow=daily_flow_storage["flow_a"].flatten(),
                storage=daily_flow_storage["storage_a"].flatten(),
            ),
            index=pd.MultiIndex.from_product(
                [dates, plants], names=["date", "eia_plant_id"]
            ),
        ).reset_index()

        # load the monthly observed power Excel file
        monthly_observed_generation = pd.read_excel(
            monthly_observed_generation_path, header=1
        )

        # transpose the eia id columns into rows
        monthly_observed_generation = (
            monthly_observed_generation.stack()
            .reset_index()
            .rename(
                columns={
                    "level_0": "month",
                    "level_1": "eia_plant_id",
                    0: "generation_MWh",
                }
            )
        )

        # convert month number to a year and month
        monthly_start_date = parse(monthly_start_date).date()
        monthly_observed_generation["year"] = (
            monthly_start_date.year + monthly_observed_generation.month // 12
        )
        monthly_observed_generation["month"] = (
            monthly_start_date.month + monthly_observed_generation.month % 12
        )
        monthly_observed_power = monthly_observed_generation[
            ["year", "month", "eia_plant_id", "generation_MWh"]
        ]

        if output_path is None:
            output_path = Path(daily_flow_storage_path).parent

        # write reservoir params to parquet file
        reservoir_parameters.to_parquet(
            f"{Path(output_path)}/{Path(reservoir_parameters_path).stem}.parquet"
        )

        # write daily flow/storage to parquet file
        daily_flow_storage.to_parquet(
            f"{Path(output_path)}/{Path(daily_flow_storage_path).stem}.parquet"
        )

        # write monthly observed power to parquet file
        monthly_observed_power.to_parquet(
            f"{Path(output_path)}/{Path(monthly_observed_generation_path).stem}.parquet"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the path to a configuration yaml file as an argument.")
        sys.exit(1)
    config_file = sys.argv[1]
    m = Model(config_file)
    m.run()
