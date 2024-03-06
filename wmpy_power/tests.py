import tempfile
import unittest

import numpy as np
import pandas as pd

from wmpy_power.sce_ua import shuffled_complex_evolution_algorithm
from wmpy_power import Model

class ShuffledComplexEvolutionTest(unittest.TestCase):
    """Test that the SCE algorithm correctly optimize simple problems"""

    @staticmethod
    def fold_volume_objective_function(fold, length, width):
        """Return the negative volume of the folded box (algorithm always tries to minimize objective function)"""
        return -((fold * (length - 2 * fold) * (width - 2 * fold))).sum()

    def test_single_box_folding(self):
        """
        Maximize the volume of a folded cardboard sheet by choosing size of fold
        Based on https://tutorial.math.lamar.edu/Solutions/CalcI/Optimization/Prob8.aspx
        Assume width w = 20 and length l = 50, maximize volume by choosing height h
        Lower bound is 0, upper bound is half of the smallest dimension
        """
        length = 50
        width = 20
        lower_bound = np.array([0])
        upper_bound = np.array([np.min(np.array([length, width]) / 2)])
        initial_guess = 0.5 * (lower_bound + upper_bound)

        h, score, logs = shuffled_complex_evolution_algorithm(
            initial_population=initial_guess,
            scoring_function=self.fold_volume_objective_function,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            additional_function_arguments={
                'length': length,
                'width': width,
            },
            maximum_trials=1e6,
            maximum_evolution_loops=4,
            minimum_geometric_range=1e-9,
            minimum_change_criteria=1e-9,
            number_of_complexes=3,
            alpha=1.0,
            beta=0.5,
            seed=2001,
        )

        self.assertAlmostEqual(
            first=h[0],
            second=4.4018,
            places=1,
            msg="Incorrect optimzation for single box folding volume problem."
        )


    def test_multi_box_folding(self):
        """
        Same as the single problem but now we want to optimize multiple boxes at once
        """

        lengths = np.array([50, 20, 40, 100, 70, 90, 35])
        widths = np.array( [20, 50, 40,  10, 30, 60, 75])
        lower_bound = np.array([0] * len(lengths))
        upper_bound = np.minimum(lengths, widths) / 2
        initial_guess = 0.5 * (lower_bound + upper_bound)

        folds, score, logs = shuffled_complex_evolution_algorithm(
            initial_population=initial_guess,
            scoring_function=self.fold_volume_objective_function,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            additional_function_arguments={
                'length': lengths,
                'width': widths,
            },
            maximum_trials=1e6,
            maximum_evolution_loops=4,
            minimum_geometric_range=1e-9,
            minimum_change_criteria=1e-9,
            number_of_complexes=3,
            alpha=1.0,
            beta=0.5,
            seed=2001,
        )

        np.testing.assert_array_almost_equal(
            folds,
            np.array([ 4.4, 4.4, 6.7, 2.4, 6.5, 11.8, 7.5 ]),
            decimal=1,
            err_msg="Incorrect optimzation for multi box folding volume problem.",
            verbose=True,
        )


class ModelTest(unittest.TestCase):
    """Test that the wmpy_power model loads correctly and interfaces with the SCE algorithm as expected"""

    def test_model(self):
        
        # use a temporary directory for input/output files
        with tempfile.TemporaryDirectory() as tmp_dir:

            # create sample input files
            # let's imagine two hydropower facilities, one with a reservoir and one run-of-river (no storage)
            # we'll generate sine waves to represent their flow, storage, and generation, periodic by season

            # parameters
            pd.DataFrame([
                {
                    'eia_plant_id': 1,
                    'balancing_authority': 'test',
                    'name': 'one',
                    'nameplate_capacity_MW': 50.0,
                    'plant_head_m': 12.0,
                    'storage_capacity_m3': 1000000.0,
                    'use_run_of_river': False,
                },
                {
                    'eia_plant_id': 2,
                    'balancing_authority': 'test',
                    'name': 'two',
                    'nameplate_capacity_MW': 20.0,
                    'plant_head_m': 4.0,
                    'storage_capacity_m3': np.nan,
                    'use_run_of_river': True,
                }
            ]).to_parquet(f'{tmp_dir}/parameters.parquet', index=False)
            
            # monthly hydropower
            generation = pd.DataFrame(
                [x for xs in [[{
                    'year': y,
                    'month': m,
                    'eia_plant_id': 1,
                    'generation_MWh': 50/2 + 50/2 * np.sin(2 * np.pi * 3 * (i*12 + j) / 12),
                }, {
                    'year': y,
                    'month': m,
                    'eia_plant_id': 2,
                    'generation_MWh': 20/2 + 20/2 * np.sin(2 * np.pi * 3 * (i*12 + j) / 12),
                }] for i, y in enumerate(np.arange(1980, 1985)) for j, m in enumerate(np.arange(1, 13))] for x in xs]
            )
            generation.to_parquet(f'{tmp_dir}/hydropower.parquet', index=False)
            
            # daily flow and storage
            pd.DataFrame(
                [x for xs in [[{
                    'date': d,
                    'eia_plant_id': 1,
                    'flow': 20/2 + 20/2 * np.sin(2 * np.pi * 91 * i / 365),
                    'storage': np.min([ np.max([1000000/2 + 1000000/2 * np.sin(2 * np.pi * 91 * i / 1827), 100000]), 900000]),
                }, {
                    'date': d,
                    'eia_plant_id': 2,
                    'flow': 10/2 + 10/2 * np.sin(2 * np.pi * 91 * i / 365),
                    'storage': np.nan,
                }] for i, d in enumerate(pd.date_range(start='1980-01-01', end='1984-12-31'))] for x in xs]
            ).to_parquet(f'{tmp_dir}/flow_storage.parquet', index=False)

            # calibrate the model to the first 4 years of data
            m = Model(
                calibration_start_year=1980,
                calibration_end_year=1983,
                balancing_authority='test',
                simulated_flow_and_storage_glob=f'{tmp_dir}/flow_storage.parquet',
                observed_hydropower_glob=f'{tmp_dir}/hydropower.parquet',
                reservoir_parameter_glob=f'{tmp_dir}/parameters.parquet',
                seed=2001,
                log_to_stdout=False,
                log_to_file=False,
                output_path=tmp_dir,
                output_type='parquet',
                parallel_tasks=1,
            )
            calibrations = m.run()

            # check that the efficiencies and penstock flexibilities are close to expected
            np.testing.assert_array_almost_equal(
                calibrations.efficiency.values,
                np.array([ 0.90, 0.91 ]),
                decimal=2,
                err_msg="Incorrect efficiency calibration.",
                verbose=True,
            )
            np.testing.assert_array_almost_equal(
                calibrations.penstock_flexibility.values,
                np.array([ 0.99, 1.01 ]),
                decimal=1,
                err_msg="Incorrect penstock flexibility calibration.",
                verbose=True,
            )

            # check that the modeled generation for the 5th year is close to the "observed" generation
            modeled_generation = Model.get_generation(
                calibration_parameters_path = f'{tmp_dir}/test_plant_calibrations.parquet',
                reservoir_parameters_path = f'{tmp_dir}/parameters.parquet',
                flow_and_storage_path = f'{tmp_dir}/flow_storage.parquet',
                run_name = 'test',
                start_year = 1984,
                end_year = 1984,
                write_output = False,
                parallel_tasks=1,
            )
            np.testing.assert_allclose(
                modeled_generation.reset_index().modeled_generation_MWh.values,
                generation[generation.year == 1984].sort_values(['eia_plant_id', 'year', 'month']).generation_MWh.values,
                rtol=0.9,
                atol=10,
                err_msg="Incorrect modeled generation.",
                verbose=True,
            )
        