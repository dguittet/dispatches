#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis
# Platform to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES),
# and is copyright (c) 2022 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#
#################################################################################
import pandas as pd
from collections import deque
from functools import partial
from idaes.apps.grid_integration.multiperiod.multiperiod import MultiPeriodModel
import pyomo.environ as pyo
from dispatches.case_studies.renewables_h2_case.re_h2_parameters import h2_mols_per_kg, pem_bar, pem_temp
from dispatches.case_studies.renewables_h2_case.wind_battery_hydrogen_flowsheet import (
    wind_battery_hydrogen_mp_block,
    size_constraints,
    calculate_variable_costs,
    wind_battery_hydrogen_variable_pairs,
    wind_battery_hydrogen_periodic_variable_pairs
)

def create_multiperiod_wind_battery_hydrogen_model(n_time_points, wind_capacity_factors, input_params):
    """This function creates a MultiPeriodModel for the wind battery hydrogen model.

    Args:
        n_time_points (int): number of time period for the model.

    Returns:
        MultiPeriodModel: a MultiPeriodModel for the wind battery hydrogen model.
    """
    input_params["wind_resource"] = {t:
                                        {'wind_resource_config': {
                                            'capacity_factor': 
                                                [wind_capacity_factors[t]]}} for t in range(len(wind_capacity_factors))}

     # create the multiperiod model object
    if 'design_opt' in input_params.keys():
        input_params['design_opt'] = False
    mp_wind_battery_hydrogen = MultiPeriodModel(
        n_time_points=n_time_points,
        process_model_func=partial(wind_battery_hydrogen_mp_block, input_params=input_params, verbose=False),
        linking_variable_func=wind_battery_hydrogen_variable_pairs,
        periodic_variable_func=wind_battery_hydrogen_periodic_variable_pairs,
    )

    mp_wind_battery_hydrogen.build_multi_period_model(input_params['wind_resource'])
    size_constraints(mp_wind_battery_hydrogen, input_params)
    calculate_variable_costs(mp_wind_battery_hydrogen, input_params)

    return mp_wind_battery_hydrogen


def transform_design_model_to_operation_model(
    mp_wind_battery_hydrogen,
):
    """Transform the multiperiod wind battery design model to operation model.

    Args:
        mp_wind_battery_hydrogen (MultiPeriodModel): a created multiperiod wind battery object
    """

    blks = mp_wind_battery_hydrogen.get_active_process_blocks()

    for t, b in enumerate(blks):
        if t == 0:
            b.fs.battery.initial_state_of_charge.fix()
            b.fs.h2_tank.tank_holdup_previous.fix()
            
        b.fs.h2_tank.outlet_to_pipeline.flow_mol.fix(0)

        # deactivate periodic boundary condition
        if t == len(blks) - 1:
            b.periodic_constraints.deactivate()

    return


def update_wind_capacity_factor(mp_wind_battery_hydrogen, new_capacity_factors):
    """Update the wind capacity factor in the model during rolling horizon.

    Args:
        mp_wind_battery (MultiPeriodModel): a created multiperiod wind battery object
        new_capacity_factors (list): a list of new wind capacity
    """

    blks = mp_wind_battery_hydrogen.get_active_process_blocks()
    for idx, b in enumerate(blks):
        b.fs.windpower.capacity_factor[0] = new_capacity_factors[idx]

    return


class MultiPeriodWindBatteryHydrogen:
    def __init__(
        self,
        model_data,
        wind_capacity_factors,
        input_params
    ):
        """Initialize a multiperiod wind battery model object for double loop.

        Args:
            model_data (GeneratorModelData): a GeneratorModelData that holds the generators params.
            wind_capacity_factors (dict):
            input_params is a dictionary containing the following key-value pairs:
                wind_pmax_mw (float): wind farm capapcity in MW
                battery_pmax_mw (float): battery power output capapcity in MW
                battery_energy_capacity_mwh (float): battery energy capapcity in MW
                pem_pmax_mw (float): 
                h2_tank_kgmax (float):
                h2_turb_pmax_mw (float): 
                turb_conv_rate (float):
                costs
        Raises:
            ValueError: if wind capacity factor is not provided, ValueError will be raised
        """

        self.model_data = model_data
        self._wind_capacity_factors = wind_capacity_factors
        self._design_params = input_params
        self._horizon = 0

        # a list that holds all the result in pd DataFrame
        self.result_list = []

    def populate_model(self, b, horizon):
        """Create a wind-battery model using the `MultiPeriod` package.

        Args:
            b (block): this is an empty block passed in from either a bidder or tracker
            horizon (int): the number of time periods
        """
        blk = b
        if not blk.is_constructed():
            blk.construct()

        self._horizon = horizon
        blk.windBatteryHydrogen = create_multiperiod_wind_battery_hydrogen_model(horizon, 
            self._wind_capacity_factors, self._design_params)
        transform_design_model_to_operation_model(
            mp_wind_battery_hydrogen=blk.windBatteryHydrogen
        )

        # deactivate any objective functions
        for obj in blk.windBatteryHydrogen.pyomo_model.component_objects(pyo.Objective):
            obj.deactivate()

        # initialize time index for this block
        b._time_idx = pyo.Param(initialize=0, mutable=True)

        active_blks = blk.windBatteryHydrogen.get_active_process_blocks()

        # create expression that references underlying power variables in multi-period rankine
        blk.HOUR = pyo.Set(initialize=range(horizon))
        blk.P_T = pyo.Expression(blk.HOUR)
        blk.tot_cost = pyo.Expression(blk.HOUR)
        blk.wind_waste_penalty = pyo.Param(default=100, mutable=True)
        blk.wind_waste = pyo.Expression(blk.HOUR)
        # blk.battery_priority_penalty = pyo.Param(default=0, mutable=True)
        # blk.battery_priority = pyo.Expression(blk.HOUR)
        for (t, b) in enumerate(active_blks):
            blk.P_T[t] = (b.fs.splitter.grid_elec[0] + b.fs.battery.elec_out[0] + b.fs.h2_turbine_elec) * 1e-3
            blk.wind_waste[t] = (b.fs.windpower.system_capacity * b.fs.windpower.capacity_factor[0] - b.fs.windpower.electricity[0]) * 1e-3
            # blk.battery_priority[t] = (b.fs.h2_tank.inlet.flow_mol[0] * 3600 / h2_mols_per_kg * self._design_params['turb_conv'] - b.fs.battery.elec_in[0]) * 1e-3
            # blk.tot_cost[t] = b.var_total_cost + blk.wind_waste_penalty * blk.wind_waste[t] + blk.battery_priority_penalty * blk.battery_priority[t]
            blk.tot_cost[t] = b.var_total_cost + blk.wind_waste_penalty * blk.wind_waste[t] 

        return

    def update_model(self, b, realized_soc, realized_energy_throughput, realized_h2_tank_holdup, realized_h2_throughput):
        """Update variables using future wind capacity the realized state-of-charge and enrgy throughput profiles.

        Args:
            b (block): the block that needs to be updated
            realized_soc (list): list of realized state of charge
            realized_energy_throughput (list): list of realized energy throughput
            realized_h2_tank_holdup (list): list of realized h2 tank holdups
        """

        blk = b
        mp_wind_battery_hydrogen = blk.windBatteryHydrogen
        active_blks = mp_wind_battery_hydrogen.get_active_process_blocks()

        new_init_soc = round(realized_soc[-1], 2)
        active_blks[0].fs.battery.initial_state_of_charge.fix(new_init_soc)

        new_init_energy_throughput = round(realized_energy_throughput[-1], 2)
        active_blks[0].fs.battery.initial_energy_throughput.fix(
            new_init_energy_throughput
        )

        new_init_h2_holdup = round(realized_h2_tank_holdup[-1], 2)
        active_blks[0].fs.h2_tank.tank_holdup_previous.fix(new_init_h2_holdup)

        # shift the time -> update capacity_factor
        time_advance = min(len(realized_soc), 24)
        b._time_idx = pyo.value(b._time_idx) + time_advance

        new_capacity_factors = self._get_capacity_factors(b)
        update_wind_capacity_factor(mp_wind_battery_hydrogen, new_capacity_factors)

        return

    def _get_capacity_factors(self, b):
        """Fetch the future capacity factor.

        Args:
            b (block): the block that needs to be updated

        Returns:
            list: the capcity factors for the immediate future
        """

        horizon_len = len(b.windBatteryHydrogen.get_active_process_blocks())
        start_ind = pyo.value(b._time_idx)
        ans = self._wind_capacity_factors[
            start_ind : start_ind + horizon_len
        ]
        if len(ans) < horizon_len:
            start_ind -= len(self._wind_capacity_factors) - len(ans)
            ans += self._wind_capacity_factors[start_ind:start_ind + horizon_len - len(ans)]
        return ans

    @staticmethod
    def get_last_delivered_power(b, last_implemented_time_step):
        """Get last delivered power.

        Args:
            b (block): a multiperiod block
            last_implemented_time_step (int):  time index for the last implemented time period

        Returns:
            float: last delivered power
        """

        blk = b
        return pyo.value(blk.P_T[last_implemented_time_step])

    @staticmethod
    def get_implemented_profile(b, last_implemented_time_step):
        """Get implemented profiles, i.e., realized state-of-charge, energy throughput and h2 holdup

        Args:
            b (block): a multiperiod block
            last_implemented_time_step (int):  time index for the last implemented time period

        Returns:
            dict: dictionalry of implemented profiles.
        """

        blk = b
        mp_wind_battery_hydrogen = blk.windBatteryHydrogen
        active_blks = mp_wind_battery_hydrogen.get_active_process_blocks()

        realized_soc = deque(
            pyo.value(active_blks[t].fs.battery.state_of_charge[0])
            for t in range(last_implemented_time_step + 1)
        )

        realized_energy_throughput = deque(
            pyo.value(active_blks[t].fs.battery.energy_throughput[0])
            for t in range(last_implemented_time_step + 1)
        )

        realized_h2_tank_holdup = deque(
            pyo.value(active_blks[t].fs.h2_tank.tank_holdup[0])
            for t in range(last_implemented_time_step + 1)
        )

        realized_h2_throughput = deque(
            pyo.value(active_blks[t].fs.battery.state_of_charge[0])
            for t in range(last_implemented_time_step + 1)
        )

        return {
            "realized_soc": realized_soc,
            "realized_energy_throughput": realized_energy_throughput,
            "realized_h2_tank_holdup": realized_h2_tank_holdup,
            "realized_h2_throughput": realized_h2_throughput
        }

    def record_results(self, b, date=None, hour=None, **kwargs):
        """Record the operations stats for the model, i.e., generator name, data, hour, horizon,
        total wind generation, total power output, wind power output, battery power output, charging power,
        state of charge, total costs.

        Args:
            b (block): a multiperiod block
            date (str, optional): current simulation date. Defaults to None.
            hour (int, optional): current simulation hour. Defaults to None.
        """
        blk = b
        mp_wind_battery_hydrogen = blk.windBatteryHydrogen
        active_blks = mp_wind_battery_hydrogen.get_active_process_blocks()

        df_list = []
        for t, process_blk in enumerate(active_blks):

            result_dict = {}

            result_dict["Generator"] = self.model_data.gen_name
            result_dict["Date"] = date
            result_dict["Hour"] = hour

            # simulation inputs
            result_dict["Horizon [hr]"] = int(t)

            # model vars
            round_digits = 3
            result_dict["Total Wind Generation [MW]"] = float(
                round(pyo.value(process_blk.fs.windpower.electricity[0]) * 1e-3, round_digits)
            )
            result_dict["Total Power Output [MW]"] = float(
                round(pyo.value(blk.P_T[t]), round_digits)
            )
            result_dict["Wind Power Output [MW]"] = float(
                round(pyo.value(process_blk.fs.splitter.grid_elec[0] * 1e-3), round_digits)
            )
            result_dict["Wind Curtailment [MW]"] = float(
                round(pyo.value(blk.wind_waste[0]), round_digits)
            )
            result_dict["Battery Power Output [MW]"] = float(
                round(pyo.value(process_blk.fs.battery.elec_out[0] * 1e-3), round_digits)
            )
            result_dict["Wind Power to Battery [MW]"] = float(
                round(pyo.value(process_blk.fs.battery.elec_in[0] * 1e-3), round_digits)
            )
            result_dict["State of Charge [MWh]"] = float(
                round(pyo.value(process_blk.fs.battery.state_of_charge[0] * 1e-3), round_digits)
            )
            result_dict["Wind Power to PEM [MW]"] = float(
                round(pyo.value(process_blk.fs.pem.electricity[0] * 1e-3), round_digits)
            )
            result_dict["PEM H2 Output [kg]"] = float(
                round(pyo.value(process_blk.fs.pem.outlet_state[0].flow_mol * 3600 / h2_mols_per_kg), round_digits)
            )
            result_dict["Tank H2 Input [kg]"] = float(
                round(pyo.value(process_blk.fs.h2_tank.inlet.flow_mol[0] * 3600 / h2_mols_per_kg), round_digits)
            )
            result_dict["H2 Sales [kg]"] = float(
                round(pyo.value(process_blk.fs.h2_tank.outlet_to_pipeline.flow_mol[0] * 3600 / h2_mols_per_kg), round_digits)
            )
            result_dict["Turbine H2 Input [kg]"] = float(
                round(pyo.value(process_blk.fs.h2_tank.outlet_to_turbine.flow_mol[0] * 3600 / h2_mols_per_kg), round_digits)
            )
            result_dict["Tank Holdup [kg]"] = float(
                round(pyo.value(process_blk.fs.h2_tank.tank_holdup[0] / h2_mols_per_kg), round_digits)
            )
            result_dict["Turbine Power Output [MW]"] = float(
                round(pyo.value(process_blk.fs.h2_turbine_elec * 1e-3), round_digits)
            )
            result_dict["Total Cost [$]"] = float(round(pyo.value(blk.tot_cost[t]), round_digits))

            for key in kwargs:
                result_dict[key] = kwargs[key]

            result_df = pd.DataFrame.from_dict(result_dict, orient="index")
            df_list.append(result_df.T)

        # append to result list
        self.result_list.append(pd.concat(df_list))

        return

    def write_results(self, path):
        """Write the saved results to a csv file.

        Args:
            path (str): the path to write the results.
        """

        pd.concat(self.result_list).to_csv(path, index=False)

    @property
    def power_output(self):
        return "P_T"

    @property
    def total_cost(self):
        return ("tot_cost", 1)
