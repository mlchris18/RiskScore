import assetra
from assetra.system import EnergySystemBuilder
from assetra.units import StochasticUnit
from assetra.units import StorageUnit
from assetra.units import EnergyUnit
from assetra.system import EnergySystem
from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np 
from logging import getLogger
import os
import csv
import argparse

from V1_custom_units import CustomStorageUnit, HydroUnit, SolarandWindUnit
# Update units and put hydro first 
assetra.units.NONRESPONSIVE_UNIT_TYPES.insert(1, SolarandWindUnit)
assetra.units.NONRESPONSIVE_UNIT_TYPES.insert(2, HydroUnit)

#read in arguments from command prompt 
parser = argparse.ArgumentParser(description='Process YEAR, GCM, and REGION.')
parser.add_argument('--year', type=int, required=True, help='Year to be processed')
parser.add_argument('--gcm', type=str, required=True, help='GCM to be processed')
parser.add_argument('--gcm_full', type=str, required=True, help='Full GCM to be processed')
parser.add_argument('--fleet_file', type=str, required=True, help='Fleet file to be processed')
parser.add_argument('--region_name', type=str, required=True, help='Region to be processed')

args = parser.parse_args()

 # Your code logic directly here
print(f'Received Year: {args.year}')
print(f'Received GCM: {args.gcm}')
print(f'Received GCM Path: {args.gcm_full}')
print(f'Received Fleet Directory: {args.fleet_file}')
print(f'Received Region Name: {args.region_name}')

year = args.year
gcm = args.gcm
gcm_full = args.gcm_full
fleet_file = args.fleet_file
region_name = args.region_name

#general generation portfolio read in 
Full_Generation_Portfolio = pd.read_csv(fleet_file)

def Hydro_ReadIn(gcm_full, gcm, year): 
    #read in hydro predictions
    hydro_predictions_df = pd.read_csv(f'/nfs/turbo/seas-mtcraig-climate/WRFDownscaled/{gcm_full}/hydro_wecc_regional.csv', usecols = ['date', 'CAMX', 'Desert Southwest', 'NWPP']).set_index('date')
    #rename DSW
    hydro_predictions_df.rename(columns={'Desert Southwest': 'DSW'}, inplace=True)
    # Ensure the date column is a datetime object
    hydro_predictions_df.index = pd.to_datetime(hydro_predictions_df.index)
    # Convert the DataFrame to an xarray DataArray
    monthly_expected_generation_xr = hydro_predictions_df.to_xarray()
    # Extract month and year from the date index
    monthly_expected_generation_xr['month'] = ('time', monthly_expected_generation_xr['date.month'].data)
    monthly_expected_generation_xr['year'] = ('time', monthly_expected_generation_xr['date.year'].data)
    # Reshape for multi-dimensional indexing
    monthly_expected_generation_xr = monthly_expected_generation_xr.rename({'date': 'time'})
    #limit time for now 
    monthly_expected_generation_xr = monthly_expected_generation_xr.sel(time = slice(np.datetime64(f'{year}-09-01'), np.datetime64(f'{year+1}-08-01')))
    return monthly_expected_generation_xr

def getRegional_demand(gcm, ba_list, unit_count): 
    BAs = ba_list
    demand_list = []

    # Read demand data for each BA and store in a list
    for ba in BAs:
        hourly_demand = pd.read_csv(f'/nfs/turbo/seas-mtcraig-climate/Martha_Research/WRF_Data/MLPAlgorithm/{gcm.upper()}_Demand/{ba}_{year}_demand.csv')
        hourly_demand = hourly_demand.rename(columns={'Unnamed: 0': 'time'})
        hourly_demand['time'] = pd.to_datetime(hourly_demand['time'])  # Ensure 'time' is in datetime format
        hourly_demand = hourly_demand.set_index('time')
        hourly_demand_xr = hourly_demand.to_xarray().rename({'Demand (MW)': ba})  # Rename for easier summation
        demand_list.append(hourly_demand_xr[ba])

    # Combine all demand data into one xarray DataArray
    combined_demand = xr.concat(demand_list, dim='BA')

    # Sum the demands across the BAs
    regional_hourly_demand = combined_demand.sum(dim='BA')
    # create demand unit
    from assetra.units import DemandUnit

    builder.add_unit(
        DemandUnit(
            id=unit_count,
            hourly_demand=regional_hourly_demand
        )
    )
    unit_count += 1
    return unit_count 

def getRegional_gens(Full_Generation_Portfolio, region_name): 
    Generation_Portfolio = Full_Generation_Portfolio[Full_Generation_Portfolio['Region'].isin([region_name])]
    solar = Generation_Portfolio.loc[Generation_Portfolio['Technology'].isin(['Solar Photovoltaic', 'Solar Thermal without Energy Storage', 'Solar Thermal with Energy Storage']), ['Summer Capacity (MW)', 'Technology', 'Latitude', 'Longitude', 'Plant Code', 'Nameplate Capacity (MW)', 'Region']]
    wind = Generation_Portfolio.loc[Generation_Portfolio['Technology'] == 'Onshore Wind Turbine', ['Summer Capacity (MW)', 'Technology', 'Latitude', 'Longitude', 'Plant Code', 'Nameplate Capacity (MW)', 'Region']]
    #not including pumped storage right now 
    storage = Generation_Portfolio.loc[
    Generation_Portfolio['Technology'].isin(['Batteries', 'Hydroelectric Pumped Storage']), 
    ['Summer Capacity (MW)', 'Technology', 'Latitude', 'Longitude', 'Plant Code', 'Nameplate Capacity (MW)', 'Region']]
    hydro = Generation_Portfolio.loc[Generation_Portfolio['Technology'] == 'Conventional Hydroelectric', ['Summer Capacity (MW)', 'Generator ID', 'Technology', 'Latitude', 'Longitude', 'Plant Code', 'Nameplate Capacity (MW)', 'Region']]
    # Group by 'ORIS Plant Code' and sum the 'Capacity (MW)'
    hydro = hydro.groupby('Region').agg({
        'Nameplate Capacity (MW)': 'sum', 
        'Summer Capacity (MW)': 'sum',
        'Technology':'first'    # Assuming region is the same for rows with the same 'ORIS Plant Code'
    }).reset_index()
    #hydro['Plant Code'] = hydro['Plant Code'].astype('Int64')
    thermal = Generation_Portfolio[
        ~Generation_Portfolio['Technology'].isin(['Solar Photovoltaic', 'Solar Thermal without Energy Storage', 'Solar Thermal with Energy Storage',
                                                 'Onshore Wind Turbine','Batteries', 'Hydroelectric Pumped Storage', 'Conventional Hydroelectric'])]
    return Generation_Portfolio, solar, wind, storage, hydro, thermal 

def compute_RH(annual_weather_dataset): 
    e = 0.622 #molecturlar wieght ratio of water to dry air
    #Compute saturation vapor pressure es (T2) using a simplified version of the Goff-Gratch equation
    # Temperature (T2) is assumed to be in Kelvin, and the output pressure (es) is in Pa.
    T2 = annual_weather_dataset['T2'] 
    PSFC = annual_weather_dataset['PSFC']
    Q2 = annual_weather_dataset['Q2']
    es=6.1078 * 100 * np.exp(17.27 * (T2 - 273.15) / (T2 - 35.85))
    qsat=(e * es) / (PSFC - (1 - e) * es)
    annual_weather_dataset['RH']=(Q2 / qsat)
    return annual_weather_dataset

def getWeather_data(gcm_full, year): 
    # load processed power generation dataset (solar cf, wind cf)
    cf_data = xr.open_dataset(f'/nfs/turbo/seas-mtcraig-climate/WRFDownscaled/{gcm_full}/Annual_Solar_Wind/Full_Solar_Wind_CapacityFactors.nc')
    pow_gen_dataset = cf_data.sel(Times = slice(np.datetime64(f'{year}-09-01T00:00:00'), np.datetime64(f'{year+1}-09-01T00:00:00'))).rename({'Times':'time'})
    annual_weather_dataset = xr.open_dataset(f'/nfs/turbo/seas-mtcraig-climate/WRFDownscaled/{gcm_full}/{year}/regrid_{year}_ssp370_d02.nc')
    #realign times
    start_time = pd.Timestamp(f'{year}-09-01 00:00')
    times = pd.date_range(start=start_time, periods=len(annual_weather_dataset.Times), freq='h')
    annual_weather_dataset['Times'] = times
    annual_weather_dataset = annual_weather_dataset.rename({'Times':'time'})
    #computing relative humidty 
    annual_weather_dataset = compute_RH(annual_weather_dataset)
    return annual_weather_dataset, pow_gen_dataset

def get_nearest_hourly_profile(
    latitude: float,
    longitude: float,
    array: xr.DataArray
) -> xr.DataArray:
    """Return time series corresponding to the nearest coordinate in a
    WRF power generation data array.

    Args:
        latitude (float): Latitude relative to equator in degrees
        start_hour (datetime): Longitude relative to meridian in degrees
        array (xr.DataArray): "solar_capacity_factor", "wind_capacity_factor",
            or "temperature" or "relative humidity"

    Returns:
        xr.DataArray: Array with time dimension and datetime coordinates.
    """
    return array.sel(
            lat=latitude, 
            lon=longitude, 
            method="nearest"
        ).squeeze(drop=True)

def get_wrf_power_generation_solar_cf(
    latitude: float,
    longitude: float) -> xr.DataArray:
    return get_nearest_hourly_profile(latitude, longitude, pow_gen_dataset["Solar_CF"])

def get_wrf_power_generation_wind_cf(
    latitude: float,
    longitude: float) -> xr.DataArray:
    return get_nearest_hourly_profile(latitude, longitude, pow_gen_dataset["Wind_CF"])

def get_wrf_power_generation_temperature(
    latitude: float,
    longitude: float) -> xr.DataArray:
    return get_nearest_hourly_profile(latitude, longitude, annual_weather_dataset["T2"])

def get_wrf_power_generation_rh(
    latitude: float,
    longitude: float) -> xr.DataArray:
    return get_nearest_hourly_profile(latitude, longitude, annual_weather_dataset["RH"])

def get_wrf_power_generation_psfc(
    latitude: float,
    longitude: float) -> xr.DataArray:
    return get_nearest_hourly_profile(latitude, longitude, annual_weather_dataset["PSFC"])
import pandas as pd

# load temperature dependent outage rate (tdfor) table
tdfor_table_file = Path("temperature_dependent_outage_rates.csv")
tdfor_table = pd.read_csv(tdfor_table_file, index_col=0)
tdfor_table = tdfor_table / 100 # percentages stored as integers

# create mapping table for tdfor table
tech_categories = {
    "CC" : ['Natural Gas Fired Combined Cycle', ],
    "CT" : ['Natural Gas Fired Combustion Turbine','Landfill Gas'],
    "DS" : ['Municipal Solid Waste','Fossil Waste', 'Natural Gas Internal Combustion Engine', 'Non-Fossil Waste', 'Other Natural Gas'],#"Natural Gas Internal Combustion Engine"],
    "ST" : ['Conventional Steam Coal',  'Natural Gas Steam Turbine', 'Petroleum Liquids'],#"Natural Gas Steam Turbine"],
    "NU" : ["Nuclear", 'Fuel Cell'],
    "HD" : ['Conventional Hydroelectric','Hydroelectric Pumped Storage', 'Biomass','Geothermal', 'Other Waste Biomass', 'Wood/Wood Waste Biomass']
                   # add "Hydroelectric Pumped Storage" in HD on next build out ,
    #"Solar Thermal with Energy Storage","Wood/Wood Waste Biomass"]
}

# create mapping from technology to category
tech_mapping = {tech : cat for cat, techs in tech_categories.items() for tech in techs}

def get_hourly_forced_outage_rate(hourly_temperature: xr.DataArray, technology: str) -> xr.DataArray:
    # index tdfor table by tech
    tdfor_map = tdfor_table[tech_mapping.get(technology, "Other")]
    map_temp_to_for = lambda hourly_temperature: tdfor_map.iloc[
            tdfor_map.index.get_indexer(hourly_temperature, method="nearest")
        ]
    return xr.apply_ufunc(
        map_temp_to_for,
        hourly_temperature
    ).rename("hourly_forced_outage_rate")

#Combusiton Turbine Derates
def derateCTs(unitMet):
    cct = .0083 #units of 1/degC
    availCapac = -cct*unitMet + 1.15
    # Clip values to be within the range [0, 1]
    availCapac = xr.where(availCapac < 0, 0, availCapac)
    availCapac = xr.where(availCapac > 1, 1, availCapac)
    return availCapac
#read ins for RC and DC derates 
def readCSVto2dList(fileNameWithDir):
    with open(fileNameWithDir,'r') as f:
        f = csv.reader(f)
        f = list(f)
    return f
def loadCoeffs():
    baseDir = os.path.join('/home/mlchris/WRFProcessing','ASSETRA')
    coalRC = readCSVto2dList(os.path.join(baseDir,'pcrecap.csv'))
    coalDC = readCSVto2dList(os.path.join(baseDir,'pcdccap.csv'))
    ngRC = readCSVto2dList(os.path.join(baseDir,'ngrecap.csv'))
    ngDC = readCSVto2dList(os.path.join(baseDir,'ngdccap.csv'))
    return coalRC,coalDC,ngRC,ngDC 
#Load derate coefficients for coal & NG RC & DC units (using Aviva Loew paper)
coalRC,coalDC,ngRC,ngDC = loadCoeffs()
def cToF(c):
    return c*9/5 + 32
def getCoeffs(coeffs,varNames,design):
    col = coeffs[0].index(design)
    rowLabels = [row[0] for row in coeffs]
    vals = list()
    for v in varNames: vals.append(float(coeffs[rowLabels.index(v)][col].split(' ')[0]))
    return vals 

#Table S10 = coal RC, S15 = NGCC RC. Designs: 90-70, 95-75, 100-80.
def derateRC(coeffs,temp, rh ,rcDesign):
    varNames = ['Air Temperature (F)','Relative Humidity (%)','Interaction Term',
                'Intercept']
    coeffVals = getCoeffs(coeffs,varNames,rcDesign)
    availCapac = (cToF(temp)*coeffVals[0] + rh*coeffVals[1] + 
            cToF(temp)*rh*coeffVals[2] + coeffVals[3])
    availCapac = xr.where(availCapac < 0, 0, availCapac)
    availCapac = xr.where(availCapac > 1, 1, availCapac)
    return availCapac
#Table S12 = coal DC, S17 = NGCC DC. Designs: ITD 25, ITD 35, ITD 45, ITD 55.
def derateDC(coeffs,temp,pressure,dcDesign):
    varNames = ['Air Temperature (F)','Air Pressure (psia)','Intercept']
    coeffVals = getCoeffs(coeffs,varNames,dcDesign)
    availCapac = (cToF(temp)*coeffVals[0] + pressure*coeffVals[1] + coeffVals[2])
    availCapac[availCapac<0],availCapac[availCapac>1] = 0,1
    return availCapac


def load_gens(unit_count, thermal, solar, wind, storage, hydro, monthly_expected_generation_xr): 
    # Iterate over each generator in the thermal dataframe
    for _, generator in thermal.iterrows():
        # get hourly temperature
        hourly_temperature = get_wrf_power_generation_temperature(
            generator["Latitude"],
            generator["Longitude"]
        )
        hourly_rh = get_wrf_power_generation_rh( 
             generator["Latitude"],
            generator["Longitude"]
        ) * 100 #to get %
        hourly_pressure = get_wrf_power_generation_psfc( 
             generator["Latitude"],
            generator["Longitude"]
        ) * 0.01 * (1/68.9475729318)# converting to hPa from Pa then to psia from hpa 

        # K to C conversion 
        hourly_temperature = hourly_temperature - 273.15
        # map temperature to hourly forced outage rate
        hourly_forced_outage_rate = get_hourly_forced_outage_rate(hourly_temperature, generator["Technology"])

        rc_dc = ['Natural Gas Steam Turbine', 'Natural Gas Fired Combined Cycle', 'Conventional Steam Coal']
        plant_type = generator['Technology']
        cooling_type = generator['Cooling Type 1']
        capacity_mw = generator["Summer Capacity (MW)"]

        if plant_type == 'Natural Gas Fired Combustion Turbine':
            derate = derateCTs(hourly_temperature)
            # get hourly capacity
            hourly_capacity = ( 
                derate * xr.ones_like(hourly_temperature).rename("hourly_capacity") 
                * generator["Summer Capacity (MW)"]
            )
        elif plant_type in rc_dc:
            if cooling_type in ('RC', 'ON'):
                if plant_type == 'Natural Gas Fired Combined Cycle':
                    derate = derateRC(ngRC, hourly_temperature, hourly_rh, '95-75')
                else:
                    derate = derateRC(coalRC, hourly_temperature, hourly_rh, '95-75')
                hourly_capacity = ( 
                    derate * xr.ones_like(hourly_temperature).rename("hourly_capacity") 
                    * generator["Summer Capacity (MW)"]
                )
            elif cooling_type == 'DC':
                if plant_type == 'Natural Gas Fired Combined Cycle':
                    derate = derateDC(ngDC, hourly_temperature, hourly_pressure, 'ITD 45')
                else:
                    derate = derateDC(coalDC, hourly_temperature, hourly_pressure, 'ITD 45')
                hourly_capacity = ( 
                    derate * xr.ones_like(hourly_temperature).rename("hourly_capacity") 
                    * generator["Summer Capacity (MW)"]
                )
            else:
                hourly_capacity = ( 
                    xr.ones_like(hourly_temperature).rename("hourly_capacity") 
                    * generator["Summer Capacity (MW)"]
                )
                # This line should never be reached given the existing options
                # raise ValueError(f"Unsupported combination of cooling type {cooling_type} and plant type {plant_type}")

        else:
            # get hourly capacity
            hourly_capacity = ( 
                xr.ones_like(hourly_temperature).rename("hourly_capacity") 
                * generator["Summer Capacity (MW)"]
            )
            #print(hourly_capacity)

        # create assetra energy unit
        thermal_unit = StochasticUnit(
                id=unit_count,
                nameplate_capacity=generator["Summer Capacity (MW)"],
                hourly_capacity=hourly_capacity,
                hourly_forced_outage_rate=hourly_forced_outage_rate
            )

        unit_count += 1
        # add unit to energy system
        builder.add_unit(thermal_unit)
    print('Thermal Loaded') 

    #solar  generation load in 
    for _, generator in solar.iterrows():
        #print(generator) 
        # get hourly temperature
        hourly_temperature = get_wrf_power_generation_temperature(
            generator["Latitude"],
            generator["Longitude"]
        )
        # get hourly temperature
        hourly_capacity = get_wrf_power_generation_solar_cf(
            generator["Latitude"],
            generator["Longitude"]
        ) * generator["Summer Capacity (MW)"]

        # map temperature to hourly forced outage rate
        hourly_temperature = hourly_temperature - 273.15 #K to C 
        hourly_forced_outage_rate = get_hourly_forced_outage_rate(hourly_temperature, generator["Technology"])

        # create assetra energy unit
        solar_unit = SolarandWindUnit(
                id=unit_count,
                nameplate_capacity=generator["Summer Capacity (MW)"],
                hourly_capacity=hourly_capacity.clip(max=generator["Summer Capacity (MW)"]),
                hourly_forced_outage_rate=hourly_forced_outage_rate
            )
        unit_count += 1

        # add unit to energy system
        builder.add_unit(solar_unit)
    print('Solar Loaded') 

    #wind generation load in 
    # add wind
    for _, generator in wind.iterrows():
        # get hourly temperature
        hourly_temperature = get_wrf_power_generation_temperature(
            generator["Latitude"],
            generator["Longitude"]
        )
        # get hourly capacity
        hourly_capacity = get_wrf_power_generation_wind_cf(
            generator["Latitude"],
            generator["Longitude"]
        ) * generator["Summer Capacity (MW)"]

        # map temperature to hourly forced outage rate
        hourly_temperature = hourly_temperature - 273.15 #K to C 
        hourly_forced_outage_rate = get_hourly_forced_outage_rate(hourly_temperature, generator["Technology"])

        # create assetra energy unit
        wind_unit = SolarandWindUnit(
                id=unit_count,
                nameplate_capacity=generator["Summer Capacity (MW)"],
                hourly_capacity=hourly_capacity,
                hourly_forced_outage_rate=hourly_forced_outage_rate
            )
        unit_count += 1
        print(unit_count)
        # add unit to energy system
        builder.add_unit(wind_unit)
    print('Wind Loaded')
    #print(hourly_forced_outage_rate)
    
    #storage load in 
    for _, generator in storage.iterrows():
        storage_unit = StorageUnit(
            id=unit_count,
            nameplate_capacity=generator["Summer Capacity (MW)"],
            charge_rate=generator["Summer Capacity (MW)"],
            discharge_rate=generator["Summer Capacity (MW)"],
            charge_capacity=generator["Summer Capacity (MW)"],
            roundtrip_efficiency=0.85 #assumed
        )
    
        
        unit_count += 1
        print(unit_count)
        # add unit to energy system
        builder.add_unit(storage_unit)
    print('Storage Loaded') 

    
      
    #hydro load in 
    for _, generator in hydro.iterrows():
        global region_name
        # create assetra energy unit
        hydro_unit = HydroUnit(
            id=unit_count,
            nameplate_capacity=generator["Summer Capacity (MW)"],
            monthly_expected_generation=monthly_expected_generation_xr[region_name],
            #hourly_forced_outage_rate=hourly_forced_outage_rate,  # optional
        )

            
        unit_count += 1
        print(unit_count)
        # add unit to energy system
        builder.add_unit(hydro_unit)

    print('Hydro Loaded')  

        
    return unit_count 







#Initialize System 
builder = EnergySystemBuilder()

# every unit must have a unique id
unit_count = 0

#hydro generation read in 
monthly_expected_generation_xr = Hydro_ReadIn(gcm_full, gcm, year)

if region_name == 'CAMX': 
    ba_list = ['CISO', 'BANC', 'LDWP']
elif region_name == 'DSW':
    ba_list = ['AZPS', 'TEPC', 'WALC', 'PNM', 'SRP', 'EPE']
elif region_name == 'NWPP':
    ba_list = ['NEVP', 'PSCO', 'PACE', 'PACW', 'BPAT', 'CPD', 'IPCO', 'PGE', 'PSEI', 'SCL', 'AVA', 'MONT', 'WACM']
    
#regional demand 
unit_count = getRegional_demand(gcm, ba_list, unit_count) 
print('Got Demand')

#generation 
Generation_Portfolio, solar, wind, storage, hydro, thermal = getRegional_gens(Full_Generation_Portfolio, region_name) 
print('Made Profiles')

#weather 
annual_weather_dataset, pow_gen_dataset = getWeather_data(gcm_full, year) 
print('Got datasets') 

#load in gens
unit_count = load_gens(unit_count, thermal, solar, wind, storage, hydro, monthly_expected_generation_xr)
print('units loaded') 

#System Set Up 
system_dir = Path(f"/nfs/turbo/seas-mtcraig-climate/Martha_Research/ASSETRA_models/ARSTesting/{year}/{gcm}/{region_name}/{region_name}_{gcm}_{fleet_file[:-4]}_{year}_MLPDemand_SINGLEHYDRO_solarwind")
energy_system = builder.build()
energy_system.save(system_dir)