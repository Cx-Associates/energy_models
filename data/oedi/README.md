# End Use Load Profiles for the U.S. Building Stock
This collection of datasets describes the energy consumption of the U.S. residential and commercial building stock. The data are broken down first by building type (single family home, office, restaurant, etc.), then by end-use (heating, cooling, lighting, etc.) at a 15-minute time interval. For details on how the datasets were created and validated, please visit the [project website](https://www.nrel.gov/buildings/end-use-load-profiles.html).

## Directory Structure
```
nrel-pds-building-stock                                                                     # top-level directory inside the NREL oedi-data-lake bucket
├── end-use-load-profiles-for-us-building-stock                                             # directory for all End Use Load Profiles and End Use Savings Shapes
│   ├── README.md                                                                           # the file you are reading right now
│   └── 2022                                                                                # year the dataset was published
│       ├── EUSS_ResRound1_Technical_Documentation                                          # technical documentation
│       └── resstock_amy2018_release_1                                                      # name of the dataset
│           ├── building_energy_models                                                      # models used to generate timeseries data
│           │   ├── upgrade=<upgrade_id>                                                    # by upgrade_id - refer to upgrades_lookup.json for upgrade id descriptions
│           │   │   ├── <building_id>-up<upgrade_id>.zip                                    # model files for specific building_id and upgrade_id - zip file includes in.xml and schedules.csv files
│           ├── geographic_information                                                      # geographic shapefiles used for mapping this dataset
│           │   ├── map_of_pumas_in_census_region_1_northeast.geojson                       # map of U.S. Census Public Use Microdata Area in Census Region 1 Northeast
│           │   ├── map_of_pumas_in_census_region_2_midwest.geojson                         # map of U.S. Census Public Use Microdata Area in Census Region 2 Midwest
│           │   ├── map_of_pumas_in_census_region_3_south.geojson                           # map of U.S. Census Public Use Microdata Area in Census Region 3 South
│           │   ├── map_of_pumas_in_census_region_4_west.geojson                            # map of U.S. Census Public Use Microdata Area in Census Region 4 West
│           │   ├── map_of_us_states.geojson                                                # map of U.S. States
│           │   └── spatial_tract_lookup_table.csv                                          # mapping between census tract identifiers and other geographies
│           ├── metadata                                                                    # building characteristic information and annual energy consumption for each building id, formatted for machine readability
│           │   ├── upgrade<upgrade_id>.parquet                                             # annual energy consumption for each building id for each respective upgrade_id
│           │   └── baseline.parquet                                                        # building characteristics and annual energy consumption for each building in the baseline (existing stock) scenario
│           ├── metadata_and_annual_results                                                 # building characteristic information and annual energy consumption, formatted for human readability
│           │   ├── by_state                                                                # by U.S. States
│           │   ├── state=<state>                                                           # building_ids (dwelling units) separated by state
│           │   │   ├── csv                                                                 # csv file format 
│           │   │   │   ├─ <state>_baseline_metadata_and_annual_results.csv                 # building characteristics and annual energy consumption by end use and fuel type for each building id for baseline scenario
│           │   │   │   ├─ <state>_upgrade<upgrade_id>_metadata_and_annual_results.csv      # building characteristics and annual energy consumption by end use and fuel type for each building id for upgrade scenarios
│           │   │   └── parquet                                                             # parquet file format
│           │   │       ├─ <state>_baseline_metadata_and_annual_results.parquet             # building characteristics and annual energy consumption by end use and fuel type for each building id for baseline scenario
│           │   │       ├─ <state>_upgrade<upgrade_id>_metadata_and_annual_results.parquet  # building characteristics and annual energy consumption by end use and fuel type for each building id for upgrade scenarios
│           │   └── national                                                                # all building_ids in one file (larger files)
│           │       ├── csv                                                                 # csv file format 
│           │       │   ├─ baseline_metadata_and_annual_results.csv                         # building characteristics and annual energy consumption by end use and fuel type for each building id for baseline scenario
│           │       │   ├─ upgrade<upgrade_id>_metadata_and_annual_results.csv              # building characteristics and annual energy consumption by end use and fuel type for each building id for upgrade scenarios
│           │       └── parquet                                                             # parquet file format
│           │           ├─ baseline_metadata_and_annual_results.parquet                     # building characteristics and annual energy consumption by end use and fuel type for each building id for baseline scenario
│           │           ├─ upgrade<upgrade_id>_metadata_and_annual_results.parquet          # building characteristics and annual energy consumption by end use and fuel type for each building id for upgrade scenarios   
│           ├── timeseries_aggregates                                                       # timeseries data aggregated by climate zones, ISO/RTO regions, and by state
│           │   ├── by_ashrae_iecc_climate_zone_2004                                        # by ASHRAE climate zone
│           │   │   ├── upgrade=<upgrade_id>                                                # by building_id and upgrade_id
│           │   │   ├── up<upgrade_id>-<iecc_climate_zone>-<building_type>.csv              # aggregate timeseries energy consumption by end use and fuel type
│           │   ├── by_building_america_climate_zone                                        # by DOE Building America Climate Zone
│           │   │   ├── upgrade=<upgrade_id>                                                # by building_id and upgrade_id
│           │   │   ├── up<upgrade_id>-<ba_climate_zone>-<building_type>.csv                # aggregate timeseries energy consumption by end use and fuel type
│           │   ├── by_iso_rto_region                                                       # by Electric System ISO
│           │   │   ├── upgrade=<upgrade_id>                                                # by building_id and upgrade_id
│           │   │   ├── up<upgrade_id>-<iso_rto_region>-<building_type>.csv                 # aggregate timeseries energy consumption by end use and fuel type
│           │   └── by_state                                                                # by U.S. States
│           │       └── upgrade=<upgrade_id>                                                # by building_id and upgrade_id
│           │           └── state=<state>                                                   # aggregation for each state is in its own folder
│           │           └── up<upgrade_id>-<state>-<building_type>.csv                      # aggregate timeseries energy consumption by end use and fuel type
│           ├── timeseries_individual_buildings                                             # timeseries data for each individual building
│           │   └── by_state                                                                # by U.S. States
│           │       ├── upgrade=<upgrade_id>                                                # by building_id and upgrade_id
│           │           ├── state=<state>                                                   # timeseries of dwelling units are organized by state that building_id resides in
│           │           ├── <building_id>-<upgrade_id>.parquet                              # timeseries file of energy consumption by end use and fuel type
│           ├── weather                                                                     # weather data used in EUSS simulations
│           │   ├── state=<state>                                                           # by U.S. States
│           │       ├── <location_id>_<year>.csv                                            # weather data sorted by location id (GISJOIN identifier)
│           ├── data_dictionary.tsv                                                         # dictionary of building characteristics available for each dwelling unit in the EUSS dataset
│           ├── enumeration_dictionary.tsv                                                  # dictionary of available options for each building characteristic
│           └── upgrades_lookup.json                                                        # upgrade_id descriptions by number (also see technical documentation)
│             ...                                                                            ...
│   │   ...                                                             ...
│   └── 2021                                                            # year the dataset was published
│       └── comstock_amy2018_release_1                                  # name of the d
│           ├── citation.txt                                            # citation for this datasetataset
│           ├── README.md                                               # description of dataset and updates since last published version
│           ├── data_dictionary.tsv                                     # column names, units, enumerations, and descriptions
│           ├── enumeration_dictionary.tsv                              # mapping between enumeration name and description
│           ├── upgrade_dictionary.tsv                                  # mapping between upgrade identifier and upgrade name and description
│           ├── building_energy_models                                  # OpenStudio models used to generate timeseries data
│           │   ├── <building_id>-up<upgrade_id>.osm.gz                 # by building_id and upgrade_id
│           ├── occupancy_schedules                                     # (Residential only) Occupancy driven schedules for various enduses. Required to run the osm for residential
│           │   ├── <building_id>-up<upgrade_id>.csv.gz                 # by building_id and upgrade_id
│           ├── correction_factors                                      # (Residential only) correction factors for the output correction model
│           │   ├── correction_factors_2018.csv                        # factors may be used to adjust certain end-uses post-simulation
│           ├── geographic_information                                  # geographic shapefiles used for mapping this dataset
│           │   ├── map_of_pumas_in_census_region_1_northeast.geojson  # map of U.S. Census Public Use Microdata Area in Census Region 1 Northeast
│           │   ├── map_of_pumas_in_census_region_2_midwest.geojson    # map of U.S. Census Public Use Microdata Area in Census Region 2 Midwest
│           │   ├── map_of_pumas_in_census_region_3_south.geojson      # map of U.S. Census Public Use Microdata Area in Census Region 3 South
│           │   ├── map_of_pumas_in_census_region_4_west.geojson       # map of U.S. Census Public Use Microdata Area in Census Region 4 West
│           │   ├── map_of_us_states.geojson                           # map of U.S. States
│           │   └── spatial_tract_lookup_table.csv                     # mapping between census tract identifiers and other geographies
│           ├── metadata                                                # building characteristics and annual energy consumption for each building
│           │   └── metadata.parquet                                    # building characteristics and annual energy consumption for each building
│           ├── timeseries_aggregates                                   # sum of all profiles in a given geography by building type and end use
│           │   ├── by_ashrae_iecc_climate_zone_2004                    # by ASHRAE climate zone
│           │   │   ├── <iecc_climate_zone>-<building_type>.csv         # aggregate timeseries energy consumption by end use and fuel type
│           │   ├── by_building_america_climate_zone                    # by DOE Building America Climate Zone
│           │   │   |── <ba_climate_zone>-<building_type>.csv           # aggregate timeseries energy consumption by end use and fuel type
│           │   ├── by_county                                           # by U.S. County
│           │   │   ├── state=<state>                                   # Counties for a grouped into folder by state
│           │   │   │    |── <county_id>-<building_type>.csv            # aggregate timeseries energy consumption by end use and fuel type
│           │   ├── by_iso_rto_region                                   # by Electric System ISO
│           │   │   ├── <iso_rto_region>-<building_type>.csv            # aggregate timeseries energy consumption by end use and fuel type
│           │   ├── by_puma                                             # by U.S. Census Public Use Microdata Area
│           │   │   ├── state=<state>                                   # PUMAs for a grouped into folder by state
│           │   │   │    ├── <puma_id>-<building_type>.csv              # aggregate timeseries energy consumption by end use and fuel type
│           │   ├── by_state                                            # by U.S. States
│           │   │   ├── state=<state>                                   # Aggregation for each state is in its own folder
│           │   │   │    ├── <state>-<building_type>.csv                # aggregate timeseries energy consumption by end use and fuel type
│           ├── timeseries_aggregates_metadata                          # metadata information about the timeseries aggregates
│           │   └── metadata.tsv                                        # building characteristics and annual energy consumption for each building
│           ├── timeseries_individual_buildings                         # individual building timeseries data, partitioned several ways for faster queries
│           │   ├── by_county                                           # by U.S. County
│           │   │    ├── upgrade=<upgrade_id>                           # numerical identifier of upgrade (0 = baseline building stock)
│           │   │    │   └── county=<county_id>                         # gisjoin identifiers for counties and PUMAs, postal abbreviation for states
│           │   │    │       ├── <building_id>-<upgrade_id>.parquet     # individual building timeseries data
│           │   ├── by_puma_midwest                                     # by U.S. Census Public Use Microdata Area in Census Region 2 Midwest
│           │   ├── by_puma_northeast                                   # by U.S. Census Public Use Microdata Area in Census Region 1 Northeast
│           │   ├── by_puma_south                                       # by U.S. Census Public Use Microdata Area in Census Region 3 South
│           │   ├── by_puma_west                                        # by U.S. Census Public Use Microdata Area in Census Region 4 West
│           │   └── by_state                                            # by State
│           ├── weather                                                 # weather data used to run the building energy models to create datasets
│           │   ├── amy<year>                                           # weather data for a specific year (from NOAA ISD, NSRDB, and MesoWest)
│           │   │   ├── <location_id>_<year>.csv                        # by location, county gisjoin identifier
│           │   ├── tmy3                                                # weather data used for typical weather run
│           │   │   ├── <location_id>_tmy3.csv                          # by location, county gisjoin identifier

```

## Citation

Please use the citation found in `citation.txt` for each dataset when referencing this work.

## Dataset Naming
```
         <dataset type>_<weather data>_<year of publication>_release_<release number>
 example:    comstock        amy2018            2021         release_1
```
  - dataset type
    - resstock = residential buildings stock
    - comstock = commercial building stock
  - weather data
    - amy2018 = actual meteorological year 2018 (2018 weather data from NOAA ISD, NSRDB, and MesoWest)
    - tmy3 = typical weather from 1991-2005 (see [this publication](https://www.nrel.gov/docs/fy08osti/43156.pdf) for details)
  - year of publication
    - 2021 = dataset was published in 2021
    - 2022 = dataset was published in 2022
    - ...
  - release
    - release_1 = first release of the dataset during the year of publication
    - release_2 = second release of the dataset during the year of publication
    - ...

## Metadata
These are the building characteristics (age, area, HVAC system type, etc.) for each of the buildings
energy models run to create the timeseries data. Descriptions of these characteristics are
included in `data_dictionary.tsv`, `enumeration_dictionary.tsv`, and `upgrade_dictionary.tsv`.

## Aggregated Timeseries
Aggregate end-use load profiles by building type and geography that can be opened
and analyzed in Excel, python, or other common data analysis tools.
Each file includes the summed energy consumption for all buildings of the
specified type in the geography of interest by 15-minute timestep.

In addition to the timeseries data, each file includes a column named `models_used`
which represents the number of building energy models used to create the aggregate. There is also a column
named `units_represented` (for residential) or `floor_area_represented` (for commercial)  which indicates the
total number of dwelling units (for resedential) or floor area (for commercial) the aggregate represents.
The list of `building_ids` and the associated characteristics of those buildings can be obtained from the 
the `metadata.csv` file by filtering and selecting rows belonging to the aggregate. For example, to find the
`building_ids` associated with the county aggregate `g0100030-mobile_home.csv`, filter the `metadata.csv` to select
the rows where `in.county` =  `g0100030` and `in.geometry_building_type_recs` = `mobile_home`. 

## Aggregated timeseries file format 

The aggregated timeseries files follow this format: <geography_id>-<building_type>.csv

The files contain timeseries aggregate energy consumption for different end uses.

The first two columns contain the <geography_id> and <building_type>. The other key columns are:
- `timestamp`: The 15-minute ending timestamp in EST. 2018-01-01 00:30:00 represents time window from 2018-01-01 00:15:00 EST to 2018-01-01 00:30:00 EST.
- `models_used`: the number of simulated dwelling units (residential) or buildings (commercial) included in the aggregation.
- `units_represented` (residential): the number of dwelling units represented by these models and this aggregate.
- `floor_area_represented` (commercial): the floor area represented by these models and this aggregate, in square feet.
- `<enduse>.energy_consumption`: Energy consumed (in kWh) in the 15-minute time period by the enduse.

Example contents of `g01000100-fullservicerestaurant.csv`.

| puma     | in.building_type     | timestamp           | models_used | floor_area_represented |...| out.electricity.exterior_lighting.energy_consumption| ... |
|---       |---                   |---|---|---|---                  |---                |---                |
| G08000100| FullServiceRestaurant| 2018-01-01 00:15:00 | 5 | 433866.38|-| 32.79||
| G08000100| FullServiceRestaurant| 2018-01-01 00:30:00 | 5 | 433866.38|-|32.79||
| G08000100| FullServiceRestaurant| 2018-01-01 00:45:00 | 5 | 433866.38|-| 32.79||


### Aggregate timeseries are available by soem or all of the following geographies depending on the dataset:
-  U.S. States
-  ASHRAE Climate Zones
-  DOE Building America Climate Zones
-  Electric System ISOs
-  U.S. Census Public Use Microdata Area
-  U.S. Counties
    - **WARNING** in sparsely-populated counties, the number of models included in
    the aggregates may be very low (single digits), causing the aggregate load profiles
    to be unrealistic. When using county-level aggregates, we highly recommend that you
    review these numbers (included in the file header) and the load profiles before using them.


## Individual Building Timeseries
The raw individual building timeseries data.  **This is a large number of individual files!**
These data are partitioned (organized) in several different ways for comstock.nrel.gov and resstock.nrel.gov data viewers

Partitions:

-  U.S. States
-  PUMAS in Census Region 1 Northeast
-  PUMAS in Census Region 2 Midwest
-  PUMAS in Census Region 3 South
-  PUMAS in Census Region 4 West

## Building Energy Models
These are the building energy models, in [OpenStudio](https://www.openstudio.net/) format, that were run to create
the dataset. These building energy models use the [EnergyPlus](https://energyplus.net/) building simulation engine.
More recent ResStock dataset models are available in HPXML format.

## Geographic Information
Information on various geographies used in the dataset provided for convenience. Includes
map files showing the shapes of the geographies (states, PUMAs) used for partitioning
and a lookup table mapping between census tracts and various other geographies.

## Weather
These files show the key weather data used as an input to run the building energy models to create the dataset.
These data are provided in `CSV` format (instead of the EnergyPlus `EPW` format) for easier analysis. The timestamps in the weather files are all local standard time.

The **AMY** (actual meteorological year) files contain measured weather data from a specific year.
See [this publication](forthcoming_EULP_final_report) for details on how the AMY files were created.
The datasets created using AMY files are appropriate for applications where it
is important that the impacts of a weather event (for example, a regional heat wave) are realistically
synchronized across locations.

The **TMY3** (typical meteorological year) files contain typical weather from 1991-2005.
See [this publication](https://www.nrel.gov/docs/fy08osti/43156.pdf) for details on how the TMY3
files were created. The datasets created using TMY3 files are appropriate for applications
where a more "average" load profile is desired. **Note:** the weather data in the TMY3 files is NOT
synchonized between locations. One location could be experiencing a heat wave while another nearby has mild temperatures.

# File formats

The tables below illustrate the format of key files in the dataset.
**Note** that TSV file format was selected to allow commas in names and descriptions.

## data_dictionary.tsv

Describes the column names found in the metadata and timeseries data files.
All baseline building characteristics start with `in.` and all timeseries outputs start with `out.`
Where post-measure characteristics differ from baseline, these start with "upgarde."
Enumerations are separated with the `|` character.

| field_location  | field_name                                  | data_type | units | field_description | allowable_enumerations  |
|----------       |--------------                               |------     |---    |---                |---                      |
| metadata        | building_id                                 | int       |       |                   |                         |
| metadata        | job_id                                      | int       |       |                   |                         |
| metadata        | in.completed_status                         | bool      |       |                   |Success\|Fail            |
| metadata        | in.code_when_built                          | string    |       |                   |90.1-2004\|90.1-2007     |
| ...             |                                             |           |       |                   |                         |
| timeseries      | Time                                        | time      |       |                   |                         |
| timeseries      | TimeDST                                     | time      |       |                   |                         |
| timeseries      | TimeUTC                                     | time      |       |                   |                         |
| timeseries      | out.electricity.cooling.energy_consumption  | double    |       |                   |                         |
| timeseries      | out.electricity.cooling.energy_consumption  | double    |       |                   |                         |
| timeseries      | out.electricity.fans.energy_consumption     | double    |       |                   |                         |
| ...             |                                             |           |       |                   |                         |

## enumeration_dictionary.tsv

Expands the definitions of the enumerations used in the metadata files.

| enumeration | enumeration_description                                                             |
|----------   |--------------                                                                       |
| Success     | Simulation completed successfully, results should exist for this simulation         |
| Fail        | Simulation failed, no results or timeseries data should exist for this simulation   |
| 90.1-2004   | ASHRAE 90.1-2004                                                                    |
| 90.1-2007   | ASHRAE 90.1-2007                                                                    |
| ...         | ASHRAE 90.1-2007                                                                    |


## upgrade_dictionary.tsv

Expands the definitions of the upgrades.

| upgrade_id| upgrade_name    | upgrade_description                                   |
|---------- |--------------   |------                                                 |
| 0         | Baseline        | Baseline existing building stock                      |
| 1         | Low-e Windows   | Low-emissivity windows key assumptions here           |
| ...       |                 |                                                       |

## Changelog

### Update (2022-11-23)
- README updates 
- Completion of EUSS directory structure

### Update (2022-11-16)
- Completed upload of 2022.1 release - first round of residential savings shapes
- Technical documentation for 2022.1 edited
- README updates

### Update (2022-06-07)
- Corrected energy use intensity results in the residential AMY2018 and TMY3 individual building timeseries files. Previous results did not have unit conversions correctly applied. 

### Update (2022-06-02)
- Corrected demand results in the residential AMY2018 and TMY3 `metadata.parquet` and `metadata.tsv` files. Previous results were a factor of four too high.

### Update (2022-03-07)
- **MAJOR** Corrected all commercial aggregates in the AMY 2018 and TMY3 datasets to fix an issue where some end uses were blank and therefore the totals of the end uses didn't match the totals by fuel.
- **MAJOR** Corrected the units in the `data_dictionary.tsv` for the commercial AMY 2018 and TMY3 datasets; all energy consumption should be in kWh.
- Corrected the total site energy consumption column in `metadata.parquet` and `metadata.tsv`; values were previously million Btus, now in kWh, matching `data_dictionary.tsv`.
- Add QOI report columns and other metadata to the commercial `metadata.parquet` and `metadata.tsv` files in the AMY 2018 and TMY3 datasets.

### Update (2022-02-02)
- Add floor, wall, duct, and door area calculations to metadata.tsv, metadata.parquet, and data_dictionary.tsv files for the residential AMY 2018 and TMY3 datasets

### Update (2022-01-20)
- Addition of residential occupancy schedule files for all building models in the [2018 AMY](https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=nrel-pds-building-stock%2Fend-use-load-profiles-for-us-building-stock%2F2021%2Fresstock_amy2018_release_1%2Foccupancy_schedules%2F) and the [TMY3](https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=nrel-pds-building-stock%2Fend-use-load-profiles-for-us-building-stock%2F2021%2Fresstock_tmy3_release_1%2Foccupancy_schedules%2F) datasets
- Addition of the ComStock OpenStudio models for the [2018 AMY](https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=nrel-pds-building-stock%2Fend-use-load-profiles-for-us-building-stock%2F2021%2Fcomstock_amy2018_release_1%2Fbuilding_energy_models%2F) and [TMY3](https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=nrel-pds-building-stock%2Fend-use-load-profiles-for-us-building-stock%2F2021%2Fcomstock_tmy3_release_1%2Fbuilding_energy_models%2F) datasets
- Addition of State timeseries aggregations by building type for the commercial [AMY 2018](https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=nrel-pds-building-stock%2Fend-use-load-profiles-for-us-building-stock%2F2021%2Fcomstock_amy2018_release_1%2Ftimeseries_aggregates%2Fby_state%2F) and [TMY3](https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=nrel-pds-building-stock%2Fend-use-load-profiles-for-us-building-stock%2F2021%2Fcomstock_tmy3_release_1%2Ftimeseries_aggregates%2Fby_state%2F) datasets
- Addition of State timeseries aggregations by building type for the residential [AMY 2018](https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=nrel-pds-building-stock%2Fend-use-load-profiles-for-us-building-stock%2F2021%2Fresstock_amy2018_release_1%2Ftimeseries_aggregates%2Fby_state%2F) and [TMY3](https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=nrel-pds-building-stock%2Fend-use-load-profiles-for-us-building-stock%2F2021%2Fresstock_tmy3_release_1%2Ftimeseries_aggregates%2Fby_state%2F) datasets
- Reorganization of the "by_puma" timeseries aggregations into State folders to make searching easier for each of the residential and commercial datasets
- Reorganization of "by_county" timeseries aggregations into State folders to make searching easier for each of the residential and commercial datasets
- Addition of a few missing "by_puma" and building type timeseries aggregations in the commercial datasets
- For the residential timeseries aggregates, `<geography>-multi-family_with_5+_units.csv` was renamed to `<geography>-multi-family_with_5plus_units.csv` so files can be downloaded when clicked.
- Hawaii's AMY weather files were renamed to be consistent with the other weather files.
- README.md updated to include the weather file timestamps are in local standard time
- README.md updated to include a Changelog section
- Add QOI report columns to the residential `metadata.parquet` and `metadata.tsv` files in the AMY 2018 and TMY3 datasets.
- Addition of QOI fields to the `data_dictionary.tsv` for the residential AMY 2018 and TMY3 datasets
- Addition of QOI fields to the `data_dictionary.tsv` for the commercial AMY 2018 and TMY3 datasets

### Initial release (2021-10-21)
- Initial publication of dataset