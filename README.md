# This repository conducts Exploratory Data Analysis with the FatigueSet Dataset

## Background

#### FatigueSet: A Multi-model Dataset for Modeling Mental Fatigue and Fatiguability

This dataset contains time series data and performance metrics obtained during a study conducted by the authors of the FatigueSet paper who seeked to investigate the relationship between cognitive performance and one's own awareness of their performance. It tracks features from 12 people going through 3 sessions each of baseline recordings, physical and mental stimulation, and multiple measurement periods, where participants self-report physical and mental levels of fatigue and take two cognitive tests. You can read the full paper [here](https://link.springer.com/chapter/10.1007/978-3-030-99194-4_14).

## Objective:

The original objective of this repository was to use the FatigueSet dataset to train a model to predict cognitive fatigue. Many different types of data were analyzed, however, focus shifted away from model building towards strictly data analysis. Two different types of data were analyzed: EEG (electroencephalogram) data, and HRV (heart rate variability) data. Data wrangling and exploratory data analysis were conducted to examine correlations and statistical significance.

## How to use the Repository:

1. Navigate to the folder you would like the repository to be cloned into. Then, clone the repository and then navigate into it.
    ```
    git clone https://github.boozallencsn.com/Krinn-Matthew/cognitive-fatigue.git
    cd cognitive-fatigue
    ```
    After cloning, your file structure should be the following:
    
    ```
    cognitive-fatigue/
    |    
    ├── archive/
    |   ├── eeg.ipynb
    |
    ├── data/
    |   ├── .gitkeep
    |   
    ├── hrv.ipynb
    |   
    ├── utils.py
    |   
    ├── README.md
    ├── environment.yml
    ├── .gitignore
    ```

2. Download the compressed [FatigueSet dataset](https://esense.io/datasets/fatigueset/). Unzip the downloaded `fatigueset.zip` and move the unzipped `fatigueset` folder into the `data` directory in the repository (`cognitive-fatigue/data`). Make sure that the unzipped folder is named `fatigueset`.

    Note: `fatigueset` contains the whole 13 hours of features for each person, but we are not analyzing all of that data. Because of this, a later step will spawn folders named `fatigueset_hrv_data` and `fatigueset_eeg_data`, depending on the type of data you want to analyze (which notebook you use).


3. If required, install conda onto your computer.

    First, install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) onto your system. Then, restart your IDE / terminal, `cd` back into the repository, and run `conda --version` to ensure a proper installation.


4. 
    Use conda and the provided `.yml` file to create the analysis environments.
    ```
    conda env create -f environment.yml
    ```

5. Spawn the relevant data and begin exploration


    One of the first cells in each notebook defines functions to grab data from `fatigueset` and organize it into a different folder. You can make changes to the `mapping` dictionary in each function to grab different csvs from `fatigueset`. Each *key* is the name of the csv in a person / session's data folder in `fatigueset`, and the associated *value* is the prefix assigned to each column in the csv. Prefixing is required as many columns across different csvs have the same column names (ex. Duration).
    
    Another cell in each notebook spawns in the relevant data. Simply run the cell and the `fatigueset_{notebook_name}_data` folder will be produced in `cognitive-fatigue/data`. Note that the initialization function does a multitude of different things, including:
    - Renames each person's sessions folders in `fatigueset` from integers to `low`, `medium`, and `high` based on the mapping provided by `fatiguset/metadata.csv`.
    - Renames duplicate named events in each person and session's `exp_markers.csv` to avoid errors down the line (ex. Renames multiple `start_nback` to `start_nback_1`, `start_nback_2`, ...).
    - Fills in NaN values in `exp_markers.csv`. Sometimes, there is not a timestamp with an associated event (ex. person 1, session low does not have a `start_nback_3` timestamp). To fix this, NaNs were replaced with the timestamp of the event prior with an additional 10 seconds to ensure the monotonicity of the dataframe.
    - Moves relevant metadata files from `fatigueset` (ones prefixed with `exp_`) into a person-specific metadata folder in `fatiguset_{notebook_name}_data`.
    - Aggregates and spawns data from relevant files of `fatigueset` into person-specific csv's.
     
    After running the initialization function `initialize_aggregated_data()`, the structure of `data` should now be the following:

     ```
    cognitive-fatigue/
    |
    ├── data/
    |   ├── fatiguset
    |   ├── fatiguset_{notebook_name}_data
    |   |   ├── 01_metadata/
    |   |   ├── 02_metadata/
    |   |   ├── ...
    |   |   ├── 01_data.csv
    |   |   ├── 02_data.csv
    |   |   ├── ...
    |   ├── .gitkeep
    |   
    ...
     ```

    The data was reformatted from `fatigueset` in order to more easily combine similar dataframes and explore the data. Now, each person has a metadata folder that tracks data obtained during measurement periods and a csv with their time series data.

## Archive Folder / eeg.ipynb:

Analysis began with eeg data, however, the data was too complex to fully analyze. eeg.ipynb was moved into `archive/` and is available for you to mess with, as analysis of heart rate variability data in `hrv.ipynb` became the main focus of the repository.

## Findings from hrv.ipynb:

### Regarding Dataset Study Protocol:

- The different time periods in the study (baseline period, physical and mental activity periods, and measurement periods) were not long enough to allow a full analysis of a person's HRV as different conditions are applied and removed.

### Aggregated HRV by Session and Period Visuals:

- While baseline, M1, and M3 mean HRV for Low and Medium sessions was exactly the same, the High session exhibited a lower HRV, even in the periods before physical activity (which is the only difference between sessions). The study did not explicitly say if / when subjects knew which physical intensity level they were doing in the moment. However, the deviation could be due to subjects expecting a harder workout, which could lead the sympathetic to lower HRV for the fight-or-flight response.

- M2 Low session HRV increased from physical activity into M2. People exhibit higher HRV in less-stressful situations as the parasympathetic system becomes more active. Because the physical activity prescribed for the Low session is only walking, this could be why we see a slight increase in HRV after physical activity.

- The strongest decline in HRV after physical activity was in the High intensity period, dropping HRV to well below baseline levels.

- HRV varied over the course of the experiment due to the introduction of physical and mental activity sessions. However, there is not a strong correlation between the session type and mean HRV across the whole session.

### Significance Testing:

- Barnard's Exact Test showed that those that have a below-average HRV difference from baseline in the Low Session continue to have a below-average HRV difference from baseline in the Medium Session. The finding does not follow for Low and High Sessions, or Medium and High Sessions. We see this most likely because the difference in energy expenditure between Low and Medium sessions is much smaller compared to the difference between Medium and High sessions. In a future study, consider making the different exercises an equal difference apart in terms of METs

- **Findings from the fatigueset paper were replicated. The data from fatigueset is not comprehensive enough to prove a significant relationship between self-reported mental fatigue scores and performance on tasks during measurement periods.**

  - Mean self-reported levels of mental fatigue in measurement period 3 were statistically significant when compared to earlier periods (low : high, p=0.0002; medium : high, p=0.0204). This means we could predict if mental fatigue scores should increase or decrease by determining if mental activity was done prior to the measurement. However, the relevance is limited if there isn't a relationship between actual cognitive performance and people's perception of their performance, which is what the dataset exemplifies.

  - The variability in accuracy and response time of tasks was not high enough to be able to be correlated with HRV, as well as with the self-reported scores. In a future study, there should be more variability in these metrics, meaning we seek more difficult tests. This would allow us to determine if there actually is a correlatation between self-reported scores and cognitive performance.
  
  - There is a statistically significant difference in self-reported physical (U=220.0, p=0.0005) and mental fatigue (U=237.0, 0.0009) between groups over or under a certain threshold of HRV difference from baseline. However, there is not a significant difference in performance on tasks. This verifies the findings of the preliminary results from the dataset article.

### Modeling HRV and Fatigue Interaction

- Increase in hrv_diff feature (difference between current mean HRV and baseline mean HRV) is correlated with a decrease in self-reported mental fatigue, self-reported physical fatigue, and total_response_time.

- A threshold separating hrv_diff by direction and magnitude is better for predicting self-reported physical and mental scores than a threshold separating only by magnitude.

- Using hrv_threshold over hrv_diff almost doubled r2 metric. Using happiness and alertness scores with hrv_threshold resulted in another r2 doubling.

# Conclusion:

The preliminary findings of the Fatiguset paper were verified, but the lack of relationship between one's own performance and perception of performance could be due to problems in the study protocol. A future study should incorporate longer time periods to better distinguish between features that come from the *transition between time periods* as opposed to the *current time period itself*. It should also use more difficult tests to obtain metrics with higher variability to better correlate accuracy or response time with other features.

# Acknowledgments:

FatigueSet: A Multi-modal Dataset for Modeling Mental Fatigue and Fatigability      
Manasa Kalanadhabhatta, Chulhong Min, Alessandro Montanari and Fahim Kawsar    
In 15th International Conference on Pervasive Computing Technologies for Healthcare (Pervasive Health), December 6–8, 2021