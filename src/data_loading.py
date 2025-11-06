"""
Data loading utilities for Census Income dataset.
"""

import pandas as pd
import yaml
from pathlib import Path
from typing import Tuple, Optional


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_data(
    train_path: str,
    test_path: str,
    metadata_path: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and test datasets.
    
    Parameters:
    -----------
    train_path : str
        Path to training data CSV file
    test_path : str
        Path to test data CSV file
    metadata_path : str, optional
        Path to metadata file
        
    Returns:
    --------
    train_df : pd.DataFrame
        Training dataset
    test_df : pd.DataFrame
        Test dataset
    """
    # Load data
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)
    
    # Load column names from metadata if available
    if metadata_path and Path(metadata_path).exists():
        # TODO: Parse metadata to get column names
        pass
    
    return train_df, test_df


def get_column_names() -> list:
    """
    Get column names for the Census Income dataset.
    
    Returns:
    --------
    column_names : list
        List of column names
    """
    # Based on metadata.txt
    columns = [
        'age', 'class_of_worker', 'detailed_industry_recode',
        'detailed_occupation_recode', 'education', 'wage_per_hour',
        'enroll_in_edu_inst_last_wk', 'marital_stat', 'major_industry_code',
        'major_occupation_code', 'race', 'hispanic_origin', 'sex',
        'member_of_labor_union', 'reason_for_unemployment',
        'full_or_part_time_employment_stat', 'capital_gains', 'capital_losses',
        'dividends_from_stocks', 'tax_filer_stat',
        'region_of_previous_residence', 'state_of_previous_residence',
        'detailed_household_and_family_stat',
        'detailed_household_summary_in_household', 'instance_weight',
        'migration_code_change_in_msa', 'migration_code_change_in_reg',
        'migration_code_move_within_reg', 'live_in_this_house_1_year_ago',
        'migration_prev_res_in_sunbelt', 'num_persons_worked_for_employer',
        'family_members_under_18', 'country_of_birth_father',
        'country_of_birth_mother', 'country_of_birth_self', 'citizenship',
        'own_business_or_self_employed',
        'fill_inc_questionnaire_for_veterans_admin', 'veterans_benefits',
        'weeks_worked_in_year', 'year', 'income'
    ]
    return columns

