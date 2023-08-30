import kmapper as km
import numpy as np
import pandas as pd
import geopandas as gpd
from statistics import mode
from tqdm import tqdm 
import matplotlib.pyplot as plt
import sys, os
import time
import sklearn
import pyclustering
from tda_adaptivecover_shim.adaptivecover_kmapper.adaptivecover_kmapper import KMapperAdaptiveCover
from mapper_xmean_cover import Cover
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils import read_sample
from tda_sklearn_pyclustering_shim.sklearn_pyclustering_shim.xmeans import PyclusteringXMeans
from kmapper.adapter import to_json

#manual inputs
data_path = "/Users/emariedelanuez/summer2023/tda_data/chicago_mayoral_acs_pct.csv"
mapper_instance = km.KeplerMapper()

columns_to_delete = ['125K_150K', 'NH_AMIN19', '75K_100K', 'SALGRIF_19', 'RO_G15', 'MAY19TP', 'normalized_area', 'HVAP', 'AMINVAP', 'H_NHPI', 'tot_pop_acs', 'H_OTHER19', 'H_WHITE', 'H_2MORE', 'ASIANVAP', '45K_50K', 'WILSON_19','2MOREVAP', 'WILS_G15', 'normalized_first_round_garcia', 'WVAP', 'NH_2MORE19', 'gt_25_pop', 'GARCIA_G15', '20K_25K', 'NH_ASIAN19', 'LESS_10K', 'gt_19_uninst_civs', 'NH_2MORE', 'full_text', 'NH_AMIN', 'H_ASIAN', 'NH_WHITE19', 'centroid_x', 'H_BLACK', 'DALEY_19', '50K_60K', 'TOTPOP19', 'TOTV_RO15', '10K_15K', 'TOTV_19', '40K_45K', 'MEND_19', 'civ_vap_acs', 'MAY19LL', 'tot_h_units_acs', 'NH_NHPI19', 'tot_hh_acs', 'VALLAS_19', '100K_125K', 'gt_16_working_pop', 'H_WHITE19', 'EMAN_G15', 'MCCART_19', 'Unnamed: 0', 'ENYIA_19', 'PRECK_19', 'FIORET_G15', 'COUNTY', 'poverty_ratio_ref_pop', 'TOTV_G15', '35K_40K', 'VAP', 'HISP', 'H_OTHER', '30K_35K', 'NH_WHITE', 'SHAPE_AREA', 'H_NHPI19', 'NH_OTHER19', 'FORD_19', 'H_BLACK19', 'tot_vap_acs', 'centroid_y', 'OTHERVAP', 'shape_len', 'H_AMIN19', 'shape_area', '15K_20K', 'TOTPOP', 'NH_BLACK', 'NH_BLACK19', '25K_30K', 'H_ASIAN19', 'gt_15_pop', 'JOYCE_19', '200K_MORE', 'TOTHH', 'FULL_TEXT', 'NH_OTHER', 'FIORETTI_1', 'RO_E15', 'SHAPE_LEN', 'cvap_acs', 'NHPIVAP', 'WALLS_G15', 'HISP19', 'CHICO_19', 'KOZLAR_19', 'JOINID', 'NH_ASIAN', '150K_200K', 'LHGTFT_19', 'BVAP', 'H_AMIN', 'H_2MORE19', 'NH_NHPI', 'Unnamed: 0.1', '60K_75K', 'normalized_first_round_eman', 'tot_occ_h_units_acs']
projection_variables_to_keep = [ 'RO_TP_19_pct', 'TP_19_pct', 'RO_GARCIA_G15_pct', 'GARCIA_G15_pct']
temporary_list_of_columns_to_delete = [ 'Ward_Prec','AMINVAP_pct', 'ASIANVAP_pct', 'NHPIVAP_pct', 'OTHERVAP_pct', '2MOREVAP_pct', 'LESS_10K_pct', '10K_15K_pct', '15K_20K_pct', '20K_25K_pct', '25K_30K_pct', '30K_35K_pct', '35K_40K_pct', '40K_45K_pct', '45K_50K_pct', '50K_60K_pct', '60K_75K_pct', '75K_100K_pct', '100K_125K_pct', '125K_150K_pct', '150K_200K_pct', '200K_MORE_pct', 'married_pct', 'divorced_pct', 'lt_highschool_pct', 'highschool_pct', 'some_college_pct', 'associates_pct', 'bachelors_pct', 'grad_and_professional_pct', 'drives_alone_work_pct', 'public_transit_work_pct', 'walk_to_work_pct', 'bike_to_work_pct', 'lt_10_min_pct', '10_to_30_min_pct', '30_to_60_min_pct', 'gt_60_min_pct', 'receiving_public_assistance_pct', 'eng_only_pct', 'esp_lim_pct', 'esp_not_lim_pct', 'other_lang_lim_pct', 'other_lang_not_lim_pct', 'non_computer_pct', 'internet_pct', 'family_pct',  'mbsa_occupation_pct', 'service_occupation_pct', 'sales_and_office_occupation_pct', 'nrcm_occupation_pct', 'ptmm_occupation_pct', 'cop_pct', 'poverty_ratio_lt_p50_pct', 'poverty_ratio_p50_p99_pct', 'poverty_ratio_1p00_1p24_pct', 'poverty_ratio_1p25_1p49_pct', 'poverty_ratio_1p50_1p84_pct', 'poverty_ratio_1p85_1p99_pct', 'poverty_ratio_gt_2p00_pct', 'medicare_medicaid_pct', 'tricare_va_pct', 'occ_per_room_lt_p50_pct', 'occ_per_room_p51_1p00_pct', 'occ_per_room_1p01_1p50_pct', 'occ_per_room_1p51_2p00_pct', 'occ_per_room_gt_2p00_pct', 'built_after_2014_pct', 'built_2010_2013_pct', 'built_00s_pct', 'built_90s_pct', 'built_80s_pct', 'built_70s_pct', 'built_60s_pct', 'built_50s_pct', 'built_40s_pct', 'built_pre_40s_pct']


num_intervals = 40
perc_overlap = 0.6
initial_cover = Cover(num_intervals=num_intervals,percent_overlap=perc_overlap)
amount_initial_centers=2




################pipeline

def load_in(data_path, columns_to_delete, projection_variables_to_keep, temporary_list_of_columns_to_delete):
    """ 
    Loads and preprocess tabular data from a CSV file, dropping unnecessary columns,
    and creating a pandas DataFrame and two lists: projection_options and column_names.

    Args:
        data_path (str): File path to the tabular data in CSV format.

        columns_to_delete (list): List of column names to be deleted from the DataFrame.

        projection_variables_to_keep (list): List of column names that are projection variables.

    Returns:
            df (pd.DataFrame): Cleaned DataFrame with unnecessary columns removed.

            projection_options (list): List of available projection options in the DataFrame.
    """
    df = pd.read_csv(data_path) 
    df = df.drop(columns = columns_to_delete)
    df = df.drop(columns = temporary_list_of_columns_to_delete)
    column_names = list(df.columns)
    projection_options = [ column for column in column_names if column in projection_variables_to_keep]
    return df, projection_options

# Load data and projection options
df, projection_options = load_in(data_path=data_path, columns_to_delete=columns_to_delete, projection_variables_to_keep=projection_variables_to_keep, temporary_list_of_columns_to_delete=temporary_list_of_columns_to_delete)

def process_all_projections(df, projection_options):
    """
    Process each projection option and return a dictionary with projection as keys and cleaned DataFrames as values.

    Args:
        df (pd.DataFrame): Cleaned DataFrame that was returned by the load_in function.
        projection_options (list): List of available projection options in the DataFrame.

    Returns:
       processed_results (dict): dictionary with projection options as keys and cleaned DataFrames as values.
    """
    processed_results = {}

    for projection in projection_options:
        result_df = drop_otherprojections(df, projection, projection_options)
        processed_results[projection] = result_df

    return processed_results

# Process projection options
processed_results = process_all_projections(df, projection_options)


def mapper_pipeline(projection_options= projection_options, processed_results=processed_results, data_path=data_path, initial_cover=initial_cover, amount_initial_centers=amount_initial_centers):
    """
    Process the entire pipeline for a given projection key and its associated DataFrame 
    that is acquired from the process_results dictionary from the process_all_projections function.

    Args:
        projection_options (list): List of available projection options in the DataFrame.

        data_path (str): File path to the tabular data in CSV format.

        processed_results (dict): dictionary returned in processed_all_projections

        initial_cover (object): An instance of the Cover class defined in the module mapper_xmean_cover.cover

        amount_initial_centers (int): Number of initial centers to begin with.

    Returns:
        None
    """
    for projection, result_df in processed_results.items():
        input_df_dictionary, column_names = input_df(df_with_selected_projection=result_df, projection_options=projection_options)
        input_data_dictionary = input_data(input_df_dictionary= input_df_dictionary)
        projection_index_dictionary = projection_index(input_df_dictionary=input_df_dictionary, selected_projection=projection)
        input_data_proj = input_data_projection(input_data_dictionary= input_data_dictionary, input_df_dictionary=input_df_dictionary, projection_index_dictionary= projection_index_dictionary)
        mapper_graph(input_df_dictionary= input_df_dictionary, input_data_dictionary=input_data_dictionary, input_data_proj= input_data_proj, projection_index_dictionary=projection_index_dictionary, column_names=column_names, initial_cover=initial_cover, amount_initial_centers=amount_initial_centers, data_path=data_path)

    return 

# Run mapper pipeline
mapper_pipeline(projection_options, processed_results, data_path, initial_cover, amount_initial_centers)





##################HELPER FUNCTIONS
def drop_otherprojections(df, selected_projection, projection_options):
    """ Creates a new pandas DataFrame that is associated with a selected projected variable,
        getting rid of the columns associated with the other projection variables. 

    Args:
        df (pd.DataFrame): Cleaned DataFrame that was returned by the load_in function.
        selected_projection (str): The selected projection variable.
        projection_options (list): List of available projection options in the DataFrame that was returned by the load_in function.

    Returns:
        result_df (pd.DataFrame): Cleaned DataFrame that does not have the columns associated with the currently unused projection options.
    """
    dropped_columns = [col for col in projection_options if col != selected_projection]
    df_with_selected_projection = df.drop(columns=dropped_columns).copy()
    df_with_selected_projection.name = f"df_{selected_projection}"
    return df_with_selected_projection


def input_df(df_with_selected_projection, projection_options):
    """ 
    Creates truncated versions of the result_df dataframe, each with a single column removed,
    and store them in a dictionary with descriptive keys.
    The only columns that cannot be removed are 'precinct', 'ward', and the projection variable currently being used. 

    Args:
        df_with_selected_projection (pd.DataFrame): Cleaned DataFrame without unused projection options (from drop_otherprojections).

        projection_options (list): List of available projection options in the DataFrame (from load_in).

    Returns:
        input_df_dictionary (dict): A dictionary where keys are in the form: selected_projection_-column_i-_eliminated.
                                    For each key, the corresponding value is a truncated version of df_with_selected_projection
                                    with only column i removed.

        column_names (list): A list of all the column names in df_with_selected_projection.
    """
    selected_projection_dataframe_dict = {}
    selected_projection_name = df_with_selected_projection.name
    selected_projection_dataframe_dict[f"{selected_projection_name}_none_eliminated"] = df_with_selected_projection
    column_names_new = df_with_selected_projection.columns
    undeletables= ['precinct', 'ward']

    for column in column_names_new:
        if column in projection_options:
            continue  # Skip the column that matches the projection variable
        elif column in undeletables:
            continue
        else:
            df_with_selected_projection_eliminated_column = df_with_selected_projection.drop(columns=column).copy()
            df_name = f"{selected_projection_name}_{column}_eliminated"
            selected_projection_dataframe_dict[df_name] = df_with_selected_projection_eliminated_column

    return selected_projection_dataframe_dict, column_names_new


def input_data(input_df_dictionary):
    """ 
        Creates a dictionary with the same structure as input_df_dictionary, 
        but with numpy arrays as values instead of pandas dataframes.

        Args:
            input_df_dictionary (dict): The dictionary returned by input_df.

        Returns:
            input_data_dict (dict): A dictionary where keys are in the form: selected_projection_-column_i-_eliminated.
                                    For each key, the corresponding value is a numpy array of the truncated dataframe.
    """
    input_data_dict = {}
    for key, df_with_eliminated_columns in input_df_dictionary.items():
        input_data_dict[key] = np.array(df_with_eliminated_columns)
    return input_data_dict


def projection_index(input_df_dictionary, selected_projection):
    """ 
    Creates a projection index dictionary that specifies the index of the column associated with the projection variable
    for each dataframe in input_df_dictionary.

    Args:
        input_df_dictionary (dict): The dictionary returned by input_df.

        selected_projection (str): String that dictates which projection variable is being used (from drop_otherprojections).

    Returns:
        projection_index_dictionary (dict): A dictionary where the key is a string indicating the selected projection,
                                            and the value is the index of the column associated with the projection variable.
    """
    projection_idx = {}
    for selected_projection_df, selected_projection_df_eliminated in input_df_dictionary.items():
        projection_idx[selected_projection_df] = selected_projection_df_eliminated.columns.get_loc(selected_projection)
    return projection_idx


def input_data_projection(input_data_dictionary, input_df_dictionary, projection_index_dictionary):
    """
    Creates a dictionary where the keys represent combinations of selected projection variables and eliminated columns,
    while the values are instances of the .project method from KeplerMapper.
    
    This is achieved by checking for matching keys in the input_data_dictionary, input_df_dictionary, and projection_index_dictionary.

    Args:
        input_data_dictionary (dict): The dictionary returned by input_data.
        input_df_dictionary (dict): The dictionary returned by input_df.
        projection_index_dictionary (dict): The dictionary returned by projection_index.
    
    Returns:
        input_data_proj (dict): A dictionary with a structure similar to input_df_dictionary.
        Keys are in the form: selected_projection_-column_i-_eliminated.
        For each key, the corresponding value is an instance of the .project method in KeplerMapper 
        that generates the projection lens for the mapper graph.
    """
    

    input_data_proj = {}#the keys are the problem here ()
    for key in input_df_dictionary.keys():
        if key in input_data_dictionary and key in projection_index_dictionary:
            data = input_data_dictionary[key]
            projection_idx = projection_index_dictionary[key]
            input_data_proj[key] = mapper_instance.project(data, projection =[projection_idx] , scaler = None)
    return input_data_proj


def mapper_graph(input_df_dictionary, input_data_dictionary, input_data_proj, projection_index_dictionary, column_names, initial_cover, amount_initial_centers, data_path):
    """
        Creates a directory structure named after the selected projection variable in the user's output folder.
        Inside this directory, an ensemble of pairs of .html and .json files is generated, where each pair is associated/named after 
        the keys in the input_df dictionary (representing the selected projection variable and eliminated column).

        Args:
            input_df_dictionary (dict): The dictionary returned by input_df.
            input_data_dictionary (dict): The dictionary returned by input_data.
            input_data_proj (dict): The dictionary returned by input_data_projection.
            projection_index_dictionary (dict): The dictionary returned by projection_index.
            column_names (list): List of columns returned by input_df.
            initial_cover (object): An instance of the Cover class defined in the module mapper_xmean_cover.cover.
            amount_initial_centers (int): Number of initial centers to begin with.
            data_path (str): File path to the original dataset as used in load-in.

        Returns:
            None
    """
    base_directory = "/Users/emariedelanuez/summer2023/outputs/"
    output_directory = os.path.join(base_directory, f"one_elimination_{selected_projection}/")
    os.makedirs(output_directory, exist_ok=True)
    for key in input_df_dictionary.keys():
        if key in input_data_dictionary and key in projection_index_dictionary and key in input_data_proj:
            file_path_html = os.path.join(output_directory, f"{key}.html")
            file_path_json = os.path.join(output_directory, f"{key}.json")
            data = input_data_dictionary[key]
            subset_column_names = [col for col in column_names if col in input_df_dictionary[key].columns]
            
            input_proj = input_data_proj[key]

            initial_centers = kmeans_plusplus_initializer(data, amount_initial_centers).initialize()
            clusterer = PyclusteringXMeans(initial_centers=initial_centers)
            output_graph = mapper_instance.map(input_proj, data, clusterer=clusterer, cover=KMapperAdaptiveCover( data, input_proj, initial_cover=initial_cover,clusterer=clusterer))
        
            mapper_instance.visualize(output_graph, 
            path_html = file_path_html, ## change file path as needed
            title=f"{key}", 
            color_values = list(data),  #list(input_data), ##add variables to color by even if not included in inputs
            color_function_name = subset_column_names,  ## second part of adding variables to color by 
            custom_tooltips = np.array([f"Ward {int(prct['ward'])} Precinct {int(prct['precinct'])}" for _, prct in input_df_dictionary[key].iterrows()]),
            X = data, 
            X_names = subset_column_names,
            include_searchbar=True)
            km.adapter.to_json(
            graph= output_graph, 
            X_projected=input_proj,
            X_data=data,
            X_names=list(subset_column_names),
            data_path=data_path,
            json_file=file_path_json)
    
    return 







