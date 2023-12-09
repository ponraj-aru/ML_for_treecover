import os
import rioxarray
import pandas as pd
import numpy as np
import gc
from joblib import dump, load
import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.enums import Resampling
from geocube.api.core import make_geocube
import geopandas as gpd

model_path = 'C:\\Users\\arumu002\\OneDrive - Wageningen University & Research\\Project_CC_ML\\2023\\models\\'
model = model_path + "GBR.joblib"

for i in range(2006,2023):
    print(i)
    tif_directory = 'C:\\Users\\arumu002\\OneDrive - Wageningen University & Research\\Project_CC_ML\\2023\\Data_to_process\\raster\\Lai_crop\\'+str(i)
    # Create a empty DataFrame to store the data
    df = pd.DataFrame()
    # Iterate over TIF files in the directory
    for file_name in os.listdir(tif_directory):
        for name in range(121, 309, 4):
            if file_name.endswith(str(name) + ".tif"):
                file_path = os.path.join(tif_directory, file_name)
                print(file_name)
                raster = rioxarray.open_rasterio(file_path)
                # Extract latitude and longitude coordinates
                # Create meshgrid for latitude and longitude
                lon, lat = np.meshgrid(raster.x.values, raster.y.values)
                lat = lat.flatten()
                lon = lon.flatten()
                values = raster.values.flatten()
                names = "LAI_" + file_name[18:21]
                file_df = pd.DataFrame({"Latitude": lat, "Longitude": lon, names: values})
                df = pd.concat([df, file_df], axis=1)
                raster.close()
                gc.collect()
    unique_columns = df.loc[:, ~df.columns.duplicated()]
    # Get columns excluding Latitude and Longitude
    data_columns = unique_columns.columns.difference(["Latitude", "Longitude"])
    # Remove rows where more than 50% of columns (excluding Latitude and Longitude) have values less than 0
    threshold_percentage = 40
    threshold = len(data_columns) * threshold_percentage / 100
    cleaned_df = unique_columns.loc[(unique_columns[data_columns] >= 0).sum(axis=1) >= threshold]
    # Columns to exclude from replacement
    exclude_columns = ['Latitude', 'Longitude']
    # Replace all values lesser than 0 with 'na', except for specified columns
    cleaned_df.loc[:, ~cleaned_df.columns.isin(exclude_columns)] = cleaned_df.loc[:,
                                                                   ~cleaned_df.columns.isin(exclude_columns)].applymap(
        lambda x: 'NaN' if x < 0 else x)
    # Columns to leave in the DataFrame
    columns_to_leave = ['Latitude', 'Longitude']
    # Extract the columns to leave
    left_columns = cleaned_df[columns_to_leave]
    # Transpose the remaining data
    transposed_data = cleaned_df.drop(columns=columns_to_leave).transpose()
    transposed_data1 = transposed_data.apply(pd.to_numeric, errors='coerce')
    cleaned_df_fill = transposed_data1.interpolate(method='linear', axis=1)
    cleaned_df_fill1 = cleaned_df_fill.transpose()
    final_df_lai = pd.concat([left_columns, cleaned_df_fill1], axis=1)
    #print(final_df_lai)
    # Print the resulting DataFrame
    # final_df_lai.to_csv("C:\\Users\\arumu002\\OneDrive - Wageningen University & Research\\Project_CC_ML\\2023\\Data_to_process\\raster\\Lai_crop\\2006\\2006_2.csv", index=False)

    tif_directory = 'C:\\Users\\arumu002\\OneDrive - Wageningen University & Research\\Project_CC_ML\\2023\\Data_to_process\\raster\\Fapar_crop\\'+str(i)
    # Create a empty DataFrame to store the data
    df = pd.DataFrame()
    # Iterate over TIF files in the directory
    for file_name in os.listdir(tif_directory):
        for name in range(121, 309, 4):
            if file_name.endswith(str(name) + ".tif"):
                file_path = os.path.join(tif_directory, file_name)
                print(file_name)
                raster = rioxarray.open_rasterio(file_path)
                # Extract latitude and longitude coordinates
                # Create meshgrid for latitude and longitude
                lon, lat = np.meshgrid(raster.x.values, raster.y.values)
                lat = lat.flatten()
                lon = lon.flatten()
                values = raster.values.flatten()
                names = "FAPAR_" + file_name[19:22]
                file_df = pd.DataFrame({"Latitude": lat, "Longitude": lon, names: values})
                df = pd.concat([df, file_df], axis=1)
                raster.close()
                gc.collect()
    unique_columns = df.loc[:, ~df.columns.duplicated()]
    # Get columns excluding Latitude and Longitude
    data_columns = unique_columns.columns.difference(["Latitude", "Longitude"])
    # Remove rows where more than 50% of columns (excluding Latitude and Longitude) have values less than 0
    threshold_percentage = 20
    threshold = len(data_columns) * threshold_percentage / 100
    cleaned_df = unique_columns.loc[(unique_columns[data_columns] >= 0).sum(axis=1) >= threshold]
    # Columns to exclude from replacement
    exclude_columns = ['Latitude', 'Longitude']
    # Replace all values lesser than 0 with 'na', except for specified columns
    cleaned_df.loc[:, ~cleaned_df.columns.isin(exclude_columns)] = cleaned_df.loc[:,
                                                                   ~cleaned_df.columns.isin(exclude_columns)].applymap(
        lambda x: 'NaN' if x < 0 else x)
    # Columns to leave in the DataFrame
    columns_to_leave = ['Latitude', 'Longitude']
    # Extract the columns to leave
    left_columns = cleaned_df[columns_to_leave]
    # Transpose the remaining data
    transposed_data = cleaned_df.drop(columns=columns_to_leave).transpose()
    transposed_data1 = transposed_data.apply(pd.to_numeric, errors='coerce')
    cleaned_df_fill = transposed_data1.interpolate(method='linear', axis=1)
    cleaned_df_fill1 = cleaned_df_fill.transpose()
    final_df_fapar = pd.concat([left_columns, cleaned_df_fill1], axis=1)
    print(final_df_lai)
    # Columns to merge on
    merge_columns = ['Latitude', 'Longitude']
    # Perform the merge based on the specified columns
    result_df = pd.merge(final_df_lai, final_df_fapar, on=merge_columns)
    result_df['Year'] = i
    # Extract the columns to leave
    left_columns = result_df[merge_columns]
    # Columns to apply the model
    columns_for_model = result_df.drop(columns=columns_to_leave)
    predictor = load(model)
    predictions = predictor.predict(columns_for_model)
    predictions_df = pd.DataFrame({'Yield': predictions})
    df = pd.concat([left_columns, predictions_df], axis=1)

    #df.to_csv('C:\\Users\\arumu002\\OneDrive - Wageningen University & Research\\Project_CC_ML\\2023\\models\\'+str(i)+'.csv')
    A = np.array(df)
    xmin = np.min(A[:, 2])
    xmax = np.max(A[:, 2])
    ymin = np.min(A[:, 1])
    ymax = np.max(A[:, 1])
    deltax = (xmax - xmin) / 10
    deltay = (ymax - ymin) / 10
    res = min([deltay, deltay])
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))
    gdf.plot()  # first image hereunder
    geotif_file = 'C:\\Users\\arumu002\\OneDrive - Wageningen University & Research\\Project_CC_ML\\2023\\sim_yields\\' +str(i)+ '.tif'
    out_grd = make_geocube(
        vector_data=gdf,
        measurements=["Yield"],
        resolution=(-0.00420530653620973, 0.00420530653620973)
    )
    out_grd["Yield"].rio.to_raster(geotif_file)

#output_path = 'C:\\Users\\arumu002\\OneDrive - Wageningen University & Research\\Project_CC_ML\\2023\\sim_yields\\'+'2006.tif'
