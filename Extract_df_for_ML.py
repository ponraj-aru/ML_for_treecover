# Author : Ponraj Arumugam
# Wageningen Environmental Research
import glob
import os
import geopandas as gpd
import rasterio
import pandas as pd
from rasterio.features import geometry_mask
from rasterio.mask import mask


main_path = "C:\\Users\\arumu002\\OneDrive - Wageningen University & Research\\Project_CC_ML\\2023\\Data_to_process\\raster\\"
lai_path = main_path+"Lai_5km\\"
fapar_path = main_path+"Fapar_5km\\"
yield_path = main_path+"yield_data\\"
gdf = gpd.read_file("C:\\Users\\arumu002\\OneDrive - Wageningen University & Research\\Project_CC_ML\\2023\\Data\\shap\\fishnet_5km_ETH_point_admin2_crop.shp")

for i in range(2006,2017):
    combined_df = pd.DataFrame()
    files_to_stack_lai = glob.glob(lai_path + "\\" + "*_"+str(i)+"_*.tif")
    # Create an empty DataFrame to store the extracted values
    # Loop through each raster file
    for raster_file in files_to_stack_lai:
        basenames_lai = os.path.basename(raster_file)
        basenames1_lai = basenames_lai.replace(".tif", "")
        parts = basenames1_lai.split('_')
        names_lai = "LAI_"+parts[3]
        print("Extracting data for: "+names_lai + " :for the year " + str(i))
        with rasterio.open(raster_file) as src:
            values_lai = []
            for index, row in gdf.iterrows():
                lon, lat = row.geometry.x, row.geometry.y
                # Sample the raster value at the point
                for val_lai in src.sample([(lon, lat)]):
                    values_lai.append(val_lai[0])  # Assuming a single-band raster
            combined_df[names_lai] = values_lai

    files_to_stack_fapar = glob.glob(fapar_path + "\\" + "*_" + str(i) + "_*.tif")
    for raster_file in files_to_stack_fapar:
        basenames_fapar = os.path.basename(raster_file)
        basenames1_fapar = basenames_fapar.replace(".tif", "")
        parts = basenames1_fapar.split('_')
        names_fapar = "FAPAR_"+parts[3]
        print("Extracting data for: " + names_fapar+ " :for the year " + str(i))
        with rasterio.open(raster_file) as src:
            values_fapar = []
            for index, row in gdf.iterrows():
                lon, lat = row.geometry.x, row.geometry.y
                # Sample the raster value at the point
                for val_fapar in src.sample([(lon, lat)]):
                    values_fapar.append(val_fapar[0])  # Assuming a single-band raster
            combined_df[names_fapar] = values_fapar

    files_to_stack_yield = glob.glob(yield_path + "\\" + str(i) + ".tif")
    for raster_file in files_to_stack_yield:
        print("Extracting data for: " + raster_file+ " :for the year " + str(i))
        with rasterio.open(raster_file) as src:
            values = []
            for index, row in gdf.iterrows():
                lon, lat = row.geometry.x, row.geometry.y
                # Sample the raster value at the point
                for val in src.sample([(lon, lat)]):
                    values.append(val[0])  # Assuming a single-band raster
            combined_df["YIELD"] = values
    combined_df["Year"] = i
    combined_df.to_csv(main_path+"df_all_"+str(i)+".csv")

