# Author : Ponraj Arumugam
# Potsdam Institute for Climate Impact Research

# load required libraries here
import os
from osgeo import gdal, ogr, osr, gdalconst, gdal_array
import rasterio
import subprocess, glob
import time
from osgeo.gdalnumeric import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.metrics import  precision_score,recall_score,average_precision_score,roc_auc_score
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import ensemble
import gc
from joblib import Parallel, delayed
import csv
#from gdalconst import GA_ReadOnly
from osgeo import gdalconst
import fiona
import rasterio.mask
import geopandas as gpd
import rasterio
from rasterio.plot import show
from rasterio.plot import show_hist
from rasterio.mask import mask
from shapely.geometry import box
from fiona.crs import from_epsg
import pycrs

os.environ = r"C:\OSGeo4W64\apps\Python37\lib\site-packages\osgeo\data\proj"
# Mention the path where MAP_work is located
# Here I am taking the working directory from os.getcwd()
path = os.getcwd()
path1 = "F:\\"
# Creating directory to keep processed files
dirName = path1+"DATA_for_processing\\"
try:
    # Create target Directory
    os.mkdir(dirName)
    print("Step 1: The folder for Data processing is Created ")
except FileExistsError:
    print("Step 1: The folder for Data processing is already exists")

# Mention the path where landsat tree cover data located
# mention the path where MODIS EVI located
# Mention the path where the data will be generated (mosaic data, re-sampled data)
# Mention the paths of gdal_merge.py and gdal_calc.py which are used for mosaicking and masking respectively
# gdal_merge.py and gdal_calc.py will generate when you are installing gdal in your python
yield_path = path+"\\sim_yield\\"
modis_path = path+"\\Lai"
data_path = path+"\\DATA_for_processing\\"
data_path1 = path1+"\\DATA_for_processing\\"
gdal_merge = "C:\\gdal_merge.py"
gdal_calc = "C:\\gdal_calc.py"
gdal_polygonize = "C:\\gdal_polygonize.py"
yield_data = yield_path + "2002.tif" #  path for crop model simulated data
rice_mask = data_path+"Reclass_RICE_mask_IRRI_resample.tif" # IRRI crop mask
rice_mask_shp = data_path+"Reclass_RICE_mask_IRRI_resample_shp_IND.shp" # IRRI crop mask shp
points_for_extraction = data_path+"MODIS_5km_points_IND.shp" # Points for data extraction


# Reprojecting stacked MODIS data
# Getting Landsat tree cover projection to convert MODIS projection
# Using gdal_wrap to reproj MODIS raster to LANDSAT's SRS
for i in range(2003,2019):
    files_to_stack = glob.glob(modis_path + "\\" + "*_"+str(i)+"_*.tif")  # list all the tif files
    stacked_modis = data_path1 +str(i)+ "_stacked.tif"  # path for stacked
    stacked_modis_reproj = data_path1 +str(i)+"_stacked_reproj.tif"  # path for reprojected stacked data
    if os.path.exists(stacked_modis):
        print("Step 2: Stacked process is done for "+str(i)+". Proceeding to next steps.")
    else:
        print("Step 2: Stacking MODIS tile "+str(i)+" Please Wait...")
        # Read metadata of first file
        with rasterio.open(files_to_stack[0]) as src0:
            meta = src0.meta
            # Update meta to reflect the number of layers
        meta.update(count=len(files_to_stack))
        # Read each layer and write it to stack
        with rasterio.open(stacked_modis, 'w', **meta) as dst:
            for id, layer in enumerate(files_to_stack, start=1):
                with rasterio.open(layer) as src1:
                    dst.write_band(id, src1.read(1))
            print("Step 2: stacked process is completed for "+ str(i))
    # reprojection of stacked file
    ds = gdal.Open(yield_data)
    prj = ds.GetProjection()
    srs = osr.SpatialReference(wkt=prj)

    input_raster = gdal.Open(stacked_modis)
    output_raster = stacked_modis_reproj

    if os.path.exists(stacked_modis_reproj):
        print("Step 3: Reprojecting the MODIS stacked "+str(i)+"data has been done already. Proceeding to to Next step.")
    else:
        print("Step 3: Reprojecting and writing MODIS stacked data "+str(i)+" to YIELD DATA srs system. Please Wait ...")
        gdal.Warp(output_raster, input_raster, dstSRS=srs.ExportToWkt())
        print("Step 3: Re-projection process is completed for "+str(i))


# masking crop area from IRRI crop mask to MODIS reprojected stacked
# resampling MDOIS reprojected stacked data to yield data
    referencefile = yield_data # 5km resolution yield data
    inputfile = stacked_modis_reproj  # stacked MODIS
    outputfile = data_path1+"MODIS_stack_reproj_masked_"+ str(i)+".tif" # masked stacked reprojected MODIS
    outputfile_sf = data_path1+"MODIS_stack_reproj_masked_sf_"+ str(i)+".tif" # masked stacked reprojected MODIS
    outputfile_resample = data_path1+"MODIS_stack_reproj_masked_resample_"+ str(i)+".tif" # masked stacked reprojected MODIS
    outputfile_resample_mask = data_path1+"MODIS_stack_reproj_masked_resample_mask_"+ str(i)+".tif" # masked stacked reprojected MODIS
    outputfile_resample_mask_1= data_path1+"MODIS_stack_reproj_masked_resample_mask_<7"+ str(i)+".tif" # masked stacked reprojected MODIS

# Converting raster to shapefile (polygon and points)
    if os.path.exists(rice_mask_shp):
        print("Step 4: Rice mask is converted already as a shapefile. Proceed to next")
    else:
        command = "python"+" "+gdal_polygonize+" "+rice_mask+" "+"-f"+" "+'"ESRI Shapefile"'+" "+rice_mask_shp
        print("Step 4: Converting rice mask as polygon is started. Please wait...")
        output = subprocess.check_output(command)
        print("Step 4: Converting rice mask as polygon is completed")

# masking process
    if os.path.exists(outputfile):
        print("Step 5: Masking stacked modis "+str(i)+"from rice mask shapefile has been done already. please proceed to next..")
    else:
        print("Step 5: Masking stacked modis "+str(i)+" from rice mask shapefile is started. Please wait")
        print("Step 5.1: Reading shapefile. Please wait")
        with fiona.open(rice_mask_shp, "r") as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]
        print("Step 5.2: Reading stacked raster. Please wait")
        with rasterio.open(inputfile) as src:
            out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
            out_meta = src.meta
        out_meta.update({"driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform})
        print("Step 5.3: Masking process. Please wait")
        with rasterio.open(outputfile, "w", **out_meta) as dest:
            dest.write(out_image)
        print("Step 5: Masking stacked modis "+str(i)+" from rice mask shapefile is completed.")
        del inputfile


# Multiplying with scale factor
    if os.path.exists(outputfile_sf):
        print("Step 6: Multiply masked stacked modis "+str(i)+" with scale factor has been done already. please proceed to next..")
    else:
        print("Step 6: Multiply masked stacked modis "+str(i)+" with scale factor is started. Please wait...")
        command = "python" + " " + gdal_calc + " " + "-A" + " " + outputfile + " " + " " + "--outfile=" + outputfile_sf + " " +"--allBands=A --NoDataValue=0 --type=Float32 --calc=" + '"A*0.1"'
        output = subprocess.check_output(command)
        print("Step 6: Multiply masked stacked modis "+str(i)+" with scale factor is completed.")
    #command = "python"+" "+gdal_calc+" "+"-A"+" "+rice_mask+" "+"-B"+" "+inputfile+" "+"--outfile="+outputfile+" "+"--NoDataValue=0 --calc="+'"B*(A>0)"'

    if os.path.exists(outputfile_resample):
        print("Step 7: Resample of stacked raster "+str(i)+" already completed . Proceed to next")
    else:
        print("Step 7: Resample of stacked raster "+str(i)+" is started. Please wait..")
        inputfile = outputfile_sf
        input = gdal.Open(inputfile)
        inputProj = input.GetProjection()
        inputTrans = input.GetGeoTransform()
        inputbands = input.RasterCount
        referencefile = referencefile
        reference = gdal.Open(referencefile, gdalconst.GA_ReadOnly)
        referenceProj = reference.GetProjection()
        referenceTrans = reference.GetGeoTransform()
        bandreference = reference.GetRasterBand(1)
        x = reference.RasterXSize
        y = reference.RasterYSize
        outputfile = outputfile_resample
        driver= gdal.GetDriverByName('GTiff')
        output = driver.Create(outputfile,x,y,inputbands,bandreference.DataType)
        output.SetGeoTransform(referenceTrans)
        output.SetProjection(referenceProj)
        gdal.ReprojectImage(input,output,inputProj,referenceProj,gdalconst.GRA_Average)
        print("Step 7: Resample of stacked raster "+str(i)+" is completed")
        del output

    if os.path.exists(outputfile_resample_mask):
        print("Step 8: Masking after resample for "+str(i)+" has been done already. please proceed to next..")
    else:
        print("Step 8: Masking after resample for "+str(i)+"is started. Please wait...")
        command = "python" + " " + gdal_calc + " " + "-A" + " " + outputfile_resample + " " + " " + "--outfile=" + outputfile_resample_mask + " " +"--allBands=A --NoDataValue=0 --type=Float32 --calc=" + '"A*(A>0)"'
        output = subprocess.check_output(command)
        print("Step 8: Masking after resample for " + str(i) + " is completed.")
    #command = "python"+" "+gdal_calc+" "+"-A"+" "+rice_mask+" "+"-B"+" "+inputfile+" "+"--outfile="+outputfile+" "+"--NoDataValue=0 --calc="+'"B*(A>0)"'
    os.unlink(data_path1+"MODIS_stack_reproj_masked_"+ str(i)+".tif")
    os.unlink(data_path1+"MODIS_stack_reproj_masked_resample_"+ str(i)+".tif")
    os.unlink(data_path1 + str(i) + "_stacked_reproj.tif")
    # os.unlink(data_path1 + str(i) + "_stacked.tif")

# extract values from tree cover data (reference data)
src_filename = yield_data
shp_filename = points_for_extraction

src_ds=gdal.Open(src_filename)
gt=src_ds.GetGeoTransform()
rb=src_ds.GetRasterBand(1)

ds=ogr.Open(shp_filename)
lyr=ds.GetLayer()
ras_values = []
lat=[]
lon=[]

for feat in lyr:
    geom = feat.GetGeometryRef()
    mx, my = geom.GetX(), geom.GetY()  # coord in map units
    # Convert from map to pixel coordinates.
    # Only works for geotransforms with no rotation.
    px = int((mx - gt[0]) / gt[1])  # x pixel
    py = int((my - gt[3]) / gt[5])  # y pixel
    intval = rb.ReadAsArray(px, py, 1, 1)
    ras_values.append(intval[0])
    lon.append(mx)
    lat.append(my)

tree_cover_values = pd.DataFrame(columns=['Lon', 'Lat', 'tree_cover'])
tree_cover_values['Lon']=lon
tree_cover_values['Lat']=lat
tree_cover_values['tree_cover']=ras_values
tree_cover_values['tree_cover'] = pd.DataFrame([str(line).strip('[').strip(']') for line in tree_cover_values['tree_cover']])
#tree_cover_values['tree_cover'] = tree_cover_values['tree_cover'].astype(int32)
tree_cover_values['lat_lon'] = tree_cover_values['Lat'].map(str)+"_"+tree_cover_values['Lon'].map(str)

print("Step 10: Yield values are extracted for random pixels")
# extract values from MODIS EVI time series data (input data)
src_filename = outputfile_resample_mask
shp_filename = points_for_extraction

src_ds=gdal.Open(src_filename)
bands_count = src_ds.RasterCount
gt=src_ds.GetGeoTransform()
rb=src_ds.GetRasterBand(1)

ds=ogr.Open(shp_filename)
lyr=ds.GetLayer()
ras_values = []
lat=[]
lon=[]

for feat in lyr:
    geom = feat.GetGeometryRef()
    mx, my = geom.GetX(), geom.GetY()  # coord in map units
    # Convert from map to pixel coordinates.
    # Only works for geotransforms with no rotation.
    for band in range(1,bands_count+1):
        rb = src_ds.GetRasterBand(band)
        px = int((mx - gt[0]) / gt[1])  # x pixel
        py = int((my - gt[3]) / gt[5])  # y pixel
        intval = rb.ReadAsArray(px, py, 1, 1)
        # print(intval[0])
        ras_values.append(intval[0])
        lon.append(mx)
        lat.append(my)

df = pd.DataFrame(columns=['Lon', 'Lat', 'tree_cover'])
df['Lon']=lon
df['Lat']=lat
df['tree_cover']=ras_values
df['tree_cover'] = pd.DataFrame([str(line).strip('[').strip(']') for line in df['tree_cover']])
#df['tree_cover'] = df['tree_cover'].astype(int32)

df['lat_lon'] = df['Lat'].map(str)+"_"+df['Lon'].map(str)
df['idx'] = df.groupby('lat_lon').cumcount()+1
df['tree_cover_idx'] = 'evi_' + df.idx.astype(str)
tree_cover = df.pivot(index='lat_lon',columns='tree_cover_idx',values='tree_cover')
reshape = pd.concat([tree_cover],axis=1)
reshape['lat_lon'] = reshape.index
modis_evi = reshape.reset_index(drop=True)
print("Step 11: The MODIS EVI time series values are extracted for random pixels")

df1= pd.merge(tree_cover_values, modis_evi, on='lat_lon')
df2 = df1.drop(columns=['lat_lon'])

evi_names =["Lon","Lat","tree_cover"]
for i in range(1,bands_count+1):
    names = "evi_"+str(i)
    evi_names.append(names)
column_names = evi_names

df_all = df2.reindex(columns=column_names)
df_all_1 = df_all.replace(0,NaN)
df_all_2 = df_all_1.apply(pd.to_numeric)
df_all_2 = df_all_2[df_all_2['tree_cover'] > 0]

print("Step 12: EVI time series and LANDSAT tree cover data are merged as data frame")
print("")
print("###################### The process of Gradient Boosting Regression Starts Here.. ############################")
print("")
df_all_2['lat_lon'] = df_all_2['Lat'].map(str)+"_"+df_all_2['Lon'].map(str)
df_all_order =["Lon","Lat","lat_lon","tree_cover"]
for i in range(1,bands_count+1):
    names = "evi_"+str(i)
    df_all_order.append(names)
column_names = df_all_order
df_all_3 = df_all_2.reindex(columns=column_names)

# The process of filling values starts here.
print("Step 13: The process of filling missing values is started using time series linear time method")
print("")
df_all_T = df_all_3.T # transpose the data for filling missing values using linear method
df_all_T1 = df_all_T.drop(df_all_T.index[:4])
date = pd.date_range('2002-01-01', '2002-12-31', freq='16D')
df_all_T1.columns = [df_all_3['lat_lon']]
df_all_T2 = df_all_T1.set_index(date)
df_all_float = df_all_T2.apply(pd.to_numeric, errors='coerce')
df_fill_miss = df_all_float.interpolate(method='linear')
df_fill_miss_T = df_fill_miss.T

df_all_order =[]
for i in range(1,bands_count+1):
    names = "evi_"+str(i)
    df_all_order.append(names)
column_names = df_all_order
df_fill_miss_T.columns = column_names

df_fill_miss_T.reset_index(level=0, inplace=True)
new = df_fill_miss_T["lat_lon"].str.split("_", n = 1, expand = True)
df_fill_miss_T['Lat'] = new[0]
df_fill_miss_T['Lon'] = new[1]
df_fill_miss_T = df_fill_miss_T.merge(df_all_3[['lat_lon','tree_cover']],on=['lat_lon'])

evi_names =["Lon","Lat","lat_lon","tree_cover"]
for i in range(1,bands_count+1):
    names = "evi_"+str(i)
    evi_names.append(names)
column_names = evi_names
df_final = df_fill_miss_T.reindex(columns=column_names)

print("Step 14: The process of gradient booster regression starts from here")
train_df = df_final


data_X = train_df.drop(["Lat","Lon","tree_cover","lat_lon"],axis=1)
data_y = train_df['tree_cover']

GBmodel = GradientBoostingRegressor(random_state=0)
param_dist = {'learning_rate': np.linspace(0.05,0.1),'max_depth': range(1, 5),
              'max_features': [20,15],'subsample': [1]}
model = GridSearchCV(estimator=GBmodel, param_grid=param_dist, n_jobs=1, cv=2, scoring='neg_mean_squared_error')
print('Gradient boosted tree regression...')
model.fit(data_X, data_y)
print('Best Params:')
print(model.best_params_)
print('Best CV Score:')
print(model.best_score_)


df_all_order =[]
src_ds=gdal.Open(outputfile_sf)
bands_count = src_ds.RasterCount
for i in range(1, bands_count + 1):
    names = "evi_" + str(i)
    df_all_order.append(names)
column_names = df_all_order

def tree_cover_estimation(multiband_raster):
    gc.collect()
    src_ds=gdal.Open(multiband_raster)
    #out_file = multiband_raster[-19:].replace('.tif', '') # output file name
    out_final_f = data_path+"2002_GBR_yield.tif" # output file name with path
    nda = src_ds.ReadAsArray()
    bands_count = src_ds.RasterCount
    nda_T = nda.T # Transposing array
    nda_2d = nda_T.reshape(int(nda_T.shape[0]*nda_T.shape[1]),int(nda_T.shape[2])) # converting 3d array to 2d array
    df = pd.DataFrame(nda_2d, columns=column_names) # converting 2d array as datafarme
    # Looping into each and every bands to replace -9999 to NaN
    for j in range(1, bands_count + 1):
        print("Replacing 0 to NaN of "+multiband_raster+" of "+str(j)+" th column")
        df['evi_'+str(j)] = df['evi_'+str(j)].replace(0, np.nan)
        rb = src_ds.GetRasterBand(j)
        rb_arr = rb.ReadAsArray()
    # for filling missing values data should be time series
    # So the column is created for a year with 8 days interval
    date = pd.date_range('2002-01-01', '2002-12-31', freq='16D')
    df.columns = date
    # each tile has almost million rows. Converting million rows as column (transpose) lead to memory error.
    # Therefore splitting dataframe into chunks, process then append
    size = 100000
    list_of_dfs = [df.loc[i:i + size - 1, :] for i in range(0, len(df), size)]
    df_all = pd.DataFrame([])
    chunk_nr=1
    for chunks in list_of_dfs:
        print("chunk number:"+str(chunk_nr)+" <- "+str(len(chunks)) + ' rows from '+multiband_raster+' .tif is processed')
        gc.collect()
        chunks_T = chunks.T
        df_test_fill = chunks_T.interpolate(method='linear')
        df_test_fill_T = df_test_fill.T
        df_fill = df_test_fill_T.dropna()
        df_fill.columns = column_names
        df_all = df_all.append(df_fill)
        gc.collect()
        chunk_nr=chunk_nr+1
    # Predicting tree cover with model
    result = model.predict(df_all)
    # writing results as data frame. After the removal on NA values some index are removed.
    # To genrate whole array we require to match this dataframe with main dataframe
    # The final dataframe will be converted as array, and array will be written to raster.
    df_all['tree_cover'] = result
    df_join = df_all.join(df, how='outer')
    tree_cover_predict = df_join['tree_cover']
    tree_cover_array = np.array(tree_cover_predict)
    # transposing to source file shape
    tree_cover_array_2d = np.reshape(tree_cover_array, (rb.XSize,rb.YSize)).transpose()
    # Converting array to raster
    geotransform = src_ds.GetGeoTransform()
    wkt = src_ds.GetProjection()
    driver = gdal.GetDriverByName("GTiff")
    output_file = out_final_f
    print("Writing "+output_file+" is completed")
    dst_ds = driver.Create(output_file, rb.XSize, rb.YSize, 1, gdal.GDT_Int16)
    # writing output raster
    dst_ds.GetRasterBand(1).WriteArray(tree_cover_array_2d)
    # setting nodata value
    dst_ds.GetRasterBand(1).SetNoDataValue(-9999)
    # setting extension of output raster
    # top left x, w-e pixel resolution, rotation, top left y, rotation, n-s pixel resolution
    dst_ds.SetGeoTransform(geotransform)
    # setting spatial reference of output raster
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    dst_ds.SetProjection(srs.ExportToWkt())
    # Close output raster dataset
    ds = None
    dst_ds = None

tree_cover_estimation(outputfile_sf)


# Extracting LANDSAT estimated tree cover and actual tree cover for randomly generated shapefile
# The scatter plot is created between them
outputfile1 = "C:\\Users\\ponraj\\Desktop\\Machine_Leraning\\ML\\DATA_for_processing\\2002_GBR_yield_resample.tif" # Output of 250m spatial resolution Landsat tree cover map (masked -
# -with MODIS extent)
act_treecover = gdal.Open(outputfile1)
gt = act_treecover.GetGeoTransform()
rb = act_treecover.GetRasterBand(1)
ds=ogr.Open(points_for_extraction)
lyr=ds.GetLayer()
ras_values_act = []
for feat in lyr:
    geom = feat.GetGeometryRef()
    mx, my = geom.GetX(), geom.GetY()  # coord in map units
    # Convert from map to pixel coordinates.
    # Only works for geotransforms with no rotation.
    px = int((mx - gt[0]) / gt[1])  # x pixel
    py = int((my - gt[3]) / gt[5])  # y pixel
    intval = rb.ReadAsArray(px, py, 1, 1)
    ras_values_act.append(intval[0])

est_treecover = gdal.Open("C:\\Users\\ponraj\\Desktop\\Machine_Leraning\\ML\\DATA_for_processing\\2002_GBR_yield.tif")
gt=est_treecover.GetGeoTransform()
rb=est_treecover.GetRasterBand(1)
ds=ogr.Open(points_for_extraction)
lyr=ds.GetLayer()
ras_values_est = []
for feat in lyr:
    geom = feat.GetGeometryRef()
    mx, my = geom.GetX(), geom.GetY()  # coord in map units
    # Convert from map to pixel coordinates.
    # Only works for geotransforms with no rotation.
    px = int((mx - gt[0]) / gt[1])  # x pixel
    py = int((my - gt[3]) / gt[5])  # y pixel
    intval = rb.ReadAsArray(px, py, 1, 1)
    ras_values_est.append(intval[0])
# Scatter plot between observed and estimated tree cover
# For quick plot I have chosen 10000 values
plt.scatter(ras_values_act, ras_values_est)
plt.title('Actual Tree Cover Vs Estimated Tree Cover (Random points)')









sns.distplot(train_df.tree_cover.values, kde=None)
plt.xlabel('tree_cover')
sns.distplot(model.predict(data_X),kde=None)