import openeo
c=openeo.connect("openeo.cloud").authenticate_oidc()

[col['id'] for col in c.list_collections() if "WORLDCEREAL" in col['id']]

c.describe_collection("ESA_WORLDCEREAL_MAIZE")

extent = {'west': 32.5, 'south': 3.2, 'east': 44.0, 'north': 15.0, 'crs': 'EPSG:4326'}
temporal = ('2020-09-12T00:00:00Z', '2021-12-20T00:00:00Z')
maize = c.load_collection("ESA_WORLDCEREAL_MAIZE",
                         temporal_extent= temporal,
                         spatial_extent=extent,
                         bands=["classification"])
#filtered_maize1 = maize.mask(mask=maize > 100)
#filtered_maize2 = filtered_maize1.mask(mask=maize < 100)
#summed_maize = filtered_maize2.reduce_dimension(dimension="t", reducer="sum")
#summed_maize = maize.reduce_dimension(dimension="t", reducer="mean")
job = maize.execute_batch("maize_ethiopia.tif")
job.get_results()

