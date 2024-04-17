import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from shapely.geometry import Point, Polygon
from sklearn.neighbors import BallTree


def loader(path_to_data):
    """Load the data, get geometry and return a GeoDataFrame."""

    data = xr.open_dataset(path_to_data)
    df = data.to_dataframe()
    df.reset_index(inplace=True)

    # drop missing
    df.replace(1e20, np.nan, inplace=True)
    df.dropna(inplace=True)

    print('Loaded data, converting to GeoDataFrame.')

    crs = {'init':'epsg:4326'}
    geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]

    geo_df = gpd.GeoDataFrame(df, 
                            crs=crs, 
                            geometry=geometry)
    
    print('GeoDataFrame loaded.')

    return geo_df


def filter_relevant(df):
    """Keep only data points that are in the right lon and lat region and have non-zero (>30) discharge."""
    
    dis_threshold = 30  # Define threshold for "close to zero" (using above plot)
    mask = (df['dis24'].abs() > dis_threshold) & (df['lon'] > 82) & (df['lon'] < 96) & (df['lat'] > 22) & (df['lat'] < 31)
    return df[mask]


def calculate_distance(point1, point2):
    """Calculate the distance between two points."""

    return point1.distance(point2)


def get_closest(df, river_points, nn=5):
    """Find the closest point in the df to each point in the river."""
    
    # Create an empty DataFrame to store the closest points
    closest_points = pd.DataFrame(columns=df.columns)

    # Iterate over each unique day in the 'time' column
    for day, one_day_data in df.groupby(df['time'].dt.to_period('D')):
        # Create a BallTree for the current day's data
        tree = BallTree(one_day_data['geometry'].apply(lambda x: [x.coords[0][0], x.coords[0][1]]).tolist())
        
        # Find the closest points in one_day_data to each point in river_points
        for true_coord in river_points:
            true_point = Point(true_coord)
            _, ind = tree.query([[true_point.x, true_point.y]], k=nn)  # Find 5 NNs --> 1 is too few

            if len(ind) > 0:
                # Choose closest point among 5 NNs
                min_distance = float('inf')
                closest_row = None

                for i in ind[0]:
                    distance = calculate_distance(one_day_data.iloc[i]['geometry'], true_point)
                    if distance < min_distance:
                        min_distance = distance
                        closest_row = one_day_data.iloc[i]
                
                closest_points.loc[len(closest_points.index)] = closest_row

    print('Finished finding closest points, converting to geo.')
    
    geo_points = gpd.GeoDataFrame(closest_points, 
                                  crs={'init':'epsg:4326'}, 
                                  geometry=closest_points['geometry'])
    
    geo_points['time'] = pd.to_datetime(geo_points['time'])
    geo_points['time'] = geo_points['time'].dt.date

    return geo_points


def remove_outliers(df):
    """Remove cluster not in correct river."""

    boundary_coords = [(82, 30.5), (83, 30), (84, 29), (86, 29), (86, 28), (82, 28)] 
    incorrect_polygon = Polygon(boundary_coords)

    # Filter out wrong points
    incorrect_region = gpd.GeoSeries([incorrect_polygon], crs={'init':'epsg:4326'})
    #incorrect_points = df[df.geometry.within(incorrect_region.unary_union)]
    df = df[~df.geometry.within(incorrect_region.unary_union)]

    return df