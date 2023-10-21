from gerrychain import Graph
import geopandas as gpd

gdf = gpd.read_file("ga_vtd.shp")
graph = Graph.from_geodataframe(gdf)
graph.to_json("ga_vtd_dual.json")
