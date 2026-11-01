"""GeoJSON planning-area join (the headline data-engineering step).

The taxi API returns only raw GPS coordinates. To answer "how many taxis per
planning area" we spatially join each taxi point to the URA Master Plan 2019
subzone polygons (MasterPlan2019SubzoneBoundaryNoSeaGEOJSON.geojson) and count by
`PLN_AREA_N` using geopandas.sjoin (same logic as the Airflow join task).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GEOJSON_PATH = ROOT / "shared_assets" / "MasterPlan2019SubzoneBoundaryNoSeaGEOJSON.geojson"


@lru_cache(maxsize=1)
def load_subzones(path: str | None = None) -> gpd.GeoDataFrame:
    """Load subzone polygons, normalised to WGS84 with an upper-case PLN_AREA_N."""
    gj = Path(path) if path else GEOJSON_PATH
    if not gj.exists():
        raise FileNotFoundError(f"Subzone GeoJSON not found: {gj}")
    sub = gpd.read_file(gj)
    if "PLN_AREA_N" not in sub.columns:
        raise ValueError("GeoJSON must contain a PLN_AREA_N column")
    sub = sub.set_crs("EPSG:4326") if sub.crs is None else sub.to_crs("EPSG:4326")
    sub = sub[["PLN_AREA_N", "geometry"]].copy()
    sub["PLN_AREA_N"] = sub["PLN_AREA_N"].astype(str).str.strip().str.upper()
    return sub


def join_taxis_to_areas(points: pd.DataFrame, subzones: gpd.GeoDataFrame | None = None):
    """Spatially join taxi points to planning areas.

    Returns (joined_points_gdf, counts_df) where counts_df has columns
    planning_area, available_taxi_count (sorted desc).
    """
    sub = subzones if subzones is not None else load_subzones()
    pts = gpd.GeoDataFrame(
        points.copy(),
        geometry=gpd.points_from_xy(points["longitude"], points["latitude"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(pts, sub, how="inner", predicate="intersects")
    counts = (
        joined.groupby("PLN_AREA_N").size()
        .rename("available_taxi_count").reset_index()
        .rename(columns={"PLN_AREA_N": "planning_area"})
        .sort_values("available_taxi_count", ascending=False)
        .reset_index(drop=True)
    )
    return joined, counts


def choropleth(counts: pd.DataFrame, subzones: gpd.GeoDataFrame | None = None,
               out_path: str | None = None, title: str = "Live taxi availability by planning area"):
    """Render a choropleth of taxi counts per planning area; save PNG if out_path."""
    import matplotlib.pyplot as plt
    sub = subzones if subzones is not None else load_subzones()
    area = sub.dissolve(by="PLN_AREA_N").reset_index()
    merged = area.merge(counts, left_on="PLN_AREA_N", right_on="planning_area", how="left")
    merged["available_taxi_count"] = merged["available_taxi_count"].fillna(0)
    fig, ax = plt.subplots(figsize=(9, 7))
    merged.plot(column="available_taxi_count", cmap="YlOrRd", linewidth=0.3,
                edgecolor="#888", legend=True, ax=ax,
                legend_kwds={"label": "available taxis", "shrink": 0.6})
    ax.set_title(title, fontsize=13); ax.axis("off"); fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=140, bbox_inches="tight"); print("saved", out_path)
    return ax
