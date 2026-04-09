#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap

PALETTE = {
    "bg": "#F3F1EE",
    "surface": "#FFFFFF",
    "surface_soft": "#F8F7F4",
    "hero_bg": "#152A20",
    "hero_bg_2": "#1F3A2B",
    "text": "#1C1C1C",
    "text_muted": "#575A56",
    "accent": "#3A6B4A",
    "accent_2": "#2B5240",
    "accent_warm": "#C48B3A",
    "axis": "#CAD3CC",
    "bar_primary": "#2F5C43",
    "bar_secondary": "#C48B3A",
}

LSOA_BOUNDARY_SERVICE = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
    "Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BGC_V5/FeatureServer/0/query"
)


def gini(values: pd.Series | np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    if np.any(x < 0):
        raise ValueError("Gini is undefined for negative values")
    if np.allclose(x, 0):
        return 0.0
    x = np.sort(x)
    n = len(x)
    csum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(csum) / csum[-1]) / n)


def geometry_exteriors(geometry: dict) -> list[np.ndarray]:
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])
    exteriors: list[np.ndarray] = []
    if gtype == "Polygon":
        if coords:
            exteriors.append(np.asarray(coords[0], dtype=float))
    elif gtype == "MultiPolygon":
        for poly in coords:
            if poly:
                exteriors.append(np.asarray(poly[0], dtype=float))
    return [arr for arr in exteriors if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) >= 3]


def query_geojson(where_clause: str) -> dict:
    cmd = [
        "curl",
        "-k",
        "--retry",
        "5",
        "--retry-all-errors",
        "--retry-delay",
        "1",
        "--max-time",
        "60",
        "-L",
        "-sG",
        LSOA_BOUNDARY_SERVICE,
        "--data-urlencode",
        f"where={where_clause}",
        "--data-urlencode",
        "outFields=LSOA21CD,LSOA21NM,LAT,LONG",
        "--data-urlencode",
        "outSR=4326",
        "--data-urlencode",
        "f=geojson",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    if not res.stdout.strip():
        raise RuntimeError("Empty response from ArcGIS service")
    obj = json.loads(res.stdout)
    if "error" in obj:
        raise RuntimeError(f"ArcGIS query failed: {obj['error']}")
    return obj


def fetch_mk_geojson(lsoa_ids: list[str], chunk_size: int = 25) -> dict:
    features: dict[str, dict] = {}
    sorted_ids = sorted(set(lsoa_ids))
    for i in range(0, len(sorted_ids), chunk_size):
        chunk = sorted_ids[i : i + chunk_size]
        quoted = ",".join([f"'{x}'" for x in chunk])
        where = f"LSOA21CD IN ({quoted})"
        part = query_geojson(where)
        for feat in part.get("features", []):
            code = feat.get("properties", {}).get("LSOA21CD")
            if code:
                features[code] = feat

    return {"type": "FeatureCollection", "features": list(features.values())}


def map_bounds(features: list[dict]) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for feat in features:
        for arr in geometry_exteriors(feat.get("geometry", {})):
            xs.extend(arr[:, 0].tolist())
            ys.extend(arr[:, 1].tolist())
    return min(xs), max(xs), min(ys), max(ys)


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(PALETTE["surface"])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _grid_step(span: float) -> float:
    if span <= 0.25:
        return 0.05
    if span <= 0.7:
        return 0.1
    if span <= 1.5:
        return 0.2
    return 0.5


def _format_lon(value: float) -> str:
    hemi = "E" if value >= 0 else "W"
    return f"{abs(value):.2f}\N{DEGREE SIGN}{hemi}"


def _format_lat(value: float) -> str:
    hemi = "N" if value >= 0 else "S"
    return f"{abs(value):.2f}\N{DEGREE SIGN}{hemi}"


def add_geo_grid(ax: plt.Axes, xmin: float, xmax: float, ymin: float, ymax: float) -> None:
    xspan = xmax - xmin
    yspan = ymax - ymin
    xstep = _grid_step(xspan)
    ystep = _grid_step(yspan)

    xstart = math.floor(xmin / xstep) * xstep
    ystart = math.floor(ymin / ystep) * ystep
    xend = math.ceil(xmax / xstep) * xstep
    yend = math.ceil(ymax / ystep) * ystep

    xticks = np.arange(xstart, xend + xstep * 0.5, xstep)
    yticks = np.arange(ystart, yend + ystep * 0.5, ystep)
    xticks = xticks[(xticks >= xmin - 1e-9) & (xticks <= xmax + 1e-9)]
    yticks = yticks[(yticks >= ymin - 1e-9) & (yticks <= ymax + 1e-9)]

    ax.set_xticks(xticks.tolist())
    ax.set_yticks(yticks.tolist())
    ax.set_xticklabels([_format_lon(v) for v in xticks], color=PALETTE["text_muted"], fontsize=8)
    ax.set_yticklabels([_format_lat(v) for v in yticks], color=PALETTE["text_muted"], fontsize=8)
    ax.grid(True, color=PALETTE["axis"], alpha=0.55, linestyle=(0, (3, 3)), linewidth=0.7)
    ax.set_xlabel("Longitude", color=PALETTE["text_muted"], fontsize=9)
    ax.set_ylabel("Latitude", color=PALETTE["text_muted"], fontsize=9)
    ax.tick_params(axis="both", length=0)

    for side in ["left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(PALETTE["axis"])
        ax.spines[side].set_linewidth(0.9)


def plot_fuel_poverty_map(geojson: dict, mk: pd.DataFrame, out_png: Path) -> None:
    value_map = mk.set_index("lsoa_id")["fuel_poverty_rate_effective"].to_dict()
    polys: list[np.ndarray] = []
    vals: list[float] = []

    for feat in geojson.get("features", []):
        code = feat.get("properties", {}).get("LSOA21CD")
        if code not in value_map:
            continue
        for arr in geometry_exteriors(feat.get("geometry", {})):
            polys.append(arr)
            vals.append(float(value_map[code]) * 100.0)

    cmap = LinearSegmentedColormap.from_list(
        "mk_poverty",
        ["#E8EEE9", "#A6C1AE", "#5F8E72", PALETTE["bar_primary"], PALETTE["hero_bg"]],
    )

    fig, ax = plt.subplots(figsize=(8.2, 8.2), facecolor=PALETTE["bg"])
    style_axis(ax)
    coll = PolyCollection(
        polys,
        array=np.asarray(vals),
        cmap=cmap,
        edgecolors=PALETTE["surface"],
        linewidths=0.25,
    )
    ax.add_collection(coll)
    xmin, xmax, ymin, ymax = map_bounds(geojson.get("features", []))
    padx = (xmax - xmin) * 0.03
    pady = (ymax - ymin) * 0.03
    ax.set_xlim(xmin - padx, xmax + padx)
    ax.set_ylim(ymin - pady, ymax + pady)
    ax.set_aspect("equal", adjustable="box")
    add_geo_grid(ax, xmin - padx, xmax + padx, ymin - pady, ymax + pady)

    cbar = fig.colorbar(coll, ax=ax, fraction=0.034, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=PALETTE["text_muted"], labelsize=9)
    cbar.set_label("Fuel poverty rate (%)", color=PALETTE["text_muted"], fontsize=10)

    fig.suptitle("1) Milton Keynes Energy Poverty (LSOA baseline)", color=PALETTE["text"], fontsize=14, y=0.97)
    fig.text(
        0.02,
        0.02,
        "Source: lsoa_master_table_v4.csv (fuel_poverty_rate_effective).",
        color=PALETTE["text_muted"],
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.90))
    fig.savefig(out_png, dpi=230, bbox_inches="tight")
    plt.close(fig)


def plot_deployment_map(geojson: dict, mk: pd.DataFrame, scenario: str, out_png: Path) -> None:
    mk_map = mk.set_index("lsoa_id")

    base_polys: list[np.ndarray] = []
    all_lon: list[float] = []
    all_lat: list[float] = []
    active_lon: list[float] = []
    active_lat: list[float] = []
    active_mw: list[float] = []

    for feat in geojson.get("features", []):
        props = feat.get("properties", {})
        code = props.get("LSOA21CD")
        for arr in geometry_exteriors(feat.get("geometry", {})):
            base_polys.append(arr)
        if code not in mk_map.index:
            continue
        row = mk_map.loc[code]
        lon = props.get("LONG")
        lat = props.get("LAT")
        if lon is None or lat is None:
            ext = geometry_exteriors(feat.get("geometry", {}))
            if not ext:
                continue
            lon = float(np.mean(ext[0][:, 0]))
            lat = float(np.mean(ext[0][:, 1]))
        all_lon.append(float(lon))
        all_lat.append(float(lat))

        mw = float(row["asset_total_mw"])
        if mw > 0:
            active_lon.append(float(lon))
            active_lat.append(float(lat))
            active_mw.append(mw)

    fig, ax = plt.subplots(figsize=(8.2, 8.2), facecolor=PALETTE["bg"])
    style_axis(ax)

    if base_polys:
        bg = PolyCollection(
            base_polys,
            facecolor=PALETTE["surface_soft"],
            edgecolors=PALETTE["axis"],
            linewidths=0.22,
        )
        ax.add_collection(bg)

    ax.scatter(all_lon, all_lat, s=8, color=PALETTE["axis"], alpha=0.65, linewidths=0)

    if active_lon:
        sizes = [28 + 22 * mw for mw in active_mw]
        sc = ax.scatter(
            active_lon,
            active_lat,
            s=sizes,
            c=active_mw,
            cmap=LinearSegmentedColormap.from_list("mw", ["#8FB199", PALETTE["bar_primary"]]),
            edgecolors=PALETTE["surface"],
            linewidths=0.55,
            alpha=0.92,
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.034, pad=0.02)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(colors=PALETTE["text_muted"], labelsize=9)
        cbar.set_label("Community capacity per LSOA (MW)", color=PALETTE["text_muted"], fontsize=10)

    xmin, xmax, ymin, ymax = map_bounds(geojson.get("features", []))
    padx = (xmax - xmin) * 0.03
    pady = (ymax - ymin) * 0.03
    ax.set_xlim(xmin - padx, xmax + padx)
    ax.set_ylim(ymin - pady, ymax + pady)
    ax.set_aspect("equal", adjustable="box")
    add_geo_grid(ax, xmin - padx, xmax + padx, ymin - pady, ymax + pady)

    fig.suptitle(f"2) {scenario}: Energy Community Deployment in Milton Keynes", color=PALETTE["text"], fontsize=14, y=0.97)
    fig.text(
        0.02,
        0.02,
        f"Active LSOAs: {(mk['asset_total_mw'] > 0).sum()} of {len(mk)} | Total capacity: {mk['asset_total_mw'].sum():.2f} MW",
        color=PALETTE["text_muted"],
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.90))
    fig.savefig(out_png, dpi=230, bbox_inches="tight")
    plt.close(fig)


def plot_outcome_story(roll: pd.DataFrame, co2_kt: float, scenario: str, out_png: Path) -> None:
    years = roll["year"].to_numpy()
    fuel_poverty = roll["equity_fuel_poverty_reduction_pct_proxy"].to_numpy()
    flex = roll["flex_peak_reduction_mw_proxy"].to_numpy()
    demand_pct = (roll["edr_demand_reduction_pct_proxy"].to_numpy() * 100.0)

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.1), facecolor=PALETTE["bg"])
    for ax in axes:
        ax.set_facecolor(PALETTE["surface"])
        ax.grid(axis="y", color=PALETTE["axis"], alpha=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(PALETTE["axis"])
        ax.spines["bottom"].set_color(PALETTE["axis"])
        ax.tick_params(colors=PALETTE["text_muted"], labelsize=9)

    axes[0].plot(years, fuel_poverty, color=PALETTE["bar_secondary"], marker="o", linewidth=2.2)
    axes[0].set_title("Fuel Poverty Reduction (%)", color=PALETTE["text"], fontsize=11)
    axes[0].set_ylim(0, max(105, np.max(fuel_poverty) * 1.08))

    axes[1].plot(years, flex, color=PALETTE["bar_primary"], marker="o", linewidth=2.2)
    axes[1].set_title("Flexibility (Peak MW Reduced)", color=PALETTE["text"], fontsize=11)
    axes[1].set_ylim(0, np.max(flex) * 1.12)

    axes[2].plot(years, demand_pct, color=PALETTE["accent"], marker="o", linewidth=2.2)
    axes[2].set_title("Demand Reduction Proxy (%)", color=PALETTE["text"], fontsize=11)
    axes[2].set_ylim(0, np.max(demand_pct) * 1.25)

    for ax in axes:
        ax.set_xticks([2025, 2030, 2035, 2040, 2045, 2050])

    axes[1].annotate(
        f"Avoided CO2 proxy: {co2_kt:.2f} kt/window",
        xy=(years[-1], flex[-1]),
        xytext=(-210, -36),
        textcoords="offset points",
        fontsize=9,
        color=PALETTE["text"],
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#F2E6D7", "edgecolor": PALETTE["bar_secondary"]},
    )

    fig.suptitle(f"3) Milton Keynes Outcomes Trajectory to 2050 ({scenario})", color=PALETTE["text"], fontsize=14, y=0.98)
    fig.text(
        0.01,
        0.02,
        "Fuel poverty, flexibility, and demand reduction from 14_rollout_lad_metrics.csv; CO2 from 05_counterfactual_delta_summary.csv.",
        color=PALETTE["text_muted"],
        fontsize=8.8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(out_png, dpi=230)
    plt.close(fig)


def plot_gini_story(metrics: pd.DataFrame, baseline_fp_gini: float, out_png: Path) -> None:
    x = np.arange(len(metrics))

    fig, ax1 = plt.subplots(figsize=(9.8, 4.6), facecolor=PALETTE["bg"])
    ax1.set_facecolor(PALETTE["surface"])
    bars = ax1.bar(
        x,
        metrics["deployment_mw_gini"].to_numpy(),
        color=PALETTE["bar_primary"],
        width=0.58,
        label="Deployment MW Gini",
    )
    ax1.axhline(baseline_fp_gini, color=PALETTE["bar_secondary"], linestyle="--", linewidth=1.9, label="Baseline fuel-poverty Gini")

    ax1.set_xticks(x, metrics["scenario_code"].tolist())
    ax1.set_ylabel("Gini coefficient", color=PALETTE["text"])
    ax1.tick_params(axis="x", colors=PALETTE["text"])
    ax1.tick_params(axis="y", colors=PALETTE["text_muted"])
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis="y", color=PALETTE["axis"], alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        metrics["targeting_ratio_top30"].to_numpy(),
        color=PALETTE["bar_secondary"],
        marker="o",
        linewidth=2.2,
        label="Targeting ratio (top-30% poverty LSOAs)",
    )
    ax2.set_ylabel("Targeting ratio (>1 means progressive targeting)", color=PALETTE["text"])
    ax2.tick_params(axis="y", colors=PALETTE["text_muted"])
    ax2.set_ylim(0, max(2.0, metrics["targeting_ratio_top30"].max() * 1.2))

    for bar in bars:
        y = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            y + 0.02,
            f"{y:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=PALETTE["text_muted"],
        )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=False, fontsize=9)

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_color(PALETTE["axis"])
    ax1.spines["bottom"].set_color(PALETTE["axis"])

    fig.suptitle("4) Equity Lens: Gini + Targeting Metric", color=PALETTE["text"], fontsize=14, y=0.995)
    fig.text(
        0.01,
        0.015,
        "Deployment Gini near 1 = concentrated assets. Targeting ratio compares MW share in highest-poverty 30% LSOAs vs population share.",
        color=PALETTE["text_muted"],
        fontsize=8.8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.90))
    fig.savefig(out_png, dpi=230, bbox_inches="tight")
    plt.close(fig)


def build_story_assets(ecmodel_root: Path, run_subpath: str, output_root: Path, scenario: str) -> dict:
    run_dir = ecmodel_root / run_subpath
    master_path = ecmodel_root / "data/processed/lsoa_master_table_v4.csv"
    deploy_path = run_dir / "08_deployment_socio_by_lsoa.csv"
    rollout_path = run_dir / "14_rollout_lad_metrics.csv"
    counterf_path = run_dir / "05_counterfactual_delta_summary.csv"

    out_data = output_root / "data"
    out_charts = output_root / "charts"
    out_data.mkdir(parents=True, exist_ok=True)
    out_charts.mkdir(parents=True, exist_ok=True)

    baseline = pd.read_csv(master_path)
    mk_base = baseline[baseline["lad_name"] == "Milton Keynes"].copy()

    deploy = pd.read_csv(deploy_path)
    dep_s = deploy[deploy["scenario_code"] == scenario].copy()

    mk = mk_base[[
        "lsoa_id",
        "lsoa_name",
        "population_effective",
        "fuel_poverty_rate_effective",
        "deprivation_score",
        "income_effective",
        "base_demand_kw",
    ]].merge(
        dep_s[["lsoa_id", "asset_total_mw", "ec_count"]],
        on="lsoa_id",
        how="left",
    )
    mk[["asset_total_mw", "ec_count"]] = mk[["asset_total_mw", "ec_count"]].fillna(0.0)
    mk["active"] = mk["asset_total_mw"] > 0
    mk["fuel_poor_households_proxy"] = mk["population_effective"] * mk["fuel_poverty_rate_effective"]

    geo = fetch_mk_geojson(mk["lsoa_id"].tolist())

    # Attach model values to GeoJSON properties for direct web map use.
    value_lookup = mk.set_index("lsoa_id")[[
        "fuel_poverty_rate_effective",
        "asset_total_mw",
        "ec_count",
        "population_effective",
        "deprivation_score",
        "active",
    ]].to_dict("index")
    for feat in geo.get("features", []):
        code = feat.get("properties", {}).get("LSOA21CD")
        if code in value_lookup:
            feat["properties"].update(value_lookup[code])

    geojson_path = out_data / f"mk_lsoa_boundaries_with_metrics_{scenario.lower()}.geojson"
    geojson_path.write_text(json.dumps(geo), encoding="utf-8")

    lsoa_csv = out_data / f"mk_story_lsoa_metrics_{scenario.lower()}.csv"
    mk.sort_values("lsoa_id").to_csv(lsoa_csv, index=False)

    rollout = pd.read_csv(rollout_path)
    mk_roll = rollout[(rollout["lad_name"] == "Milton Keynes") & (rollout["scenario_code"] == scenario)].copy()
    mk_roll = mk_roll.sort_values("year")
    outcomes_csv = out_data / f"mk_story_outcomes_{scenario.lower()}.csv"
    mk_roll[[
        "year",
        "equity_fuel_poverty_reduction_pct_proxy",
        "flex_peak_reduction_mw_proxy",
        "edr_demand_reduction_pct_proxy",
        "composite_score_0_100",
    ]].to_csv(outcomes_csv, index=False)

    counterf = pd.read_csv(counterf_path)
    co2_kt = float(counterf.loc[counterf["scenario_code"] == scenario, "delta_avoided_co2_tonnes_proxy_window"].iloc[0] / 1000.0)

    gini_rows = []
    baseline_fp_gini = gini(mk["fuel_poor_households_proxy"])
    for sc in ["SA", "SB", "SC", "SD"]:
        dep_sc = deploy[deploy["scenario_code"] == sc][["lsoa_id", "asset_total_mw"]]
        d = mk_base[["lsoa_id", "population_effective", "fuel_poverty_rate_effective"]].merge(dep_sc, on="lsoa_id", how="left")
        d["asset_total_mw"] = d["asset_total_mw"].fillna(0.0)
        d["fuel_poor_households_proxy"] = d["population_effective"] * d["fuel_poverty_rate_effective"]
        top_cut = d["fuel_poor_households_proxy"].quantile(0.7)
        top = d[d["fuel_poor_households_proxy"] >= top_cut]
        share_mw = float(top["asset_total_mw"].sum() / d["asset_total_mw"].sum()) if d["asset_total_mw"].sum() > 0 else 0.0
        share_pop = float(top["population_effective"].sum() / d["population_effective"].sum())
        gini_rows.append(
            {
                "scenario_code": sc,
                "deployment_mw_gini": gini(d["asset_total_mw"]),
                "fuel_poverty_proxy_gini": baseline_fp_gini,
                "targeting_ratio_top30": share_mw / share_pop if share_pop > 0 else np.nan,
                "poverty_deployment_corr": float(np.corrcoef(d["fuel_poor_households_proxy"], d["asset_total_mw"])[0, 1]),
                "active_lsoas": int((d["asset_total_mw"] > 0).sum()),
                "total_mw": float(d["asset_total_mw"].sum()),
            }
        )
    gini_df = pd.DataFrame(gini_rows)
    gini_csv = out_data / "mk_story_gini_metrics.csv"
    gini_df.to_csv(gini_csv, index=False)

    c1 = out_charts / "mk_story_1_fuel_poverty_map.png"
    c2 = out_charts / f"mk_story_2_deployment_map_{scenario.lower()}.png"
    c3 = out_charts / f"mk_story_3_outcomes_{scenario.lower()}.png"
    c4 = out_charts / "mk_story_4_gini_targeting.png"

    plot_fuel_poverty_map(geo, mk, c1)
    plot_deployment_map(geo, mk, scenario, c2)
    plot_outcome_story(mk_roll, co2_kt, scenario, c3)
    plot_gini_story(gini_df, baseline_fp_gini, c4)

    summary = {
        "scenario": scenario,
        "run_dir": str(run_dir),
        "lsoas_total": int(len(mk)),
        "lsoas_active": int(mk["active"].sum()),
        "total_capacity_mw": float(mk["asset_total_mw"].sum()),
        "co2_kt_proxy_window": co2_kt,
        "outputs": {
            "geojson": str(geojson_path),
            "lsoa_metrics_csv": str(lsoa_csv),
            "outcomes_csv": str(outcomes_csv),
            "gini_csv": str(gini_csv),
            "chart_1_poverty_map": str(c1),
            "chart_2_deployment_map": str(c2),
            "chart_3_outcomes": str(c3),
            "chart_4_gini": str(c4),
        },
    }

    summary_path = out_data / "mk_story_chart_pack_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Milton Keynes 4-chart stakeholder story assets")
    parser.add_argument(
        "--ecmodel-root",
        default="/Users/federicogambedotti/Documents/GitHub/ECModel",
        help="Absolute path to ECModel repository",
    )
    parser.add_argument(
        "--run-subpath",
        default="outputs/policy_benchmark/run_archive/20260408_074422__50_lsoa_smoketest",
        help="Run directory under ECModel root",
    )
    parser.add_argument(
        "--output-root",
        default="/Users/federicogambedotti/Documents/GitHub/PEM-website",
        help="Website repo root where data/charts will be written",
    )
    parser.add_argument(
        "--scenario",
        default="SD",
        choices=["SA", "SB", "SC", "SD"],
        help="Scenario used for maps and trajectory chart",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_story_assets(
        ecmodel_root=Path(args.ecmodel_root),
        run_subpath=args.run_subpath,
        output_root=Path(args.output_root),
        scenario=args.scenario,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
