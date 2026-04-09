#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap

PALETTE = {
    "surface": "#FFFFFF",
    "text": "#1C1C1C",
    "text_muted": "#5A5A5A",
    "accent": "#3A6B4A",
    "accent_warm": "#C48B3A",
    "border": "#D6D1CA",
    "hero": "#1A2E22",
}

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw_data"
OUT = ROOT


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


def map_bounds(features: list[dict]) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for feat in features:
        for arr in geometry_exteriors(feat.get("geometry", {})):
            xs.extend(arr[:, 0].tolist())
            ys.extend(arr[:, 1].tolist())
    return min(xs), max(xs), min(ys), max(ys)


def chart_1_poverty_map(geo: dict) -> None:
    polys: list[np.ndarray] = []
    vals: list[float] = []
    for feat in geo.get("features", []):
        rate = feat.get("properties", {}).get("fuel_poverty_rate_effective")
        if rate is None:
            continue
        for arr in geometry_exteriors(feat.get("geometry", {})):
            polys.append(arr)
            vals.append(float(rate) * 100.0)

    cmap = LinearSegmentedColormap.from_list(
        "fuel_poverty",
        ["#EAF1EC", "#A7C2AE", "#5F8C72", PALETTE["accent"], PALETTE["hero"]],
    )

    fig = plt.figure(figsize=(10, 6.4), facecolor=PALETTE["surface"])
    ax = fig.add_axes([0.05, 0.08, 0.80, 0.84])
    ax.set_facecolor(PALETTE["surface"])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    coll = PolyCollection(
        polys,
        array=np.asarray(vals),
        cmap=cmap,
        edgecolors=PALETTE["surface"],
        linewidths=0.35,
    )
    ax.add_collection(coll)
    xmin, xmax, ymin, ymax = map_bounds(geo.get("features", []))
    padx = (xmax - xmin) * 0.03
    pady = (ymax - ymin) * 0.03
    ax.set_xlim(xmin - padx, xmax + padx)
    ax.set_ylim(ymin - pady, ymax + pady)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        "Fuel poverty baseline by LSOA (%)",
        loc="left",
        fontsize=13,
        color=PALETTE["text"],
        pad=8,
    )

    cax = fig.add_axes([0.88, 0.16, 0.025, 0.68])
    cbar = fig.colorbar(coll, cax=cax)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=PALETTE["text_muted"], labelsize=10)
    cbar.set_label("Fuel poverty rate", color=PALETTE["text_muted"], fontsize=10)

    fig.savefig(OUT / "mk_story_1_fuel_poverty_map.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def chart_2_deployment_map(geo: dict) -> None:
    base_polys: list[np.ndarray] = []
    all_lon: list[float] = []
    all_lat: list[float] = []
    active_lon: list[float] = []
    active_lat: list[float] = []
    active_mw: list[float] = []

    for feat in geo.get("features", []):
        props = feat.get("properties", {})
        for arr in geometry_exteriors(feat.get("geometry", {})):
            base_polys.append(arr)

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

        mw = float(props.get("asset_total_mw", 0.0))
        if mw > 0:
            active_lon.append(float(lon))
            active_lat.append(float(lat))
            active_mw.append(mw)

    fig = plt.figure(figsize=(10, 6.4), facecolor=PALETTE["surface"])
    ax = fig.add_axes([0.05, 0.08, 0.80, 0.84])
    ax.set_facecolor(PALETTE["surface"])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    bg = PolyCollection(
        base_polys,
        facecolor="#F2F0EC",
        edgecolors=PALETTE["border"],
        linewidths=0.28,
    )
    ax.add_collection(bg)

    ax.scatter(all_lon, all_lat, s=10, color=PALETTE["border"], alpha=0.75, linewidths=0)
    sc = ax.scatter(
        active_lon,
        active_lat,
        s=[36 + 24 * mw for mw in active_mw],
        c=active_mw,
        cmap=LinearSegmentedColormap.from_list("deploy", ["#9CB9A5", PALETTE["accent"]]),
        edgecolors=PALETTE["surface"],
        linewidths=0.6,
        alpha=0.95,
    )

    xmin, xmax, ymin, ymax = map_bounds(geo.get("features", []))
    padx = (xmax - xmin) * 0.03
    pady = (ymax - ymin) * 0.03
    ax.set_xlim(xmin - padx, xmax + padx)
    ax.set_ylim(ymin - pady, ymax + pady)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        "Energy community deployment (MW per active LSOA)",
        loc="left",
        fontsize=13,
        color=PALETTE["text"],
        pad=8,
    )

    cax = fig.add_axes([0.88, 0.16, 0.025, 0.68])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=PALETTE["text_muted"], labelsize=10)
    cbar.set_label("Capacity (MW)", color=PALETTE["text_muted"], fontsize=10)

    fig.savefig(OUT / "mk_story_2_deployment_map_sd.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def chart_3_outcomes(outcomes: pd.DataFrame) -> None:
    years = outcomes["year"].to_numpy()
    fuel = outcomes["equity_fuel_poverty_reduction_pct_proxy"].to_numpy()
    peak = outcomes["flex_peak_reduction_mw_proxy"].to_numpy()
    demand = (outcomes["edr_demand_reduction_pct_proxy"].to_numpy() * 100.0)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 6.4),
        sharex=True,
        facecolor=PALETTE["surface"],
        gridspec_kw={"hspace": 0.18},
    )

    rows = [
        (fuel, PALETTE["accent_warm"], "Fuel poverty reduction (%)"),
        (peak, PALETTE["accent"], "Peak reduction (MW)"),
        (demand, "#2F5A43", "Demand reduction proxy (%)"),
    ]

    for ax, (series, color, label) in zip(axes, rows):
        ax.set_facecolor(PALETTE["surface"])
        ax.plot(years, series, color=color, linewidth=2.4, marker="o", markersize=3.8)
        ax.fill_between(years, 0, series, color=color, alpha=0.12)
        ax.grid(axis="y", color=PALETTE["border"], linewidth=0.8, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(PALETTE["border"])
        ax.spines["bottom"].set_color(PALETTE["border"])
        ax.tick_params(axis="y", colors=PALETTE["text_muted"], labelsize=9)
        ax.tick_params(axis="x", colors=PALETTE["text_muted"], labelsize=9)
        ax.set_ylabel(label, color=PALETTE["text"], fontsize=9)
        ax.text(
            years[-1] + 0.25,
            series[-1],
            f"{series[-1]:.2f}",
            color=color,
            fontsize=9,
            va="center",
            ha="left",
        )

    axes[-1].set_xticks([2025, 2030, 2035, 2040, 2045, 2050])
    axes[-1].set_xlim(years.min(), years.max() + 2)
    fig.suptitle("Outcomes trajectory to 2050", x=0.07, y=0.98, ha="left", fontsize=13, color=PALETTE["text"])
    fig.savefig(OUT / "mk_story_3_outcomes_sd.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def chart_4_gini(gini_df: pd.DataFrame) -> None:
    scenarios = gini_df["scenario_code"].to_numpy()
    gini_vals = gini_df["deployment_mw_gini"].to_numpy()
    targeting_vals = gini_df["targeting_ratio_top30"].to_numpy()
    baseline = float(gini_df["fuel_poverty_proxy_gini"].iloc[0])

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(10, 6.4),
        facecolor=PALETTE["surface"],
        gridspec_kw={"wspace": 0.23},
    )
    x = np.arange(len(scenarios))

    for ax in (ax1, ax2):
        ax.set_facecolor(PALETTE["surface"])
        ax.grid(axis="y", color=PALETTE["border"], linewidth=0.8, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(PALETTE["border"])
        ax.spines["bottom"].set_color(PALETTE["border"])
        ax.tick_params(colors=PALETTE["text_muted"], labelsize=9)

    bars = ax1.bar(x, gini_vals, color=PALETTE["accent"], width=0.55)
    ax1.axhline(baseline, color=PALETTE["accent_warm"], linestyle="--", linewidth=1.8)
    ax1.set_xticks(x, scenarios)
    ax1.set_ylim(0, 1.0)
    ax1.set_title("Deployment concentration (Gini)", fontsize=11, color=PALETTE["text"], pad=8)
    ax1.set_ylabel("Gini coefficient", color=PALETTE["text"], fontsize=9)
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=9, color=PALETTE["text_muted"])

    bars2 = ax2.bar(x, targeting_vals, color=PALETTE["accent_warm"], width=0.55)
    ax2.axhline(1.0, color=PALETTE["text_muted"], linestyle="--", linewidth=1.5)
    ax2.set_xticks(x, scenarios)
    ax2.set_ylim(0, max(2.0, float(targeting_vals.max()) * 1.20))
    ax2.set_title("Targeting ratio (top-30% poverty LSOAs)", fontsize=11, color=PALETTE["text"], pad=8)
    ax2.set_ylabel("Ratio (>1 means progressive targeting)", color=PALETTE["text"], fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.04, f"{h:.2f}", ha="center", va="bottom", fontsize=9, color=PALETTE["text_muted"])

    fig.suptitle("Equity and targeting by scenario", x=0.07, y=0.98, ha="left", fontsize=13, color=PALETTE["text"])
    fig.savefig(OUT / "mk_story_4_gini_targeting.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.titleweight"] = "semibold"

    geo = json.loads((RAW / "mk_lsoa_boundaries_with_metrics_sd.geojson").read_text())
    outcomes = pd.read_csv(RAW / "mk_story_outcomes_sd.csv").sort_values("year")
    gini_df = pd.read_csv(RAW / "mk_story_gini_metrics.csv")

    chart_1_poverty_map(geo)
    chart_2_deployment_map(geo)
    chart_3_outcomes(outcomes)
    chart_4_gini(gini_df)

    print("Regenerated charts:")
    print(" -", OUT / "mk_story_1_fuel_poverty_map.png")
    print(" -", OUT / "mk_story_2_deployment_map_sd.png")
    print(" -", OUT / "mk_story_3_outcomes_sd.png")
    print(" -", OUT / "mk_story_4_gini_targeting.png")


if __name__ == "__main__":
    main()
