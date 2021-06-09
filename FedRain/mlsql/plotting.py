import colorsys
import os
import subprocess
from functools import reduce
from operator import iand, ior
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import altair as alt
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ipywidgets import interact, interact_manual, widgets
from sklearn.preprocessing import LabelEncoder


def grid(
    df: pd.DataFrame,
    plot_func: Callable[..., Any],
    row_name: Optional[str] = None,
    col_name: Optional[str] = None,
    figsize: Tuple[int, int] = (4, 4),
    **kwargs: Any,
) -> None:

    if row_name is not None and col_name is not None:
        nrow = len(df[row_name].unique())
        ncol = len(df[col_name].unique())

        row_encoder = LabelEncoder().fit(df[row_name].unique())
        col_encoder = LabelEncoder().fit(df[col_name].unique())

        df = df.sort_values([row_name, col_name])
        rows = []
        for row, rdf in df.groupby(row_name):
            cols = []
            for col, group in rdf.groupby(col_name):
                (ridx,) = row_encoder.transform([row])
                (cidx,) = col_encoder.transform([col])

                chart = plot_func(group)
                cols.append(chart)
            rows.append(cols)
        rows = [reduce(ior, row) for row in rows]
        chart = reduce(iand, rows)

    elif row_name is not None and col_name is None:
        nrow = len(df[row_name].unique())
        ncol = 1

        row_encoder = LabelEncoder().fit(df[row_name].unique())

        df = df.sort_values([row_name])
        rows = []
        for row, group in df.groupby([row_name]):
            (ridx,) = row_encoder.transform([row])
            cidx = 0

            chart = plot_func(group)
            rows.append(chart)
        chart = reduce(iand, rows)

    elif row_name is None and col_name is not None:
        nrow = 1
        ncol = len(df[col_name].unique())

        col_encoder = LabelEncoder().fit(df[col_name].unique())

        df = df.sort_values([col_name])
        cols = []
        for col, group in df.groupby([col_name]):
            ridx = 0
            (cidx,) = col_encoder.transform([col])

            chart = plot_func(group)
            cols.append(chart)
        chart = reduce(ior, cols)

    elif row_name is None and col_name is None:
        chart = plot_func(df)

    return chart


def save(
    df: pd.DataFrame,
    plot_func: Callable[..., Any],
    row_name: Optional[str] = None,
    col_name: Optional[str] = None,
    path: str = ".",
    name_prefix: Optional[str] = None,
    **kwargs: Any,
) -> None:
    if row_name is not None and col_name is not None:
        for (row, col), group in df.groupby([row_name, col_name]):
            fig = plt.figure(**kwargs)
            plot_func(plt.gca(), group, row, col)
            fig.tight_layout(pad=0)
            if name_prefix is not None:
                name = "-".join([name_prefix, str(row), str(col)])
            else:
                name = "-".join([row, col])
            savepdfviasvg(fig, Path(path) / f"{name}")
            plt.close()
    elif row_name is not None and col_name is None:
        for row, group in df.groupby([row_name]):
            fig = plt.figure(**kwargs)
            plot_func(plt.gca(), group, row)
            fig.tight_layout(pad=0)
            if name_prefix is not None:
                name = "-".join([name_prefix, str(row)])
            else:
                name = row
            savepdfviasvg(fig, Path(path) / f"{name}")
            plt.close()
    elif row_name is None and col_name is not None:
        for col, group in df.groupby([col_name]):
            fig = plt.figure(**kwargs)
            plot_func(plt.gca(), group, col)
            fig.tight_layout(pad=0)
            if name_prefix is not None:
                name = "-".join([name_prefix, str(col)])
            else:
                name = col
            savepdfviasvg(fig, Path(path) / f"{name}")
            plt.close()
    elif row_name is None and col_name is None:
        fig = plt.figure(**kwargs)
        plot_func(plt.gca(), df)
        fig.tight_layout(pad=0)
        savepdfviasvg(fig, Path(path) / f"{name_prefix}")
        plt.close()


def iplot(
    df,
    func,
    partitions: List[str] = [],
    col_name: Optional[str] = None,
    row_name: Optional[str] = None,
    save_path: Optional[str] = None,
    name_prefix: Optional[str] = None,
    figsize: Tuple[int, int] = (4, 4),
):
    options = {}
    for partition in partitions:
        options[partition] = widgets.ToggleButtons(
            options=np.sort(df[partition].unique()),
            description=partition,
            disabled=False,
            button_style="",
        )

    options["__save"] = widgets.ToggleButtons(
        options=[False, True],
        description="Save Plots",
        disabled=False,
        button_style="warning",
    )

    def imp(__save: bool, **kwargs):
        mask = reduce(iand, [df[key] == value for key, value in kwargs.items()])
        subdf = df[mask]

        return grid(subdf, func, col_name=col_name, row_name=row_name, figsize=figsize)
        if __save:
            partition_params = "-".join([str(value) for value in kwargs.values()])
            (proc,) = df.proc.unique()
            prefix = proc.rstrip("Processor") + "-" + partition_params
            if name_prefix is not None:
                prefix = f"[{name_prefix}]{prefix}"
            save(
                subdf,
                func,
                col_name=col_name,
                row_name=row_name,
                figsize=figsize,
                path=save_path,
                name_prefix=prefix,
            )

    interact(imp, **options)


def normalize_array_lengths(arrays):
    maxlen = max(len(a) for a in arrays)
    normalized = np.empty((len(arrays), maxlen))
    for i, array in enumerate(arrays):
        normalized[i, :] = np.pad(
            array, (0, maxlen - len(array)), "constant", constant_values=array[-1]
        )
    return normalized


def convert_array(
    df: pd.DataFrame,
    cols: List[str] = ["deletions", "truth", "query_results", "elapses"],
):
    for col in cols:
        if col in df:
            df[col] = df[col].map(np.asarray)
    return df


def calc_recall(df: pd.DataFrame):
    recalls = []

    for idx in range(len(df)):
        deletions = df.loc[idx].deletions
        truth = df.loc[idx].truth
        deltas = np.ones((len(deletions), len(truth)))
        for i, deletion in enumerate(deletions):
            deltas[i:, deletion] = 0
        # Faster calculating the recall
        tp = ((1 - deltas).astype(bool) & truth).sum(axis=1)
        total = truth.sum()
        recall = tp / total

        recalls.append(recall)

    df["recall"] = recalls

    return df


def flatten_params(df: pd.DataFrame):
    df = df.join(df.params.apply(pd.Series))
    del df["params"]
    return df


def calc_AUCCR(df: pd.DataFrame):
    def imp(row):
        ntruth = row.truth.sum()
        return row.recall[:ntruth].sum() / ntruth * 2

    df["AUCCR"] = df.apply(imp, axis=1)
    return df


def expand_list_rows(df: pd.DataFrame, cols: List[str], npoints: int = 15):
    assert len(cols) > 0

    lens = df[cols[0]].str.len()
    for col in cols:
        assert (lens == df[col].str.len()).all()

    def calc_sample_mask(len):
        return np.arange(len) % int(len / npoints) == 0

    sample_mask = np.concatenate([calc_sample_mask(len) for len in lens])

    Ks = np.concatenate([np.arange(len) for len in lens])[sample_mask]

    repeated = pd.DataFrame(
        {
            col: np.repeat(df[col].values, lens)[sample_mask]
            for col in df.columns.drop(cols)
        }
    )
    repeated["K"] = Ks
    for key in cols:
        repeated[key] = np.concatenate(df[key].values)[sample_mask]

    return repeated


def savepdfviasvg(fig, name: Path, **kwargs):

    fig.savefig(name.with_suffix(".svg"), format="svg", **kwargs)
    incmd = ["inkscape", str(name.with_suffix(".svg")), f"--export-eps={name}.eps"]
    subprocess.check_output(incmd)
    os.remove(name.with_suffix(".svg"))


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
