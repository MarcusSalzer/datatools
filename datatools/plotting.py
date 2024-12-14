import numpy as np
from plotly import graph_objects as go
from plotly import io as pio


def set_plotly_template(auto_size=False, w=600, h=300, transparent_background=True):
    """Some kind of plot template"""
    plot_temp = pio.templates["plotly_dark"]
    plot_temp.layout.margin = dict(t=0, l=0, r=0, b=0)

    if not auto_size:
        plot_temp.layout.width = w
        plot_temp.layout.height = h
        plot_temp.layout.autosize = False
    if transparent_background:
        plot_temp.layout.paper_bgcolor = "rgba(0,0,0,0)"
        plot_temp.layout.plot_bgcolor = "rgba(0,0,0,0)"
    pio.templates.default = plot_temp


def plot_feature_pair(x_data, y_data, x=0, y=1, discrete=True):
    raise NotImplementedError
    if discrete:
        labels = y_data.astype(str)
    else:
        labels = y_data

    fig = go.Figure()
    return fig


def heatmap(
    X: np.ndarray,
    labels: list[str] = None,
    log_scale=False,
    pseudo_count=1,
    size=400,
):
    """Plot a matrix as a heatmap, optionally in log-scale to compress range."""
    if log_scale:
        z = np.log(X + pseudo_count)
    else:
        z = X

    if labels is None:
        return go.Figure(
            go.Heatmap(
                z=z,
                text=X,
            ),
            dict(
                xaxis=dict(title="prediction"),
                yaxis=dict(title="true", scaleanchor="x"),
            ),
        )

    return go.Figure(
        go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            text=X,
        ),
        dict(
            xaxis=dict(title="prediction", type="category", dtick=1),
            yaxis=dict(title="true", scaleanchor="x", type="category", dtick=1),
            width=size + 80,
            height=size,
        ),
    )
