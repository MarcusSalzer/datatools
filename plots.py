from plotly import express as px
from plotly import io as pio


def set_plotly_template(auto_size=False, w=600, h=300, transparent_background=True):
    """Some kind of plot template"""
    plot_temp = pio.templates["plotly_dark"]
    if not auto_size:
        plot_temp.layout.width = w
        plot_temp.layout.height = h
        plot_temp.layout.autosize = False
    if transparent_background:
        plot_temp.layout.paper_bgcolor = "rgba(0,0,0,0)"
        plot_temp.layout.plot_bgcolor = "rgba(0,0,0,0)"
    pio.templates.default = plot_temp


def plot_feature_pair(x_data, y_data, x=0, y=1, discrete=True):
    if discrete:
        labels = y_data.astype(str)
    else:
        labels = y_data

    fig = (
        px.scatter(
            x_data,
            x=x,
            y=y,
            color=labels,
            title="Feature space",
            width=500,
            height=400,
        )
        .update_traces(marker=dict(size=3))
        .update_layout(margin=dict(t=80, l=10, b=10, r=10))
    )
    return fig
