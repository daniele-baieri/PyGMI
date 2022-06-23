import functools
import plotly.graph_objects as go
import pygmi.utils.extract
from torch import Tensor
from numpy import ndarray
from plotly.subplots import make_subplots
from typing import Callable, Dict, Tuple, Union
from pygmi.utils import label_to_interval


def validate_figure(func: Callable):
    """Decorator allowing to call plotting functions without
    passing a Figure - it is automatically created, passed to 
    the plotting function and returned by the decorator.

    Parameters
    ----------
    func : Callable
        A plotting function from ngt.utils.visual
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if 'fig' not in kwargs.keys():
            ffig = None
            newargs = []
            for arg in args:
                if type(arg) == go.Figure():
                    ffig = arg
                else:
                    newargs.append(arg)
            kwargs['fig'] = go.Figure() if ffig is None else ffig 
        func(*newargs, **kwargs)
        return kwargs['fig']
    return wrapper

def make_3d_subplots(rows: int = 1, cols: int = 1) -> go.Figure:
    """Creates a plotly Figure with 3D subplots.

    Parameters
    ----------
    rows : int, optional
        Number of rows in subplot grid, by default 1
    cols : int, optional
        Number of columns in subplot grid, by default 1

    Returns
    -------
    go.Figure
        Plotly Figure containing the subplots
    """    
    return make_subplots(
        rows, cols, specs=[[{'type': 'surface'}] * cols] * rows
    )

@validate_figure
def plot_trisurf(
    vert: Union[Tensor, ndarray], 
    triv: Union[Tensor, ndarray], 
    fig: go.Figure = None
) -> go.Figure:
    """Plots a 3D mesh.

    Parameters
    ----------
    vert : Union[Tensor, ndarray]
        Vertices of the mesh
    triv : Union[Tensor, ndarray]
        Triangles of the mesh
    fig : go.Figure, optional
        Figure to append plot to, by default None

    Returns
    -------
    go.Figure
        Either `fig` or a new `go.Figure` containing the plot
    """    
    fig.add_trace(
        go.Mesh3d(
            x=vert[:, 0], y=vert[:, 1], z=vert[:, 2], 
            i=triv[:, 0], j=triv[:, 1], k=triv[:, 2]
        )
    )

@validate_figure
def plot_isosurfaces(
    grid_coords: Union[Tensor, ndarray], 
    F: Union[Tensor, ndarray],
    min_level: float = -0.5,
    max_level: float = 0.5,
    num_surfs: int = 3,
    fig: go.Figure = None
) -> go.Figure:
    """Plots multiple isosurfaces extracted from an implicit function.

    Parameters
    ----------
    grid_coords : Union[Tensor, ndarray]
        Point coordinates of the grid
    F : Union[Tensor, ndarray]
        Implicit function evaluated on `grid_coords`
    min_level : float, optional
        Minimum function level, by default -0.5
    max_level : float, optional
        Maximum function level, by default 0.5
    num_surfs : int, optional
        Number of isosurfaces to extract, with linearly spaced levels
        between `min_level` and `max_level`, by default 3
    fig : go.Figure, optional
        Figure to append plot to, by default None

    Returns
    -------
    go.Figure
        Either `fig` or a new `go.Figure` containing the plot
    """    
    fig.add_trace(go.Volume(
        x=grid_coords[:, 0],
        y=grid_coords[:, 1],
        z=grid_coords[:, 2],
        value=F,
        isomin=min_level,
        max_level=max_level,
        surface_count=num_surfs,
        opacity=0.1
    ))

@validate_figure
def isosurf_animation(
    F_volume: Union[Tensor, ndarray],
    min_level: float = -0.5,
    max_level: float = 0.5,
    axes: Tuple[float] = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
    steps: int = 3,
    fig: go.Figure = None
) -> go.Figure:
    """Plots an animated figure allowing to singularly inspect level surfaces
    of any given implicit function.

    Parameters
    ----------
    F_volume : Union[Tensor, ndarray]
        `N x N x N` tensor containing function values
    min_level : float, optional
        Minimum surface level, by default -0.5
    max_level : float, optional
        Maximum surface level, by default 0.5
    steps : int, optional
        Number of linear steps between `min_level` and `max_level`, by default 3
    fig : go.Figure, optional
        Figure to append plot to, by default None

    Returns
    -------
    go.Figure
        Either `fig` or a new `go.Figure` containing the plot
    """    
    frames = []
    voxel_size = (2.0) / (F_volume.shape[0] - 1)
    for s in range(steps):
        t = label_to_interval(s, min_level, max_level, steps)
        V, T = pygmi.utils.extract.marching_cubes(F_volume, voxel_size, t)
        V -= 1.0
        if s == 0:
            fig.add_trace(go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=T[:, 0], j=T[:, 1], k=T[:, 2]))
        frames.append(go.Frame(
            data=go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=T[:, 0], j=T[:, 1], k=T[:, 2]),
            name='{:1.3f}-lvl set'.format(label_to_interval(s, min_level, max_level, steps)), traces=[0])) 
    fig.update(frames=frames)

    def frame_args(duration: float) -> Dict:
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }
    sliders = [
        dict(
            pad={"b": 10, "t": 60},
            len=0.9,
            x=0.1,
            y=0,
            steps=[
                dict(
                    method='animate',
                    label='{:1.3f}-lvl set'.format(label_to_interval(k, min_level, max_level, steps)),
                    args=[[f.name], frame_args(0)]
                ) for k, f in enumerate(fig.frames)
            ], 
        )
    ]
    fig.update_layout(
        title='Level sets animation',
        width=600,
        height=600,
        scene=dict(
            zaxis=dict(range=[axes[4], axes[5]], autorange=False),
            yaxis=dict(range=[axes[2], axes[3]], autorange=False),
            xaxis=dict(range=[axes[0], axes[1]], autorange=False)
        ),
        updatemenus = [
            {
                "buttons": [
                    {
                        "args": [None, frame_args(50)],
                        "label": "&#9654;", # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;", # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
        ],
        sliders=sliders
    )
