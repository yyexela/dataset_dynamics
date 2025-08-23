###############################
# Imports # Imports # Imports #
###############################

import os
import torch
import seaborn as sns
import numpy as np
import pandas as pd
import pprint as pp
import matplotlib.pyplot as plt
from src import helpers
import src.global_config as global_config
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Load config
config = global_config.config

################################################################
# Plotting functions # Plotting functions # Plotting functions #
################################################################

def calculate_and_plot_pareto(data, angle:float = -45, plot:bool = False):
    """
    data: Data (using loss curve) to calculate pareto point of 
    angle: Angle of the line at which we're calculating the pareto optimal point
           in degrees
    plot: Make a plot of the plotted point

    Returns:
    Point of intersection used to generate pareto point
    """

    # Scale input data
    data_scaled, data_min, data_max = helpers.min_max_fit(data, 0., 1.)
    # Calculate slope of line
    slope = np.tan(angle/180*np.pi)
    # Create line based on slope
    line_scaled = np.linspace(0, slope, data.shape[0])

    # Bisection to find optimal intercept
    min_b = 0.
    max_b = 1.
    for _ in range(1000):
        test_b = (max_b+min_b)/2.
        test_line = line_scaled + test_b
        # check if under or above curve
        if np.any(test_line > data_scaled):
            # test line above data
            max_b = test_b
        else:
            # test line below data
            min_b = test_b
        # break if below tolerance
        if max_b - min_b < 1e-8:
            line_scaled = line_scaled + test_b
            break
    
    # Calculate point closest to line
    distances = np.abs(line_scaled-data_scaled)
    # Pareto optimal point and its value
    pareto_x = np.argmin(distances)
    pareto_loss = data[pareto_x]

    # Inverse line
    line = helpers.min_max_fit_inv(line_scaled, data_min, data_max, 0., 1.,)

    # Plot the data, line, and pareto point
    pareto = (pareto_x,pareto_loss)

    x = np.arange(data.shape[0]); x = [x, x];
    y = [data, line]
    label = ['loss', 'line']

    title = "Training Loss"
    xaxis_title = "Epoch"
    yaxis_title = "Loss"
    fname = "pareto"

    plot_pareto(x, y, pareto, label, title, xaxis_title, yaxis_title, fname = fname, save=True)

    # Return pareto
    return pareto

def plot_score_dict(sd):
    n_terms = list(sd.keys())
    all_threshs = list()
    all_scores = list()
    all_n = list()
    combined = list()

    for n in n_terms:
        n_threshs = [result[0] for result in sd[n]]
        n_scores = [result[2] for result in sd[n]]

        for result in sd[n]:
            combined.append((result[0], result[2], n))
            all_threshs.append(result[0])
            all_scores.append(result[2])
            all_n.append(n)

    df = pd.DataFrame(data = {"threshold": all_threshs, "score": all_scores, "n_terms": all_n})
    pp.pprint(df)

    fig = px.scatter(df, x="threshold", y="score", color="n_terms",
                    size='score')
    fig.show()

def plot_transition_frames(x:tuple[np.ndarray,list[np.ndarray]], y:tuple[np.ndarray,list[np.ndarray]], label:tuple[str,list[str]], title:str, threshold:float=None, xaxis_title:str = None, yaxis_title:str = None, fname:str = None, save:bool = False, width:int = 15000) -> None:
    """
    Plot one or multiple lines of transition frames

    If plotting multiple, len(x)=len(y)=len(labels)
    """

    lines = list()

    # Plot data
    lines.append(
        go.Scatter(x=x, y=y, name=label,
                mode = 'lines',
                marker = dict(
                    size = 5,
                    symbol='circle'
                )
        )
    )

    if threshold is not None:
        # Plot threshold
        lines.append(
            go.Scatter(x=x, y=np.ones_like(x)*threshold, name="Threshold",
                    mode = 'lines',
                    marker = dict(
                        size = 5,
                        symbol='circle'
                    )
            )
        )

    fig = go.Figure(data=lines)
    #fig.update_layout(title=title,
                    #width=width,
                    #xaxis = dict(
                        #tickmode = 'linear',
                        #tick0 = 0.00,
                        #dtick = 1.00)
                    #)
    #fig.update_xaxes(range=[np.min(x), np.max(x)])

    if xaxis_title is not None:
        fig.update_layout(xaxis_title=xaxis_title)
    if yaxis_title is not None:
        fig.update_layout(yaxis_title=yaxis_title)

    if save:
        if fname is None:
            raise Exception(f"Filename fname ({fname}) must not be None.")
        fig.write_image(os.path.join(config.pdf_dir, f'{fname}.pdf'))
    else:
        fig.show()

    return None

def plot_ensemble(x:tuple[np.ndarray,list[np.ndarray]], means:tuple[np.ndarray,list[np.ndarray]], stdevs:tuple[np.ndarray,list[np.ndarray]], title:str, xaxis_title:str = None, yaxis_title:str = None, fname:str = None, save:bool = False) -> None:
    """
    Plot one or multiple lines with a pareto point

    If plotting multiple, len(x)=len(y)
    """
    lines = list()

    lines.append(
        go.Scatter(
            x=x,
            y=means,
            error_y=dict(
                array=stdevs
            ),
            mode='markers',
            marker=dict(
                size=5,
                line=dict(
                    color='black',
                    width=1
                )
            )
        )
    )

    fig = go.Figure(data=lines)
    fig.update_layout(title=title)

    if xaxis_title is not None:
        fig.update_layout(xaxis_title=xaxis_title)
    if yaxis_title is not None:
        fig.update_layout(yaxis_title=yaxis_title)

    if save:
        if fname is None:
            raise Exception(f"Filename fname ({fname}) must not be None.")
        fig.write_image(os.path.join(config.pdf_dir, f'{fname}.pdf'))
    else:
        fig.show()

def plot_ensembles(x:list, means:dict, stdevs:dict, label:tuple[str,list[str]], title:str, color:tuple[str,list[str]] = 'blue', xaxis_title:str = None, yaxis_title:str = None, fname:str = None, save:bool = False) -> None:
    """
    Plot one or multiple sets of ensembles

    x: list of x-coordinates (model bottlenecks)
    y: dictionary containing number of layers as keys and means of bottlenecks as values
    """

    lines = list()
    layers = sorted(list(means.keys()))

    for i, layer in enumerate(layers):
        lines.append(
            go.Scatter(x=x, y=means[layer], name=label[i], 
                            error_y=dict(array=stdevs[layer]),
                            mode='markers',
                            marker=dict(
                                color=color[i],
                                size=5,
                                line=dict(
                                    color='black',
                                    width=1
                                )
                            ),
                            hoverlabel=dict(
                                namelength=-1
                            )
                       )
        )

    fig = go.Figure(data=lines)
    fig.update_layout(title=title)

    fig = go.Figure(data=lines)
    fig.update_layout(title=title)

    if xaxis_title is not None:
        fig.update_layout(xaxis_title=xaxis_title)
    if yaxis_title is not None:
        fig.update_layout(yaxis_title=yaxis_title)

    if save:
        if fname is None:
            raise Exception(f"Filename fname ({fname}) must not be None.")
        fig.write_image(os.path.join(config.pdf_dir, f'{fname}.pdf'))
    else:
        fig.show()

def plot_scatters(x:tuple[np.ndarray,list[np.ndarray]], y:tuple[np.ndarray,list[np.ndarray]], label:tuple[str,list[str]], title:str, color:tuple[str,list[str]] = 'blue', xaxis_title:str = None, yaxis_title:str = None, fname:str = None, save:bool = False) -> None:
    """
    Plot one or multiple lines with a pareto point

    If plotting multiple, len(x)=len(y)=len(labels)
    """
    if not isinstance(x, list):
        x = [x]
        y = [y]
        label = [label]
        color = [color]

    lines = list()

    for i in range(len(x)):
        lines.append(
            go.Scatter(x=x[i], y=y[i], name=label[i], 
                            mode='markers',
                            marker=dict(
                                color=color[i],
                                size=5,
                                line=dict(
                                    color='black',
                                    width=1
                                )
                            ),
                            hoverlabel=dict(
                                namelength=-1
                            )
                       )
        )

    fig = go.Figure(data=lines)
    fig.update_layout(title=title)

    fig = go.Figure(data=lines)
    fig.update_layout(title=title)

    if xaxis_title is not None:
        fig.update_layout(xaxis_title=xaxis_title)
    if yaxis_title is not None:
        fig.update_layout(yaxis_title=yaxis_title)

    if save:
        if fname is None:
            raise Exception(f"Filename fname ({fname}) must not be None.")
        fig.write_image(os.path.join(config.pdf_dir, f'{fname}.pdf'))
    else:
        fig.show()

def plot_pareto(x:tuple[np.ndarray,list[np.ndarray]], y:tuple[np.ndarray,list[np.ndarray]], pareto:tuple[float,float], label:tuple[str,list[str]], title:str, xaxis_title:str = None, yaxis_title:str = None, fname:str = None, save:bool = False) -> None:
    """
    Plot one or multiple lines with a pareto point

    If plotting multiple, len(x)=len(y)=len(labels)
    """
    if not isinstance(x, list):
        x = [x]
        y = [y]
        label = [label]

    lines = list()

    for i in range(len(x)):
        lines.append(
            go.Scatter(x=x[i], y=y[i], name=label[i])
        )

    lines.append(go.Scatter(x=[pareto[0]], y=[pareto[1]],
                            name="Pareto Optimal",
                            mode='markers',
                            marker=dict(
                                color='limegreen',
                                size=10,
                                line=dict(
                                    color='black',
                                    width=2
                                )
                            ))
    )

    fig = go.Figure(data=lines)
    fig.update_layout(title=title)

    fig = go.Figure(data=lines)
    fig.update_layout(title=title)

    if xaxis_title is not None:
        fig.update_layout(xaxis_title=xaxis_title)
    if yaxis_title is not None:
        fig.update_layout(yaxis_title=yaxis_title)

    if save:
        if fname is None:
            raise Exception(f"Filename fname ({fname}) must not be None.")
        fig.write_image(os.path.join(config.pdf_dir, f'{fname}.pdf'))
    else:
        fig.show()

def plot_line(x:tuple[np.ndarray,list[np.ndarray]], y:tuple[np.ndarray,list[np.ndarray]], label:tuple[str,list[str]], title:str, xaxis_title:str = None, yaxis_title:str = None, fname:str = None, save:bool = False) -> None:
    """
    Plot one or multiple lines

    If plotting multiple, len(x)=len(y)=len(labels)
    """

    if not isinstance(x, list):
        x = [x]
        y = [y]
        label = [label]

    lines = list()

    for i in range(len(x)):
        lines.append(
            go.Scatter(x=x[i], y=y[i], name=label[i])
        )

    fig = go.Figure(data=lines)
    fig.update_layout(title=title)

    if xaxis_title is not None:
        fig.update_layout(xaxis_title=xaxis_title)
    if yaxis_title is not None:
        fig.update_layout(yaxis_title=yaxis_title)

    if save:
        if fname is None:
            raise Exception(f"Filename fname ({fname}) must not be None.")
        fig.write_image(os.path.join(config.pdf_dir, f'{fname}.pdf'), width=400, height=300)
    else:
        fig.show()

    return None

def plot_means_and_stdev(x:list[np.ndarray], means:list[np.ndarray], stdevs:list[np.ndarray], colors:list[tuple[int,int,int]]) -> None:
    """
    Plots means and standard deviations over specified domain and color  
    `colors` is an RGB tuple (ie. (255,0,0))
    """

    fig = go.Figure()

    for i in range(len(x)):
        fig.add_trace(go.Scatter(x=x[i], y=means[i]-stdevs[i], mode='lines', line=dict(width=0.), name="Std", showlegend=False))
        fig.add_trace(go.Scatter(x=x[i], y=means[i]+stdevs[i], mode='lines', line=dict(width=0.), fill='tonexty', fillcolor=f'rgba({colors[i][0]},{colors[i][1]},{colors[i][2]},0.5)', name="Std"))
        fig.add_trace(go.Scatter(x=x[i], y=means[i], mode='lines', name='Mean', line_color=f'rgba({colors[i][0]},{colors[i][1]},{colors[i][2]},1)'))

    # Show the plot
    fig.show()

    return None

def plot_surface(z: tuple[np.ndarray,list[np.ndarray]], x:tuple[np.ndarray,list[np.ndarray]], y:tuple[np.ndarray,list[np.ndarray]], hovertemplate:str, colorscale:tuple[str,list[str]]=None, title:str=None, xaxis_title:str=None, yaxis_title:str=None, zaxis_title:str=None) -> None:
    """
    Plot one or multiple surfaces

    If plotting multiple, len(z)=len(x)=len(y)=len(colorscale)
    """

    if not isinstance(z, list):
        z = [z]
        x = [x]
        y = [y]
        colorscale = [colorscale]
        
    surfaces = list()
    for i in range(len(z)):
        surfaces.append(
            go.Surface(z=z[i], x=x[i], y=y[i],
                hovertemplate=hovertemplate,
                colorscale=colorscale[i])
        )
    fig = go.Figure(data=surfaces)

    fig.update_layout(title=title, autosize=False,
                        width=1000, height=500,
                        margin=dict(l=10, r=10, b=10, t=40),
                        scene = dict(
                            xaxis_title=xaxis_title,
                            yaxis_title=yaxis_title,
                            zaxis_title=zaxis_title,
                            camera_eye=dict(x=1,y=1,z=0.2),
                            camera=dict(
                                eye=dict(x=-1, y=-1, z=0.25)
                            ),
                            aspectratio=dict(x=1,y=1,z=.2)
                            #yaxis=dict(autorange="reversed")
                        ))

    fig.show()
    return None

def plot_wave_and_spectrum(x: np.ndarray, yox: np.ndarray, n: int, filename: str = "wave_and_spectrum.pdf", inline: bool = False) -> None:
    """
    Plot provided x and y values over an interval as well as a separate plot of the fourier transform.

    `x`: x-values  
    `yox`: y-values  
    `n`: Number of points used to generate data  
    `filename`: Filename of saved file if we're saving it  
    `inline`: True if we don't want to save the file and instead want to plot it in Jupyter  

    Returns: None
    """
    # Btw, "yox" --> "y of x"
    # Compute fft
    k, yok = helpers.fft(yox, n)
    # Plot
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9, 4))
    ax0.set_title("Function")
    ax0.plot(x, yox)
    ax0.set_xlabel("x")
    ax0.set_ylabel("f(x)")
    ax1.set_title("FT of Function")
    ax1.plot(k, yok)
    ax1.set_xlabel("k")
    ax1.set_ylabel("f(k)")
    if inline:
        plt.show()
    else:
        plt.savefig(os.path.join(config.pdf_dir, filename))
    plt.close()

    return None

def plot_spectral_dynamics(all_frames, freqs: list[float], amps: list[float], n: int, filename:str='spectral_dynamics.pdf', inline:bool=False) -> None:
    """
    Given model predictions at various iterations during training, plot their prediction frequencies. x-axis shows frequency and y-axis shows iteration. We only plot the frequencies that are present in the dataset we're fitting to.

    `all_frames`: list of size `config.repeats` by `config.iters/config.save_freq`
    `freqs`: Input frequencies for what we're fitting models to
    `amps`: Input amplitudes for what we're fitting models to
    `n`: Number of points for each generated function we're fitting models to
    `filename`: Filename of saved file if we're saving it  
    `inline`: True if we don't want to save the file and instead want to plot it in Jupyter  

    Returns: `None`
    """
    all_dynamics = []
    # Compute spectra for all frames
    for frames in all_frames: 
        frq, dynamics, xticks = helpers.compute_spectra(frames, n)
        all_dynamics.append(dynamics)
    # Average dynamics over multiple frames
    # mean_dynamics.shape = (num_iterations, num_frequencies)
    mean_dynamics = np.array(all_dynamics).mean(0)
    # Select the frequencies which are present in the target spectrum
    freq_selected = mean_dynamics[:, np.sum(frq.reshape(-1, 1) == np.array(freqs).reshape(1, -1), 
                                            axis=-1, dtype='bool')]
    # Normalize by the amplitude. Remember to account for the fact that the measured spectra 
    # are single-sided (positive freqs), so multiply by 2 accordingly
    norm_dynamics = 2 * freq_selected / np.array(amps).reshape(1, -1)
    # Plot heatmap
    plt.figure(figsize=(7, 6))
    # plt.title("Evolution of Frequency Spectrum (Increasing Amplitudes)")
    sns.heatmap(norm_dynamics[::-1], 
                xticklabels=freqs, 
                yticklabels=[(frame.iter_num if frame.iter_num % 10000 == 0 else '') 
                             for _, frame in zip(range(norm_dynamics.shape[0]), frames)][::-1], 
                vmin=0., vmax=1., 
                cmap=sns.cubehelix_palette(8, start=.5, rot=-.75, reverse=True, as_cmap=True))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Training Iteration")
    if inline:
        plt.show()
    else:
        plt.savefig(os.path.join(config.pdf_dir, filename))
    plt.close()
    #plt.show()

    return None

def plot_multiple_spectral_norms(all_frames, filename:str='multiple_spectral_norms.pdf', inline:bool=False) -> None:
    """
    Plots the mean and standard deviations of spectral norms of each model layer. The mean and standard deviation is computed across repeats.

    `all_frames`: Saved data from training we're using to make the plots.  
    `filename`: Filename of saved file if we're saving it  
    `inline`: True if we don't want to save the file and instead want to plot it in Jupyter  

    Returns: None
    """
    iter_nums = np.array([frame.iter_num for frame in all_frames[0]])
    norms = np.array([np.array(list(zip(*[frame.spectral_norms for frame in frames]))).squeeze() for frames in all_frames])
    means = norms.mean(0)
    stds = norms.std(0)
    plt.xlabel("Training Iteration")
    plt.ylabel("Spectral Norm of Layer Weights")
    for layer_num, (mean_curve, std_curve) in enumerate(zip(means, stds)): 
        p = plt.plot(iter_nums, mean_curve, label=f'Layer {layer_num + 1}')
        plt.fill_between(iter_nums, mean_curve + std_curve, mean_curve - std_curve, color=p[0].get_color(), alpha=0.15)
    plt.legend()
    if inline:
        plt.show()
    else:
        plt.savefig(os.path.join(config.pdf_dir, filename))
    plt.close()
    #plt.show()

    return None

def plot_MNIST_AE_sigmas_zoom(MNIST_sigmas: np.ndarray, frames, plot_iters: list[int], filename:str='MNIST_AE_sigmas_zoom.pdf', inline:bool=False) -> None:
    """
    Given MNIST sigma values from SVD, make a simple plot of them, zooming into bottom left corner

    `MNIST_sigmas`: True singular values of MNIST dataset  
    `frames`: Frames from training containing predicted MNIST dataset  
    `filename`: Filename of saved file if we're saving it  
    `inline`: True if we don't want to save the file and instead want to plot it in Jupyter  

    Returns: `None` 
    """
    # Plot MNIST sigmas
    plt.figure()
    plt.title(f"MNIST Singular Values")
    plt.plot(MNIST_sigmas, label="MNIST $\sigma_i$")

    # Plot AE sigmas
    for iter in plot_iters:
        svd = torch.svd(torch.from_numpy(frames[iter].prediction))
        plt.plot(svd[1], label=f"AE (iter {iter+1}) $\sigma_i$")

    # Adjust axes to zoom
    plt.ylim(int((min(MNIST_sigmas)-1)*0.1), int((max(MNIST_sigmas)+1)*0.1))
    plt.xlim(-1, len(MNIST_sigmas)+1)

    # Add legend
    plt.legend()

    if inline:
        plt.show()
    else:
        plt.savefig(os.path.join(config.pdf_dir, filename))

    plt.close()

    return None

def plot_MNIST_AE_sigmas_full(MNIST_sigmas: np.ndarray, frames, plot_iters: list[int], filename:str='MNIST_AE_sigmas_full.pdf', inline:bool=False):
    """
    Given MNIST sigma values from SVD, make a simple plot of them

    `MNIST_sigmas`: True singular values of MNIST dataset  
    `frames`: Frames from training containing predicted MNIST dataset  
    `filename`: Filename of saved file if we're saving it  
    `inline`: True if we don't want to save the file and instead want to plot it in Jupyter  

    Returns: `None` 
    """
    # Plot MNIST sigmas
    plt.figure()
    plt.title(f"MNIST Singular Values")
    plt.plot(MNIST_sigmas, label="MNIST $\sigma_i$")

    # Plot AE sigmas
    for iter in plot_iters:
        svd = torch.svd(torch.from_numpy(frames[iter].prediction))
        plt.plot(svd[1], label=f"AE (iter {iter+1}) $\sigma_i$")

    # Adjust axes to fit MNIST sigmas
    plt.ylim(min(MNIST_sigmas)-1, max(MNIST_sigmas)+1)
    plt.xlim(-1, len(MNIST_sigmas)+1)

    # Add legend
    plt.legend()

    if inline:
        plt.show()
    else:
        plt.savefig(os.path.join(config.pdf_dir, filename))

    plt.close()

    return None

def plot_MNIST_sigmas(sigmas: np.ndarray, filename:str="MNIST_sigmas.pdf", inline:bool=False) -> None:
    """
    Given MNIST sigma values from SVD, make a simple plot of them

    `MNIST_sigmas`: True singular values of MNIST dataset  
    `filename`: Filename of saved file if we're saving it  
    `inline`: True if we don't want to save the file and instead want to plot it in Jupyter  

    Returns: `None` 
    """
    plt.figure()
    plt.title(f"MNIST Singular Values")
    plt.plot(sigmas)
    plt.ylim(min(sigmas)-1, max(sigmas)+1)
    plt.xlim(-1, len(sigmas)+1)
    if inline:
        plt.show()
    else:
        plt.savefig(os.path.join(config.pdf_dir, filename))
    plt.close()

    return None

def plot_2D(xx: np.ndarray, yy: np.ndarray, zz: np.ndarray, title:str = "", filename:str = "2D_waves.pdf", inline:bool=False, bar:bool=False):
    """
    Plot a sample of the MNIST dataset

    `xx`: 2D numpy array of x-values  
    `yy`: 2D numpy array of y-values  
    `zz`: 2D numpy array of z-values  
    `title`: Title for the plot  
    `filename`: Filename of saved file if we're saving it  
    `inline`: True if we don't want to save the file and instead want to plot it in Jupyter  
    `bar`: Whether or not to plot colorbar

    Returns: `None`
    """
    plt.figure()
    plt.title(f"2D Waves")
    plt.imshow(zz,\
        extent=[xx[0,0], xx[0,-1], yy[-1,0], yy[0,0]])
    plt.grid(False)
    plt.title(title)
    if bar: plt.colorbar()
    if inline:
        plt.show()
    else:
        plt.savefig(os.path.join(config.pdf_dir, filename))
    plt.close()

    return None

def plot_2D_ffts(lxx: list[np.ndarray], titles: list[str], plot_log:bool=False, integral_data = None) -> None:
    """
    Calculate then plot the 2D FFT of the list of inputs in `lx`

    `lxx`: List of 2D data FFT data  
    `titles`: Names of data in lxx
    `plot_log`: Plot log(|FFT|+1.) instead for axes  
    `integral_data`: Pre-computed integral data
    `filename`: Filename of saved file if we're saving it  

    Return: `None`
    """

    rows = 1 + int(integral_data is not None)
    cols = len(lxx)

    # Set up titles
    postfix = "log" if plot_log else "standard"
    title = ""
    filename = "2DFFT_{j:004}_{postfix}.pdf"

    # Iterate over FFT indices
    for im_idx in range(lxx[0].shape[0]):
        fig, axs = plt.subplots(rows, cols, figsize=(int(4*cols), int(4*rows)))
        if int(rows+cols) == 2:  
            axs = [axs]

        # Per column
        for col_idx in range(cols):
            # Plot FFT
            ax = axs[0] if int(cols) == 1 else axs[0][col_idx]
            lxx[col_idx][im_idx] = np.log(1.+lxx[col_idx][im_idx]) if plot_log else lxx[col_idx][im_idx]
            ax.imshow(lxx[col_idx][im_idx], cmap='viridis')
            ax.set_title(f"{titles[col_idx]} 2DFFT")
            #ax.axis('off')  # Turn off axis ticks and labels

            if integral_data is not None:
                # Calculate and plot integral
                ax = axs[1] if int(cols) == 1 else axs[1][col_idx]

                rx_vals, _, int_vals = (integral_data['r_x'], integral_data['r_y'], integral_data['ints'])

                ax.plot(rx_vals, int_vals[im_idx])
                ax.set_xlabel("Radius (x-value)")
                ax.set_ylabel("Average value")
                ax.set_title(f"p={integral_data['p']:0.3f} s={integral_data['smoothing']:.3f} integrals")

        fig.suptitle(title)

        filename_filled = filename.format(postfix = postfix,j = im_idx)

        plt.tight_layout()
        plt.grid(False)
        plt.savefig(os.path.join(config.pdf_mass_dir, filename_filled))
        plt.close()

def plot_1D_fft(lx: list[np.ndarray], labels: list[str], title="1D FFT", fft_pos_freq: bool = True, fft_normalize: bool = True, fft_centering: bool = False, plot_log:bool=False, plot_style:str='.', plot_lims:list[float, float]=None, filename="Multiple1DFFT.pdf", inline=False) -> None:
    """
    Calculate then plot the 1D FFT of the list of inputs in `lx`

    `lx`: List of 1D data to plot the FFTs of  
    `labels`: Labels for each `x` to add to the legend  
    `title`: Plot title  
    `plot_log`: Plot log(|FFT|+1.) instead  
    `plot_style`: Style for FFT plot  
    `plot_lims`: x-range of plot  
    `fft_pos_freq`: Output positive frequencies  
    `fft_normalize`: Normalize output  
    `fft_centering`: Input signal is centered along axes  
    `filename`: Filename of saved file if we're saving it  
    `inline`: True if we don't want to save the file and instead want to plot it in Jupyter  

    Return: `None`
    """
    for x, label in zip(lx, labels):
        fft, freq = helpers.fft1d(x, pos_freq=fft_pos_freq, normalize=fft_normalize, centering=fft_centering)
        if plot_log:
            fft = np.log(fft+1.)
        plt.plot(freq, fft, plot_style, label=label)

    plt.xlabel("freq")
    plt.ylabel("amp")
    plt.xlim(plot_lims)
    plt.title(title)
    plt.legend()

    if inline:
        plt.show()
    else:
        plt.savefig(os.path.join(config.pdf_dir, filename))
    plt.close()

def plot_several_image_samples(sample_sets: list[np.ndarray], titles: list[str], original_shape = tuple[int, int]):
    """
    Plot two image samples on a 1x2 subplot.

    `sample_sets`: List of eigenimage sets  
    `titles`: Title of each image to plot  
    `filename`: Filename of saved file if we're saving it  
    `inline`: True if we don't want to save the file and instead want to plot it in Jupyter  

    Returns: `None`
    """

    for im_idx in range(sample_sets[0].shape[0]):
        samples = [sample_set[:,im_idx] for sample_set in sample_sets]
        filename = f"multiple_{im_idx:004}.pdf"
        if len(samples) >= 2:
            fig, axs = plt.subplots(1, len(samples), figsize=(5*len(samples), 5))

            for i, (image, ax) in enumerate(zip(samples, axs.flat)):
                ax.imshow(image.view(original_shape[0], original_shape[1]).numpy(), cmap='viridis')
                if titles is not None:
                    ax.set_title(f"{titles[i]} SVD $V_{{{im_idx}}}")
                ax.axis('off')  # Turn off axis ticks and labels
        else:
            fig, axs = plt.subplots(1, 1, figsize=(5,5))
            axs.imshow(samples[0].view(original_shape[0], original_shape[1]).numpy(), cmap='viridis')
            if titles is not None:
                axs.set_title(f"{titles[0]} SVD $V_{{{im_idx}}}")
            axs.axis('off')  # Turn off axis ticks and labels

        plt.suptitle(filename)
        plt.tight_layout()
        plt.grid(False)
        plt.savefig(os.path.join(config.pdf_mass_dir, filename))
        plt.close()

    return None

def plot_image_sample(sample, title="MNIST Image", shape: torch.Size = None, filename:str = "MNIST_example.pdf", inline:bool=False):
    """
    Plot a sample of an image from a dataset

    `sample`: Specific MNIST image to plot
    `title`: Title for plot  
    `shape`: Shape of original image  
    `filename`: Filename of saved file if we're saving it  
    `inline`: True if we don't want to save the file and instead want to plot it in Jupyter  

    Returns: `None`
    """
    plt.figure()
    plt.title(title)
    plt.imshow(sample.view(shape[0], shape[1]).numpy())
    plt.grid(False)
    if inline:
        plt.show()
    else:
        plt.savefig(os.path.join(config.pdf_dir, filename))
    plt.close()

    return None

def plot_MNIST_noisy_loss(frames, filename:str='MNIST_noisy_loss.pdf', inline:bool=False): 
    """
    Plots two curves showing training and validation loss of MNIST noisy data.

    `frames`: Training frames  
    `filename`: Filename of saved file if we're saving it  
    `inline`: True if we don't want to save the file and instead want to plot it in Jupyter  
    """
    its, val_loss, tr_loss = zip(*[(frame.iter_num, frame.val_loss, frame.loss) for frame in frames])
    plt.figure()
    plt.semilogy(its, tr_loss, label='Training')
    plt.semilogy(its, val_loss, label='Validation')
    plt.xlabel("Training Iteration")
    plt.ylabel("MSE Loss")
    plt.legend()
    if inline:
        plt.show()
    else:
        plt.savefig(os.path.join(config.pdf_dir, filename))
    plt.close()

    return None
    
def compare_loss(frames0, frames1, filename:str='compare_loss.pdf', inline:bool=False): 
    """
    TODO

    `frames0`: TODO  
    `frames1`: TODO  
    `filename`: Filename of saved file if we're saving it  
    `inline`: True if we don't want to save the file and instead want to plot it in Jupyter  
    """
    its, val_loss0 = zip(*[(frame.iter_num, frame.val_loss) for frame in frames0])
    its, val_loss1 = zip(*[(frame.iter_num, frame.val_loss) for frame in frames1])
    plt.figure()
    plt.semilogy(its, val_loss0, label='Set 0')
    plt.semilogy(its, val_loss1, label='Set 1')
    plt.xlabel("Training Iteration")
    plt.ylabel("MSE Loss")
    plt.legend()
    if inline:
        plt.show()
    else:
        plt.savefig(os.path.join(config.pdf_dir, filename))

    plt.close()