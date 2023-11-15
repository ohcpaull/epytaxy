import numpy as np
import scipy
from matplotlib.patches import ConnectionPatch
from skimage.measure import profile_line
import lmfit
import matplotlib.pyplot as plt
import matplotlib as mpl

def _line_profile_coordinates(src, dst, linewidth=1):
    """
    Return the coordinates of the profile of an image along a scan line.

    Parameters
    ----------
    src : 2-tuple of numeric scalar (float or int)
        The start point of the scan line.
    dst : 2-tuple of numeric scalar (float or int)
        The end point of the scan line.
    linewidth : int, optional
        Width of the scan, perpendicular to the line

    Returns
    -------
    coords : array, shape (2, N, C), float
        The coordinates of the profile along the scan line. The length of the
        profile is the ceil of the computed length of the scan line.

    Notes
    -----
    This is a utility method meant to be used internally by skimage functions.
    The destination point is included in the profile, in contrast to
    standard numpy indexing.
    """
    src_row, src_col = src = np.asarray(src, dtype=float)
    dst_row, dst_col = dst = np.asarray(dst, dtype=float)
    d_row, d_col = dst - src
    theta = np.arctan2(d_row, d_col)

    length = int(np.ceil(np.hypot(d_row, d_col) + 1))
    # we add one above because we include the last point in the profile
    # (in contrast to standard numpy indexing)
    line_col = np.linspace(src_col, dst_col, length)
    line_row = np.linspace(src_row, dst_row, length)

    # we subtract 1 from linewidth to change from pixel-counting
    # (make this line 3 pixels wide) to point distances (the
    # distance between pixel centers)
    col_width = (linewidth - 1) * np.sin(-theta) / 2
    row_width = (linewidth - 1) * np.cos(theta) / 2
    perp_rows = np.stack([np.linspace(row_i - row_width, row_i + row_width,
                                    linewidth) for row_i in line_row])
    perp_cols = np.stack([np.linspace(col_i - col_width, col_i + col_width,
                                    linewidth) for col_i in line_col])
    return np.stack([perp_rows, perp_cols])


class LineProfile:
    """
    Class for taking an arbitrary line profile of a 2D dataset. This can be for
    the case of a scanning probe line profile, or a X-ray diffraction reciprocal 
    space map along an arbitrary or crystallographic direction.

    Parameters
    ----------


    Attributes
    ----------

        - p_i       :   tuple, (initial x pixel, initial y pixel)
            Initial pixel points for line profile
        - p_f       :   tuple, (final x pixel, final y pixel)
            Final pixel points for the line profile
        - px_width  :   int
            Width in pixels for the line profile to integrate over. 
        - px_dist   :   int
            Length of the line profile in pixels
        
    """

    def __init__(self, px_i, px_f, width, color="black", linewidth=1):
        self.px_i = px_i
        self.px_f = px_f        
        self.px_width = width

        # Find distance in pixels of line profile
        self.px_dist = int(np.round(np.hypot(self.px_f[0] - self.px_i[0], self.px_f[1] - self.px_i[1])))
        self.s_dist = None
        # Calculate the angle the line makes with the x-axis
        self.line_vec = np.array([self.px_f[0] - self.px_i[0], self.px_f[1] - self.px_i[1]])

        self.angle = np.angle(self.line_vec[0] + self.line_vec[1]*1j, deg=False) 

        # Calculate the offset in X and Y for the linewidth start
        # and end points at the start point for the line profile
        self.xyA_i = (
            (self.px_i[0] - width/2 * np.sin(self.angle)),
            (self.px_i[1] + width/2 * np.cos(self.angle)),
        )
        self.xyB_i = (
            (self.px_i[0] + width/2 * np.sin(self.angle)),
            (self.px_i[1] - width/2 * np.cos(self.angle)),
        )

        self.cpatch_i = ConnectionPatch(
            xyA=self.xyA_i,
            xyB=self.xyB_i,
            coordsA="data",
            coordsB="data",
            color=color,
            lw=linewidth
        )
        
        self.cpatch_line = ConnectionPatch(
            xyA=self.px_i,
            xyB=self.px_f,
            coordsA="data",
            coordsB="data",
            color=color,
            lw=linewidth
        )

        self.xyA_f = (
            (self.px_f[0] - width/2 * np.sin(self.angle)),
            (self.px_f[1] + width/2 * np.cos(self.angle)),
        )
        self.xyB_f = (
            (self.px_f[0] + width/2 * np.sin(self.angle)),
            (self.px_f[1] - width/2 * np.cos(self.angle)),
        )

        self.cpatch_f = ConnectionPatch(
            xyA=self.xyA_f,
            xyB=self.xyB_f,
            coordsA="data",
            coordsB="data",
            color=color,
            lw=linewidth
        )

    def __len__(self):
        return print(f"Length = {self.px_dist} pixels\nWidth = {self.px_width} pixels")

    def cut_channel(self, channel_data):

        self.line_profile = profile_line(
            channel_data.T,
            self.px_i,
            self.px_f,
            linewidth = self.px_width,
            mode="nearest"
        )

    def _plot_over_channel(self, axis, color="black"):
        
        y_min, y_max = axis.get_ylim()
        yrange = y_max - y_min

        x_min, x_max = axis.get_xlim()
        xrange = x_max - x_min
        
        for (x, y) in [self.xyA_i, self.xyB_i, self.xyA_f, self.xyB_f, self.px_i, self.px_f]:
            if x > xrange:
                raise RuntimeError("Coordinates of line slice are outside the"\
                    "coordinates of the axis."
                )
            elif y > yrange:
                raise RuntimeError("Coordinates of line slice are outside the"\
                    "coordinates of the axis."
                )

        axis.add_artist(self.cpatch_i)
        axis.add_artist(self.cpatch_line)
        axis.add_artist(self.cpatch_f)
        return axis

    def plot_over_channel(self, axis, **kwargs):
        
        y_min, y_max = axis.get_ylim()
        yrange = y_max - y_min

        x_min, x_max = axis.get_xlim()
        xrange = x_max - x_min
        
        for (x, y) in [self.xyA_i, self.xyB_i, self.xyA_f, self.xyB_f, self.px_i, self.px_f]:
            if x > xrange:
                raise RuntimeError("Coordinates of line slice are outside the"\
                    "coordinates of the axis."
                )
            elif y > yrange:
                raise RuntimeError("Coordinates of line slice are outside the"\
                    "coordinates of the axis."
                )
        cp_i = ConnectionPatch(
            xyA=self.xyA_i,
            xyB=self.xyB_i,
            coordsA="data",
            **kwargs
        )
        cp_f = ConnectionPatch(
            xyA=self.xyA_f,
            xyB=self.xyB_f,
            coordsA="data",
            **kwargs
        )
        cp_line = ConnectionPatch(
            xyA=self.px_i,
            xyB=self.px_f,
            coordsA="data",
            **kwargs
        )
        axis.add_artist(cp_i)
        axis.add_artist(cp_line)
        axis.add_artist(cp_f)
        return axis

def line_profile(data, scan_size, width=50.0, linewidth=1, **kwargs):
    """
    Plots a data channel in an interactive jupyter notebook figure.

    Interactive clicking on the data channel plot can create a line profile, 
    which is then plotted in the adjacent window. Clicking on channel 2D plot 
    creates a a startpoint, and a second click creates the end-point for 
    the line profile.

    Parameters
    ----------
    channel     :   str or np.ndarray
        Channel to plot and obtain line profile
    width       :   float
        Width in nanometres to integrate the linescan over. If you think this
        unit of width is silly then come fight me 

    """

    if "cmap" not in kwargs.keys():
        kwargs.update(cmap = "afmhot")

    if "color" in kwargs.keys():
        color = kwargs.pop("color")
    else:
        color = "black"

    fig, ax = plt.subplots(2,1)

    im, cbar = plot_map(ax[0], data,  x_vec = scan_size, y_vec = scan_size, cbar_label="nm", **kwargs)
    ax[0].set(xlabel="X (µm)", ylabel="Y (µm)") 
    pos = []
    line = []
    px, py = [], []
    xyA, xyB = (), ()
    
    axis_length = len(data)
    # Convert width between nanometres and pixels
    width = int(np.round(width / 1000 / scan_size * axis_length))


    def onclick(event):
        if len(pos) == 0:
            # plot first scatter
            scatter = ax[0].scatter(event.xdata, event.ydata, s=5, color=color)
            pos.append(scatter)
            px.append(event.xdata)
            py.append(event.ydata)

        elif len(pos) == 1:
            # plot second scatter and line
            scatter = ax[0].scatter(event.xdata, event.ydata, s=5, color=color)
            pos.append(scatter)
            px.append(event.xdata)
            py.append(event.ydata)
            x_values = [px[0], px[1]]
            y_values = [py[0], py[1]]

            # Plot line profile of data
            lp = LineProfile(
                (x_values[0], y_values[0]),
                (x_values[1], y_values[1]),
                width=width,
                color=color,
                linewidth=linewidth
            )


            lp.cut_channel(data)
            line.append(lp.cpatch_line)
            line.append(lp.cpatch_i)
            line.append(lp.cpatch_f)

            


            diff_x = (lp.px_f[0] - lp.px_i[0]) / axis_length * scan_size
            diff_y = (lp.px_f[1] - lp.px_i[1]) / axis_length * scan_size
            sample_distance = np.hypot(diff_x, diff_y)

            lp.s_dist = np.linspace(0, sample_distance, len(lp.line_profile))

            # Plot line profile in adjacent subplot
            ax[1].plot(
                lp.s_dist,
                lp.line_profile,
                label=f"line profile",
                color="k"
            )
            ax[1].set(xlabel="Distance (µm)", ylabel=f"Height (nm)", title="Line Profile")

            # Plot line and width 
            lp._plot_over_channel(ax[0])
            
            plt.savefig("LNO2_2A_LP.png", dpi=300)


        else:
        # clear variables 
            for scatter in pos:
                scatter.remove()

            px.clear()
            py.clear()
            pos.clear()
            ax[1].clear()
            line[0].remove()
            line[1].remove()
            line[2].remove()
            line.clear()



        fig.canvas.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()


def plot_map(axis, img, show_xy_ticks=True, show_cbar=True, x_vec=None, y_vec=None,
             num_ticks=4, stdevs=None, cbar_label=None, tick_font_size=None, infer_aspect=False, **kwargs):
    """
    Plots an image within the given axis with a color bar + label and appropriate X, Y tick labels.
    This is particularly useful to get readily interpretable plots for papers

    Parameters
    ----------
    axis : matplotlib.axes.Axes object
        Axis to plot this image onto
    img : 2D numpy array with real values
        Data for the image plot
    show_xy_ticks : bool, Optional, default = None, shown unedited
        Whether or not to show X, Y ticks
    show_cbar : bool, optional, default = True
        Whether or not to show the colorbar
    x_vec : 1-D array-like or Number, optional
        if an array-like is provided, these will be used for the tick values on the X axis
        if a Number is provided, this will serve as an extent for tick values in the X axis.
        For example x_vec=1.5 would cause the x tick labels to range from 0 to 1.5
    y_vec : 1-D array-like or Number, optional
        if an array-like is provided - these will be used for the tick values on the Y axis
        if a Number is provided, this will serve as an extent for tick values in the Y axis.
        For example y_vec=225 would cause the y tick labels to range from 0 to 225
    num_ticks : unsigned int, optional, default = 4
        Number of tick marks on the X and Y axes
    stdevs : unsigned int (Optional. Default = None)
        Number of standard deviations to consider for plotting.  If None, full range is plotted.
    cbar_label : str, optional, default = None
        Labels for the colorbar. Use this for something like quantity (units)
    tick_font_size : unsigned int, optional, default = None
        Font size to apply to x, y, colorbar ticks and colorbar label
    infer_aspect : bool, Optional. Default = False
        Whether or not to adjust the aspect ratio of the image based on the provided x_vec and y_vec
        The values of x_vec and y_vec will be assumed to have the same units.
    kwargs : dictionary
        Anything else that will be passed on to matplotlib.pyplot.imshow

    Returns
    -------
    im_handle : handle to image plot
        handle to image plot
    cbar : handle to color bar
        handle to color bar

    Note
    ----
    The origin of the image will be set to the lower left corner. Use the kwarg 'origin' to change this

    """
    if not isinstance(axis, mpl.axes.Axes):
        raise TypeError('axis must be a matplotlib.axes.Axes object')
    if not isinstance(img, np.ndarray):
        raise TypeError('img should be a numpy array')
    if not img.ndim == 2:
        raise ValueError('img should be a 2D array')
    if not isinstance(show_xy_ticks, bool):
        raise TypeError('show_xy_ticks should be a boolean value')
    if not isinstance(show_cbar, bool):
        raise TypeError('show_cbar should be a boolean value')
    # checks for x_vec and y_vec are done below
    if num_ticks is not None:
        if not isinstance(num_ticks, int):
            raise TypeError('num_ticks should be a whole number')
        if num_ticks < 2:
            raise ValueError('num_ticks should be at least 2')
    if stdevs is not None:
        data_mean = np.mean(img)
        data_std = np.std(img)
        kwargs.update({'clim': [data_mean - stdevs * data_std, data_mean + stdevs * data_std]})

    kwargs.update({'origin': kwargs.pop('origin', 'lower')})

    if show_cbar:
        vector = np.squeeze(img)
        if not isinstance(vector, np.ndarray):
            raise TypeError('vector should be of type numpy.ndarray. Provided object of type: {}'.format(type(vector)))
        if np.max(np.abs(vector)) == np.max(vector):
            exponent = np.log10(np.max(vector))
        else:
            # negative values
            exponent = np.log10(np.max(np.abs(vector)))
        
        y_exp = int(np.floor(exponent))
        z_suffix = ''
        if y_exp < -2 or y_exp > 3:
            img = np.squeeze(img) / 10 ** y_exp
            z_suffix = ' x $10^{' + str(y_exp) + '}$'

    assert isinstance(show_xy_ticks, bool)

    ########################################################################################################

    def set_ticks_for_axis(tick_vals, is_x):
        if is_x:
            tick_vals_var_name = 'x_vec'
            tick_set_func = axis.set_xticks
            tick_labs_set_func = axis.set_xticklabels
        else:
            tick_vals_var_name = 'y_vec'
            tick_set_func = axis.set_yticks
            tick_labs_set_func = axis.set_yticklabels

        img_axis = int(is_x)
        img_size = img.shape[img_axis]
        chosen_ticks = np.linspace(0, img_size - 1, num_ticks, dtype=int)

        if tick_vals is not None:
            if isinstance(tick_vals, (int, float)):
                if tick_vals > 0.01:
                    tick_labs = [str(np.round(ind * tick_vals / img_size, 2)) for ind in chosen_ticks]
                else:
                    tick_labs = ['{0:.1E}'.format(ind * tick_vals / img_size) for ind in chosen_ticks]
                    print(tick_labs)
                tick_vals = np.linspace(0, tick_vals, img_size)
            else:
                if not isinstance(tick_vals, (np.ndarray, list, tuple, range)) or len(tick_vals) != img_size:
                    raise ValueError(
                        '{} should be array-like with shape equal to axis {} of img'.format(tick_vals_var_name,
                                                                                            img_axis))
                if np.max(tick_vals) > 0.01:
                    tick_labs = [str(np.round(tick_vals[ind], 2)) for ind in chosen_ticks]
                else:
                    tick_labs = ['{0:.1E}'.format(tick_vals[ind]) for ind in chosen_ticks]
        else:
            tick_labs = [str(ind) for ind in chosen_ticks]

        tick_set_func(chosen_ticks)
        tick_labs_set_func(tick_labs)

        if tick_font_size is not None:
            for tick in axis.xaxis.get_major_ticks():
                tick.label.set_fontsize(tick_font_size)
            for tick in axis.yaxis.get_major_ticks():
                tick.label.set_fontsize(tick_font_size)    

        return tick_vals

    ########################################################################################################

    if show_xy_ticks is True or x_vec is not None:
        x_vec = set_ticks_for_axis(x_vec, True)
    else:
        axis.set_xticks([])

    if show_xy_ticks is True or y_vec is not None:
        y_vec = set_ticks_for_axis(y_vec, False)
    else:
        axis.set_yticks([])

    if infer_aspect:
        # Aspect ratio determined by this function will take precedence.
        _ = kwargs.pop('infer_aspect', None)

        """
        At this stage, if x_vec and y_vec are not None, they should be arrays.
        
        This will be very useful when one dimension is coarsely sampled while another is finely sampled
        and we want to visualize the image with the physically correct aspect ratio.
        This CANNOT be performed automatically due to potentially incompatible units which are unknown to this func.
        """

        if x_vec is not None or y_vec is not None:
            x_range = x_vec.max() - x_vec.min()
            y_range = y_vec.max() - y_vec.min()
            kwargs.update({'aspect': (y_range / x_range) * (img.shape[1] / img.shape[0])})

    im_handle = axis.imshow(img, **kwargs)

    cbar = None
    if not isinstance(show_cbar, bool):
        show_cbar = False

    if show_cbar:
        cbar = plt.colorbar(im_handle, ax=axis, orientation='vertical',
                            fraction=0.046, pad=0.04, use_gridspec=True)
        # cbar = axis.cbar_axes[count].colorbar(im_handle)

        if cbar_label is not None:
            if not isinstance(cbar_label, str):
                raise TypeError('cbar_label should be a string')

            if tick_font_size is not None:
                cbar.set_label(cbar_label + z_suffix)
            else:
                cbar.set_label(cbar_label + z_suffix, fontsize=tick_font_size)
        else:
            if z_suffix != '':
                cbar.set_label(z_suffix)

        if tick_font_size is not None:
            cbar.ax.tick_params(labelsize=tick_font_size)
    return im_handle, cbar        

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    
    if np.linalg.norm(v1) == 0 or  np.linalg.norm(v2) == 0:
        return 0
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def get_2DFFT(image):
    '''
    Calculates the 2D fourier transform of a bitmap image

    Parameters
    ----------
    image       :   np.array of shape (M,N)
        M rows in image, and N columns
    
    Returns
    ----------
    fft_image   :   Fourier transform of input image
    '''
    image_raw = image.get_n_dim_form().squeeze()
    fft_image = np.fft.fft2(image_raw)

    fft_image = np.fft.fftshift(fft_image)
    
    return fft_image

# Curve fitting functions

def gaussian2d(M, *args):
    """
    Multiple 2-dimensional gaussian generator to be plotted as a contour or
    mesh plot. Just put in the sets of starting parameters for each 2D gaussian 

    Parameters
    ----------
    M       :   np.array with shape (2,N)
        x and y data for the 2D plot.
    args    :   2D gaussian parameters
        x0  -   x centre for gaussian
        y0  -   y centre for gaussian
        xalpha - standard deviation in x
        yalpha - standard deviation in y
        A - amplitude
        offs - offset in Z

    Returns
    -------
    arr :   np.array (X,Y)
        X is width of 2D map
        Y is height of 2D map

    Notes
    ---------
    Must be a multiple of 6 arguments for the 2D gaussian otherwise it will not work. 
    """
    
    def _gaussian2d(x, y, x0, y0, xalpha, yalpha, A, offs):
        return A * np.exp(
        -((x - x0) / xalpha) ** 2 - ((y - y0) / yalpha)**2
        ) + offs

    x, y = M
    arr = np.zeros(x.shape)
    for i in range(len(args)//6):
       arr += _gaussian2d(x, y, *args[i*6:i*6+6])
    return arr

def Gauss2d(M, **params):
    """
    function to calculate any number of general two dimensional Gaussians. 
    Requires the x and y axes to be concatenated into a tuple of arrays, and
    that the number of parameters be divisible by the number of parameters
    for a 2D gaussian (i.e. 7) 

    Parameters
    ----------
    x, y :  array-like
        coordinate(s) where the function should be evaluated
    p :     list
        list of parameters of the Gauss-function
        [XCEN, YCEN, SIGMAX, SIGMAY, AMP, BACKGROUND, ANGLE];
        SIGMA = FWHM / (2*sqrt(2*log(2)));
        ANGLE = rotation of the X, Y direction of the Gaussian in radians

    Returns
    -------
    array-like
        the value of the Gaussian described by the parameters p at
        position (x, y)
    """
    x, y = M
    arr = np.zeros(x.shape)
    p = []
    if isinstance(params, dict):
        for key in ["XCEN", "YCEN", "SIGMAX", "SIGMAY", "AMP", "BACKGROUND", "ANGLE"]:
            p.append(params[key])


    #print(p)
    for i in range(len(p)//7):

        rcen_x = p[i*7] * np.cos(np.radians(p[i*7+6])) - p[i*7+1] * np.sin(np.radians(p[i*7+6]))
        rcen_y = p[i*7] * np.sin(np.radians(p[i*7+6])) + p[i*7+1] * np.cos(np.radians(p[i*7+6]))
        xp = x * np.cos(np.radians(p[i*7+6])) - y * np.sin(np.radians(p[i*7+6]))
        yp = x * np.sin(np.radians(p[i*7+6])) + y * np.cos(np.radians(p[i*7+6]))

        arr += p[i*7+5] + p[i*7+4] * np.exp(-(((rcen_x - xp) / p[i*7+2]) ** 2 +
                                  ((rcen_y - yp) / p[i*7+3]) ** 2) / 2.)
    return arr

# gaussian function
def gaussian(x, amp, mu, std, bg):
    """
    Gaussian function.

    f(x) = amp * exp(-(x-mu)^2/(2*std)^2) + bg

    To get an educated guess of the skewness parameter alpha, use:

    `alpha_guess = scipy.stats.skew(hist_data)` and use this in your 
    initial parameters.
    
    Parameters
    ----------
    x       :   1D np.array
            x-axis array of 
    amp     :   float
            Amplitude of function
    mu      :   float
            Peak centre of function
    std     :   float
            Standard deviation of function


    Returns
    ----------
    output  :   numpy array
            Array of values with length `len(x)`

    """
    
    return amp*np.exp(-np.power(x - mu, 2)/(2 * np.power(std, 2))) + bg

# skewed gaussian
def skewed_gauss(x, amp, mu, std, alpha):
    """
    Skewed Gaussian function.

    f(x) = ....

    To get an educated guess of the skewness parameter alpha, use:

    `alpha_guess = scipy.stats.skew(hist_data)` and use this in your 
    initial parameters.
    

    Parameters
    ----------
    x       :   1D np.array
            x-axis array of 

    amp     :   float
            Amplitude of function
    mu      :   float
            Peak centre of function
    std     :   float
            Standard deviation of function
    alpha   :   float
            Level of skewness of gaussian

    Returns
    ----------
    output  :   numpy array
            Array of values with length `len(x)`

    """
    #normpdf = np.exp(-np.power(x - mu, 2)/(2 * np.power(std, 2)))
    normpdf = (1/(std*np.sqrt(2*np.pi)))*np.exp(-(np.power((x-mu),2)/(2*np.power(std,2))))
    normcdf = (0.5*(1+scipy.special.erf((alpha*((x - mu)/std))/(np.sqrt(2)))))
    return 2*amp*normpdf*normcdf

# rayleigh distribution
def rayleigh(x, amp, sigma, bg):
    return amp*x*np.exp((-x**2/(2*sigma**2)))*(1/sigma**2) + bg

# exponential distribution
def exp_dist(x, A, beta):
    return A * np.exp(-x/beta) 

# lorentz distribution
def lorentz(x, amp, width, xc, bg):
    return amp*(1/np.pi) * (0.5 * width)/((x - xc)**2 + (0.5*width)**2)

#line 
def line(x, a, b):
    return a*x + b 

#parabola 
def parabola(x, a, b, c):
    return a*x**2 + b*x + c

# second degree polynomial 
def second_poly(x, a, b, c):
    return a*x**2 + b*x + c 

# cubic polynomial 
def cubic(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d 

# exponential 
def exp(x, a, b):
    return a*np.exp(b*x)

# logarithmic 
def log(x, a, b):
    return a*np.log(x) + b 

# sine wave 
def sine(x, a, b, c, d):
    return a*np.sin(b*x + c) + d


# cosine wave 
def cosine(x, a, b, c, d):
    return a*np.cos(b*x + c) + d


def single_gaussian_function(params, M, data, *args):
    """
    Function to feed `lmfit.minimize` for a single 2D gaussian function fit
    
    Parameters
    ----------
    params : `lmfit.Parameters`
        set of parameters for gaussian function
            "XCEN" : x centre for 2d gaussian
            "YCEN" : y centre for 2d gaussian
            "SIGMAX" : standard deviation in x for 2d gaussian
            "SIGMAY" : standard deviation in y for 2d gaussian
            "AMP" : amplitude of 2d gaussian
            "BACKGROUND" : background of 2d gaussian
            "ANGLE" : angle of rotation of the 2d gaussian
    M : [X,Y] tuple of numpy.array 
        X and Y arrays for X and Y data. X and Y data should each be 2-dimensional
    data : numpy.array
        Z data to be fitted with a single 2D gaussian
    
    
    Returns
    --------
    point - data : np.array
        The difference between the measured data and the simulated 2d gaussian at points X,Y
    """
   
    g1_params = {
        "XCEN" : params["XCEN"].value,
        "YCEN" : params["YCEN"].value,
        "SIGMAX" : params["SIGMAX"].value,
        "SIGMAY" : params["SIGMAY"].value,
        "AMP" : params["AMP"].value,
        "BACKGROUND" : params["BACKGROUND"].value,
        "ANGLE" : params["ANGLE"].value,
    }
    
    
    x, y = M
    
    point = Gauss2d(M, **g1_params)
    return point - data