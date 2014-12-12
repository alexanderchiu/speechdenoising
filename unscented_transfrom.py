import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).

    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)


def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip

if __name__ == '__main__':
    #-- Example usage -----------------------
    # Generate some random, correlated data
    points = np.random.multivariate_normal(
        mean=(1, 1), cov=[[1, 2], [1, 5]], size=1000
    )
    # Plot the raw points...
    x, y = points.T
    plt.plot(x, y, 'ro', markersize=5)
    plt.plot(1, 1, '*b', markersize=10)
    # Plot a transparent 3 standard deviation covariance ellipse
    plot_point_cov(points, nstd=3, alpha=0.5, color='green')

    mean = np.mean(points, axis=0)
    cov = np.cov(points.T)

    n, d = points.shape
    L = 2 * d + 1
    alpha = 0.01
    beta = 2
    kappa = 0
    print cov

    lmda = np.power(alpha, 2) * (d + kappa) - d
    sigma = np.zeros((L, d))
    wm = np.zeros(L)
    wc= np.zeros(L)
    print lmda
    sigma[0,:] = mean
    wm[0] = lmda/(d+lmda)
    wc[0] = lmda/(d+lmda) + (1 - np.power(alpha,2)+beta)




    for i in range(d):
       print (d+lmda)*cov
       print linalg.sqrtm((d+lmda)*cov)
       sigma[i+1,:] = mean + np.linalg.cholesky((d+lmda)*cov)[i,:]

    for i in range(d, 2*d):
        sigma[i+1,:] = mean - np.linalg.cholesky((d+lmda)*cov)[i-d,:]

    for i in range(2*d):
    	wc[i+1] = 1/(2*(d+lmda))
    	wm[i+1] = 1/(2*(d+lmda))

    print sigma
    print wc
    plt.plot(sigma[:, 0], sigma[:, 1], '*g')

    plt.show()
