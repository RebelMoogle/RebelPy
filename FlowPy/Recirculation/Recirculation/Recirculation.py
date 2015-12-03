import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.integrate import ode
import VectorField

def DefaultAnalytic(t, y):
    """ Default Analytic Vector Field calculation, 
        takes time t as scalar and 
        position y as ndarray"""
    return np.array([3*(1-t)**2-12*t*(1-t)+3*t**2+(y[2]-3*t*(1-t)**2-3*t**2*(1-t)-(1/2*(1-y[0]+3*t*(1-t)**2-3*t**2*(1-t)))*(1+y[0]-3*t*(1-t)**2+3*t**2*(1-t))*(1-y[1]+3*t*(1-t)**2-3*t**2*(1-t))*(1+y[1]-3*t*(1-t)**2+3*t**2*(1-t))+1/2)*((1/2)*(y[0]-3*t*(1-t)**2+3*t**2*(1-t))**2+(3/10)*(y[1]-3*t*(1-t)**2+3*t**2*(1-t))**2-(2/5*(y[0]-3*t*(1-t)**2+3*t**2*(1-t)))*(y[1]-3*t*(1-t)**2+3*t**2*(1-t))**2-4/5),
                    3*(1-t)**2-12*t*(1-t)+3*t**2+(y[2]-3*t*(1-t)**2-3*t**2*(1-t)-(1/2*(1-y[0]+3*t*(1-t)**2-3*t**2*(1-t)))*(1+y[0]-3*t*(1-t)**2+3*t**2*(1-t))*(1-y[1]+3*t*(1-t)**2-3*t**2*(1-t))*(1+y[1]-3*t*(1-t)**2+3*t**2*(1-t))+1/2)*((1/5)*(y[0]-3*t*(1-t)**2+3*t**2*(1-t))**2-(3/5)*(y[1]-3*t*(1-t)**2+3*t**2*(1-t))**2-(2/5*(y[0]-3*t*(1-t)**2+3*t**2*(1-t)))*(y[1]-3*t*(1-t)**2+3*t**2*(1-t))**2-1/2),
                    3*(1-t)**2-3*t**2+(y[2]-3*t*(1-t)**2-3*t**2*(1-t)-(1/2*(1-y[0]+3*t*(1-t)**2-3*t**2*(1-t)))*(1+y[0]-3*t*(1-t)**2+3*t**2*(1-t))*(1-y[1]+3*t*(1-t)**2-3*t**2*(1-t))*(1+y[1]-3*t*(1-t)**2+3*t**2*(1-t))+1/2)*((7/10)*(y[0]-3*t*(1-t)**2+3*t**2*(1-t))**2-(1/5)*(y[1]-3*t*(1-t)**2+3*t**2*(1-t))**2-(1/2*(y[0]-3*t*(1-t)**2+3*t**2*(1-t)))*(y[1]-3*t*(1-t)**2+3*t**2*(1-t))**2-4/5)])

currentField = VectorField.VectorField(DefaultAnalytic, "Test33")

def CalculateAllDistancesFor(startPosition, minTime, maxTime, display=True, numSteps = 100):
    """ Calculate all distances for every t and tau between min and maxtime, 
        with # of steps """
    minT, minTau = minTime
    maxT, maxTau = maxTime
    allTs = np.linspace(minT, maxT, numSteps)
    allTaus = np.linspace(minTau, maxTau, numSteps)
    distDataType = np.dtype([('Distance', np.float64), ('DistanceTau', np.float64)])
    allDistances = np.empty((numSteps,numSteps), dtype=distDataType)
    doprIntegrator = ode(currentField.GetDataAt).set_integrator("dopri5")

    distIterator = np.nditer(allDistances, flags=['multi_index'], op_flags=['readwrite'])
    while not distIterator.finished:
        t = allTs[distIterator.multi_index[0]]
        tau = allTaus[distIterator.multi_index[1]]
        doprIntegrator.set_initial_value(startPosition, t)
        distance = linalg.norm(startPosition - doprIntegrator.integrate(t+tau))
        print(" distance calculated (t: {}, tau: {}): {}".format(t, tau, distance))
        distIterator[0] = (distance, distance / tau)
        distIterator.iternext()

    if display:
        import mpl_toolkits.mplot3d.axes3d as axes3d
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
        ax.plot_surface(allTs, allTaus, allDistances['DistanceTau'], cmap=plt.get_cmap('jet'), rstride=1, cstride=1, vmin=-150, vmax=150, shade=True)
        plt.show()
        
    return allDistances
    

if __name__ == "__main__":
    import sys
    if(len(sys.argv) < 2):
        print("Not enough arguments, using standard field: {}".format(currentField.fieldName))
    else:
       currentField = VectorField(sys.argv[1], sys.argv[2])
    CalculateAllDistancesFor(np.array([0,0,0]), (-1,-1), (1, 3))
    #plot distances

