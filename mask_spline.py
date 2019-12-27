from geomdl import BSpline
from geomdl import utilities
from geomdl import exchange
from geomdl.visualization import VisMPL
from PIL import Image, ImageDraw
from scipy.misc import imresize
from scipy.optimize import minimize
import numpy as np


def bspline2mask(cps, width, height, delta=0.05, scaling=5):
    connecs = []
    for i in range(len(cps)):
        curve = BSpline.Curve()
        curve.degree = 3
        curve.ctrlpts = cps[i]
        curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))
        curve.delta = delta
        curve.evaluate()
        connecs.append(curve.evalpts)

    polygon = np.array(connecs).flatten().tolist()
    img = Image.new('L', (width, height), 255)
    ImageDraw.Draw(img).polygon(polygon, outline=0, fill=0)
    mask = np.array(img.resize((width//scaling, height//scaling), Image.NEAREST))
    return mask == False


def crl2mask(crl, delta=.05, scaling=5):
    c, r, l = crl if type(crl) == list else crl.tolist()
    cps = []
    for i in range(len(c)):
        ip = (i+1) % len(c)
        cps.append([c[i], r[i], l[ip], c[ip]])
    return bspline2mask(cps, delta, scaling)


def fit2mask(crl, scaling=5):
    target = crl2mask(crl, scaling=scaling)

    def loss(pred, target=target):
        return (target ^ pred).sum()/target.sum()

    def fun(crl):
        pred = crl2mask(crl.reshape(3, -1, 2), scaling)
        return loss(pred)

    x0 = np.asarray(crl).flatten()
    res = minimize(fun, x0, method='nelder-mead')
    succ = res.success and res.fun < 0.1
    return succ, res.x.reshape(3, -1, 2)
