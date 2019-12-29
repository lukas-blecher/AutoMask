from geomdl import BSpline
from geomdl import utilities
from geomdl import fitting
from PIL import Image, ImageDraw
from scipy.misc import imresize
from skimage import measure
import numpy as np
import logging
logger = logging.getLogger('global')

mapping = {0: np.array([-1., -1.]),
           1: np.array([-1.,  0.]),
           2: np.array([0., -1.]),
           3: np.array([-1.,  1.]),
           4: np.array([0., 0.]),
           5: np.array([1., 1.]),
           6: np.array([1., 0.]),
           7: np.array([0., 1.]),
           8: np.array([1., -1.])}


def dir2num(x):
    length = np.sqrt((x**2).sum())
    if length > 0:
        x=np.round(x/length)
        x=np.round(np.sign(x)))
    for i in range(9):
        if (x == mapping[i]).all():
            return i
    print(x)
    return 4


def bspline2mask(cps, width, height, delta=0.05, scaling=5):
    connecs = []
    for i in range(len(cps)):
        curve = BSpline.Curve()
        curve.degree = 3
        curve.ctrlpts = cps[i]
        curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))
        # print('delta',delta)
        curve.delta = delta
        curve.evaluate()
        connecs.append(curve.evalpts)

    polygon = np.array(connecs).flatten().tolist()
    img = Image.new('L', (width, height), 255)
    ImageDraw.Draw(img).polygon(polygon, outline=0, fill=0)
    mask = np.array(img.resize((width//scaling, height//scaling), Image.NEAREST))
    return mask == False


def crl2mask(crl, width, height, delta=.05, scaling=1):
    c, r, l = crl if type(crl) == list else crl.tolist()
    cps = []
    for i in range(len(c)):
        ip = (i+1) % len(c)
        cps.append([c[i], r[i], l[ip], c[ip]])
    return bspline2mask(cps, width, height, delta=delta, scaling=scaling)


def mask2rect(mask):
    y, x = np.where(mask == 1)
    mi, ma = np.array((x.min(), y.min())), np.array((x.max(), y.max()))
    return (mi+ma)/2, (ma-mi)


def make_cirular(cps, distance=3):
    # takes cps returns crl
    cps = np.array(cps)
    start, end = cps[:, 0, :], cps[:, -1, :]
    # first test for approximate circularity
    absmask = [np.abs(start[:, i].T[None, ...]-end[:, None, i]) for i in (0, 1)]
    for i in range(2):
        matrix = absmask[i] < distance
        assert matrix.sum(0).all() and matrix.sum(1).all()
    e, s = np.where(matrix)
    c = np.mean(np.array([end[e], start[s]]), axis=0)
    r = cps[:, 1, :]
    l = cps[:, 2, :]
    return np.array([c, r, l])[...,[1,0]]


def fit2mask(target, maxnum=4, distance=3):
    contours = measure.find_contours(target, .8)
    # choose contour with highest point count
    c = contours[np.argmax([len(c) for c in contours])]
    import matplotlib.pyplot as plt
    #plt.plot(c[:, 1], c[:, 0], linewidth=2)
    #plt.show()
    # convert to directions and remove unnecessary points
    direction = []
    last = c[0]
    del_inds = []
    for i in range(-1, len(c)):
        number = dir2num(last-c[i])
        if number == 4:
            del_inds.append(i % len(c))
            continue
        direction.append(number)
        last = c[i]
    c = np.delete(c, del_inds, axis=0)
    direction = np.array(direction)
    # split curve into segments
    uni, first = [], 0
    for i in range(len(direction)):
        if len(np.unique(direction[:i])) > maxnum:
            first = i-1
            break
    breaks = [first]
    for i in range(first, first+len(direction)):
        i = i % len(direction)
        if i >= breaks[-1]:
            lui = len(np.unique(direction[breaks[-1]:i]))
        else:
            lui = len(np.unique(direction[np.concatenate([np.arange(breaks[-1], len(direction)), np.arange(i)])]))
        if lui > maxnum:
            breaks.append((i-1) % len(direction))
        uni.append(lui % (maxnum+1))
    # sort points into found segements
    segments = []
    split_ind = np.split(np.arange(len(direction)), sorted(breaks))
    split_ind[0] = np.concatenate((split_ind[-1], split_ind[0]))
    del split_ind[-1]
    for ind in split_ind:
        segments.append(c[ind%len(c)])
    succ = True
    crl = None


    try:
        # check that we have all points
        assert sum([len(s) for s in segments]) >= len(c)
        # fit bspline curves to the segments
        final_cps = []
        for i in range(len(segments)):
            points = segments[i].tolist()
            curve = fitting.approximate_curve(points, 3, ctrlpts_size=4)
            final_cps.append(curve.ctrlpts)
            #curve.delta = .2
            #evalpts = np.array(curve.evalpts)
            #pts = np.array(points)
            #plt.plot(evalpts[:, 0], evalpts[:, 1],label=i)
            #plt.scatter(pts[:, 0], pts[:, 1], s=.2,color="red")
        #plt.legend()
        #plt.show()
        crl = make_cirular(final_cps, distance).tolist()
    except AssertionError:
        succ = False
        logger.info('No approximation to the mask could be found. Try again with other parameters.')
    return succ, crl
