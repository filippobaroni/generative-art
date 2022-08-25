from itertools import pairwise
import sys

from numba import jit

import numpy as np
from scipy.spatial import Voronoi
from scipy.ndimage import gaussian_filter, label
from skimage import morphology
import cv2

from shapely.geometry import Polygon

import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFilter as ImageFilter
from PIL.ImageColor import getrgb


'''
Run as
> python3 mosaic.py <width> <height>
The mosaic will be saved as "out_<width>x<height>.png".

Other options can be customised by editing the source directly.
* supersample: anything above 4 should be indistinguishable from 4.
* points_density: parameter describing how many regions the final mosaic will have;
    the higher the parameter, the more regions there will be.
* palette: the palette used to colour the Julia fractal.
* reference_img: by default, the Julia set is used as a reference for the mosaic,
    but an arbitrary image can be used instead.
'''

img_w, img_h = int(sys.argv[1]), int(sys.argv[2])

# supersample
# must be a positive integer
supersample = 8

# points_density
# the higher the number, the denser the mosaic will be
points_density = 1 / 500
# to get finer control on the density of the mosaic, one can directly
# change the value of points_n, controlling the exact number of regions
# in the mosaic
points_n = int(img_w * img_h * points_density)



def make_palette(d):
    d[1.0] = d[0]
    return np.concatenate(
        [255 * np.linspace(np.array(u), np.array(v), int((b - a) * 1e6)) for (a, u), (b, v) in pairwise(sorted(d.items()))]
        ).astype('uint8')

def img_from_palette(img, palette):
    img = np.log2(np.maximum(img, 1))
    img = img * .3 + .0
    return Image.fromarray(palette[((img % 1) * palette.shape[0]).astype('int')], 'RGB')

# palette
# the palette used to colour the Julia fractal
# it is a dictionary whose entries are of the form "x : (r, g, b)"
palette = make_palette({        
        .00 : (1., .2, 0.),
        .20 : (1., .6, 0.),
        .35 : (1., 1., 0),
        .45 : (.2, .9, 0.),
        .75 : (0., .4, .4),
        .90 : (0., 0., 0.)
    })

tocolor = lambda x: tuple(x.astype('int'))
white = np.array(getrgb('white'))
black = np.array(getrgb('black'))

rng = np.random.default_rng()

@jit
def make_julia(c, width, height, radius, max_iterations):
    x2 = width / np.hypot(width, height) * radius
    y2 = height / np.hypot(width, height) * radius
    x1 = -x2
    y1 = -y2
    div_time = np.zeros((height, width), dtype = 'int')
    zs = np.zeros((height, width), dtype = 'float64')
    xs = np.linspace(x1, x2, width)
    ys = np.linspace(y1, y2, height)
    for i in range(width):
        for j in range(height):
            z = xs[i] + 1j * ys[j]
            for t in range(max_iterations):
                z = z ** 2 + c
                if abs(z) > 4:
                    zs[j, i] = np.abs(z)
                    div_time[j, i] = t
                    break
    div_time = div_time + 1 - np.log2(np.maximum(1, np.log(zs) / np.log(4)))
    return div_time

julia = img_from_palette(make_julia(-.8 + .18j, img_w, img_h, .9, 2000), palette)

print('Julia done')

# reference_img
# the image used as a reference for the mosaic
# to use an arbitrary picture, simply put the image file in the same directory
# as the script and replace "julia" in the following line with "Image.open(<filename>)"
reference_img = julia

reference_w, reference_h = reference_img.size
if img_w / img_h > reference_w / reference_h:
    new_w = reference_w
    new_h = int(reference_w * img_h / img_w)
else:
    new_h = reference_h
    new_w = int(reference_h * img_w / img_h)
delta_x = (reference_w - new_w) // 2
delta_y = (reference_h - new_h) // 2
reference_img = reference_img.crop((delta_x, delta_y, delta_x + new_w, delta_y + new_h)).resize((img_w, img_h))

def compute_saliency(img):
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    _, saliency_map = saliency.computeSaliency(np.array(img))
    saliency_map = gaussian_filter(saliency_map, sigma = (img.width + img.height) / 1000)
    saliency_map /= np.max(saliency_map)
    return (saliency_map + .01) / 1.01

density = compute_saliency(reference_img)

img = Image.new('RGB', (img_w, img_h), 'white')
draw = ImageDraw.Draw(img)

def sample_with_density(density_map, n):
    points = np.zeros((n, 2), 'float64')
    while n > 0:
        y = rng.integers(0, density_map.shape[0])
        x = rng.integers(0, density_map.shape[1])
        p = rng.random()
        if p <= density_map[y, x]:
            n -= 1
            points[n] = [x, y]
    return points

points = sample_with_density(density, points_n)
a = np.linspace(0, 2 * np.pi, 10, endpoint = False)
points = np.vstack([points, 10 * (img_w + img_h) * np.array([np.cos(a), np.sin(a)]).T])

voronoi = Voronoi(points)
vertices = voronoi.vertices
regions = [r + [r[0]] for r in voronoi.regions if len(r) > 0 and -1 not in r]

print('Voronoi done')

def dominant_color(img, mask):
    c = np.sum(mask)
    if c:
        return np.sum(img * mask[:, :, None], axis = (0, 1)) / c
    else:
        return np.array([0, 0, 0])

cnt = 0
for r in regions:
    vs = vertices[r].astype('int')
    bbox1 = np.maximum([0, 0], np.min(vs, axis = 0))
    bbox2 = np.minimum([img_w, img_h], np.max(vs, axis = 0) + 1)
    if np.any(bbox1 >= bbox2):
        continue
    
    mask_img = Image.new('1', tuple(bbox2 - bbox1), 0)
    mask_draw = ImageDraw.Draw(mask_img)
    mask_draw.polygon(list(np.hstack(vs - bbox1)), fill = 1)
    w = 3 * np.sqrt(np.sum(np.array(mask_img)) * points_n / (img_w * img_h))
    col = dominant_color(np.array(reference_img.crop(tuple(np.hstack([bbox1, bbox2])))), np.array(mask_img))
    
    mask_img = Image.new('1', tuple(supersample * (bbox2 - bbox1)), 0)
    mask_draw = ImageDraw.Draw(mask_img)
    mask_draw.polygon(list(supersample * np.hstack(vs - bbox1)), fill = 1, outline = 0,
        width = int(w * supersample))
    
    poly = Polygon(supersample * (vs - bbox1))
    poly = poly.buffer(-w * supersample).buffer(.6 * w * supersample, resolution = 16 * supersample)
    coords = sum(poly.exterior.coords,())
    if len(coords) == 0:
        continue
    
    mask_img = Image.new('L', tuple(supersample * (bbox2 - bbox1)), 0)
    mask_draw = ImageDraw.Draw(mask_img)
    mask_draw.polygon(sum(poly.exterior.coords,()), fill = 255)
    mask_img = mask_img.convert('RGB').resize((mask_img.width // supersample, mask_img.height // supersample),
        resample = Image.Resampling.BILINEAR).convert('L')
    img.paste(tocolor(col), tuple(bbox1), mask_img)
    
    cnt += 1
    if int(cnt * 100 / len(regions)) > int((cnt - 1) * 100 / len(regions)):
        print(f'{int(cnt * 100 / len(regions)): 2}%')

img.save(f'out-{img_w}x{img_h}.png')
