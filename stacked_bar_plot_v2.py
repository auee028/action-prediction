"""
https://medium.com/@priteshbgohil/stacked-bar-chart-in-python-ddc0781f7d5f

https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/horizontal_barchart_distribution.html#sphx-glr-gallery-lines-bars-and-markers-horizontal-barchart-distribution-py

https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

https://hot-time.tistory.com/2888
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict


results = [
    ['GT', 'I3D', 'Multi-scale I3D'],
    [['coming', 'sitting', 'reading', 'nodding off'], ['coming', 'sitting', 'reading', 'nodding off'], ['coming', 'sitting', 'reading', 'None']],
    [[5, 5, 5, 5], [4, 7, 2, 7], [7, 8, 3, 2]]
]

with open('categories.txt') as f:
    categories = map(lambda x: x.strip(), f.readlines())

category_colors = plt.get_cmap('Paired')(
        np.linspace(0.1, 0.9, len(categories)))
np.random.shuffle(category_colors)

categories = ['None'] + categories
category_colors = np.array([[0.55, 0.55, 0.55, 1.0]] + category_colors.tolist())


save_name = 'stacked_bar_horizontal_1.png'


#
models = results[0]
labels = results[1]
data = np.array(results[2])

data_cum = data.cumsum(axis=1)

fig, ax = plt.subplots(figsize=(9.2, 3))
ax.invert_yaxis()
ax.xaxis.set_visible(False)
ax.set_xlim(0, np.sum(data, axis=1).max())

for k in range(len(models)):
    for i, (colname, width, width_cum) in enumerate(zip(labels[k], data[k], data_cum[k])):
        start = width_cum - width
        color = category_colors[categories.index(colname)]
        ax.barh(models[k], width, left=start, height=0.5,
                label=colname, color=color)

# ax.legend(ncol=len(categories), bbox_to_anchor=(0, 1),
#               loc='lower left', fontsize='small')

# get rid of the frame
for spine in plt.gca().spines.values():
    spine.set_visible(False)

# remove all the ticks and directly label each bar with respective value
    plt.tick_params(left='off')


plt.savefig(save_name)
plt.show()

