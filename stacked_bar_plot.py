"""
https://medium.com/@priteshbgohil/stacked-bar-chart-in-python-ddc0781f7d5f
"""
'''
# stacked bar plot
import numpy as np
import matplotlib.pyplot as plt

# Get values from the group and categories
quarter = np.array(["Q1", "Q2", "Q3", "Q4"])
jeans = np.array([100, 75, 50, 133])
tshirt = np.array([44, 120, 150, 33])
formal_shirt = np.array([70, 90, 111, 80])
total = jeans + tshirt + formal_shirt
proportion_jeans = np.true_divide(jeans, total) * 100
proportion_tshirts = np.true_divide(tshirt, total) * 100
proportion_formal = np.true_divide(formal_shirt, total) * 100

# add colors
colors = ['#FF9999', '#00BFFF', '#C1FFC1', '#CAE1FF', '#FFDEAD']
# The position of the bars on the x-axis
r = range(len(quarter))
barWidth = 0.5
# plot bars
plt.figure(figsize=(10, 7))
ax1 = plt.barh(r, proportion_jeans, bottom=proportion_tshirts + proportion_formal, color=colors[0], edgecolor='white',
              width=barWidth, label="jeans")
ax2 = plt.barh(r, proportion_tshirts, bottom=proportion_formal, color=colors[1], edgecolor='white', width=barWidth,
              label='tshirt')
ax3 = plt.barh(r, proportion_formal, color=colors[2], edgecolor='white', width=barWidth, label='formal shirt')
# plt.legend()
# plt.xticks(r, quarter, fontweight='bold')
# plt.ylabel("sales")
plt.savefig("percentileStacked.png")
plt.show()
'''


"""
https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/horizontal_barchart_distribution.html#sphx-glr-gallery-lines-bars-and-markers-horizontal-barchart-distribution-py

https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

https://hot-time.tistory.com/2888
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict


category_names = ['Strongly disagree', 'Disagree',
                  'Neither agree nor disagree', 'Agree', 'Strongly agree']
results = OrderedDict({
    'Question 1': [10, 15, 17, 32, 26],
    'Question 2': [26, 22, 29, 10, 13],
    'Question 3': [35, 37, 7, 2, 19],
    'Question 4': [32, 11, 9, 15, 33],
    'Question 5': [21, 29, 5, 5, 40],
    'Question 6': [8, 19, 5, 30, 38]
})

# results = OrderedDict({
#     'GT': {'class': ['coming', 'sitting', 'reading', 'nodding off'], 'num_frames':[5, 5, 5, 5]},
#     'I3D': {'class': ['coming', 'sitting', 'reading', 'nodding off'], 'num_frames':[4, 7, 2, 7]},
#     'Multi-scale I3D': {'class': ['coming', 'sitting', 'reading', 'nodding off'], 'num_frames':[7, 8, 3, 2]}
# })

save_name = 'stacked_bar_horizontal_1.png'


def survey(results, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())

    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('PuBu')(
        np.linspace(0.15, 0.85, data.shape[1]))
    np.random.shuffle(category_colors)

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,
                label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            ax.text(x, y, str(int(c)), ha='center', va='center',
                    color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    # get rid of the frame
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    # remove all the ticks and directly label each bar with respective value
    plt.tick_params(left='off')

    return fig, ax


survey(results, category_names)
# survey(results)

plt.savefig(save_name)
plt.show()

