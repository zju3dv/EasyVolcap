"""
Load recorded timing json file and plot the timing results with matplotlib
"""
from easyvolcap.utils.console_utils import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.style.use('dark_background')
# Font path and registration
font_path = 'assets/fonts/CascadiaCodePL-SemiBold.otf'
font_prop = FontProperties(fname=font_path)


def smooth_data(data: np.ndarray, weight=0.85):
    smoothed = []
    last = data[0]
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    # left = data * weight
    # right = data * (1 - weight)
    # weights = np.full_like(data, weight)

    # cum_weights = np.cumprod(weights)
    # cum_weights = np.concatenate([[1], cum_weights[:-1]])
    # smoothed = cum_weights * right

    return smoothed


@ catch_throw
def main():
    args = dotdict(
        data='data/record/tfgs_bike_f37300_time/tfgs_bike_f37300_time.json',
    )
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))

    with open(args.data, 'r') as f:
        data = json.load(f)
    data = dotdict(data)

    os.makedirs(splitext(args.data)[0], exist_ok=True)

    ylim_ratio = 0.999
    # ylim = 0

    for name, values in data.items():
        steps = np.arange(len(values))
        values = np.asarray(values)
        smoothed_values = smooth_data(values)

        plt.figure(figsize=(8, 6), dpi=300)  # Set figure size and resolution
        plt.plot(steps, values, alpha=0.3)  # Original data with reduced opacity
        plt.plot(steps, smoothed_values, label=name)

        ylim_top = int(len(steps) * (1 - ylim_ratio))
        ylim = min(values[np.argpartition(values, ylim_top)[-ylim_top:]]) * 1.5
        # ylim = max(ylim, min(values[np.argpartition(values, ylim_top)[-ylim_top:]]))

        plt.ylim(0, ylim)
        plt.xlabel('Steps', fontproperties=font_prop)
        plt.ylabel('Timing (s)', fontproperties=font_prop)
        plt.title(name, fontproperties=font_prop)
        plt.legend()
        # plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

        fig_name = join(splitext(args.data)[0], f'{name}.pdf')
        plt.savefig(fig_name)
        plt.close()

        log(yellow(f'Plotted image saved to: {blue(fig_name)}'))


if __name__ == '__main__':
    main()
