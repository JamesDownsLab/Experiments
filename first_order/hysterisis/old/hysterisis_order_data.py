from labvision import images, video
from particletracking import dataframes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def order_duty_mapped(df):
    df['order'] = (df.order_r**2+df.order_i**2)**0.5
    group = df.groupby('frame')['order'].mean()
    start = df.head().Duty.mean()
    end = df.tail().Duty.mean()
    D = pd.Series(group.values, np.linspace(start, end, len(group.index)))
    rolling_mean = D.rolling(window=100).mean()
    return D.index, rolling_mean

def order_duty(df):
    group = df.groupby('Duty')['order'].mean()
    return group.index, group.values

direc = "/media/data/Data/March2020/Hysterisis/HalfTray"
save_direc = "/media/data/Data/March2020/Hysterisis/HalfTray/Results/OrderFigure/"
files = {
    '0.6': ['16870001', '16870002'],
    '0.4': ['16870003', '16870004'],
    '0.2': ['16870005', '16870006'],
    '0.1': ['16870007', '16870008'],
    # '0.05': ['16870009', '16870010'],
    # '0.8': ['16870011', '16870012'],
    # '1': ['16870013', '16870014'],
    # '2': ['16870015', '16870016']
}

files = {r: [direc+d, direc+u] for r, (d, u) in files.items()}
for i, (rate, (down, up)) in enumerate(files.items()):
    print(i)
    fig, ax = plt.subplots()
    ax.set_title(rate)
    down_data = dataframes.DataStore(down+'.hdf5')
    up_data = dataframes.DataStore(up+'.hdf5')

    # if i == 0:
    #     up_video = video.ReadVideo(up + '.MP4')
    #     frame = up_video.read_next_frame()
    #     frame = images.crop(frame, up_data.metadata['crop'])
    #     result = images.crop_polygon(frame)
    #     bbox = result.bbox

    down_df = down_data.df#[down_data.df.x.between(bbox.xmin, bbox.xmax)]
    # down_df = down_df#[down_df.y.between(bbox.ymin, bbox.ymax)]

    up_df = up_data.df#[up_data.df.x.between(bbox.xmin, bbox.xmax)]
    # up_df = up_df[up_df.y.between(bbox.ymin, bbox.ymax)]

    down_duty, down_order = order_duty(down_df)
    up_duty, up_order = order_duty(up_df)

    ax.plot(down_duty, down_order, label='down')
    ax.plot(up_duty, up_order, label=' up ')
    ax.set_xlabel('Duty')
    ax.set_ylabel('Order')
    ax.legend()

    results = pd.DataFrame.from_dict(
        {
            'duty_down': down_duty,
            'order_down': down_order,
            'duty_up': up_duty,
            'order_up': up_order
        },
        orient='index'
    ).T
    results.to_csv(save_direc+'rate={}'.format(rate))

plt.show()