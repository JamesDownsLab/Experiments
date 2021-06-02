from labvision import images, video
from particletracking import dataframes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def density_duty(df):
    group = df.groupby('Duty')['density'].mean()
    return group.index, group.values

def density_duty_mapped(df):
    group = df.groupby('frame')['density'].mean()
    start = df.head().Duty.mean()
    end = df.tail().Duty.mean()
    D = pd.Series(group.values, np.linspace(start, end, len(group.index)))
    return D.index, D.values


direc = "/media/data/Data/March2020/Hysterisis/"
save_direc = "/media/data/Data/March2020/Hysterisis/Results/Figure/"

files = {
'0.4': ['16870003', '16870004'],
    '0.6': ['16870001', '16870002'],

    '0.2': ['16870005', '16870006'],
    '0.1': ['16870007', '16870008'],
    '0.05': ['16870009', '16870010'],
    '0.8': ['16870011', '16870012'],
    '1': ['16870013', '16870014'],
    '2': ['16870015', '16870016']
}

files = {r: [direc+d, direc+u] for r, (d, u) in files.items()}
for i, (rate, (down, up)) in enumerate(files.items()):
    print(i)
    print(rate)
    fig, ax = plt.subplots()
    ax.set_title(rate)
    down_data = dataframes.DataStore(down+'.hdf5')
    up_data = dataframes.DataStore(up+'.hdf5')

    if i == 0:
        up_video = video.ReadVideo(up + '.MP4')
        frame = up_video.read_next_frame()
        frame = images.crop(frame, up_data.metadata['crop'])
        result = images.crop_polygon(frame)
        bbox = result.bbox
        with open(save_direc+'bbox.txt', 'w') as f:
            f.writelines(['BBOX xmin, xmax, ymin, ymax', str(bbox.xmin), str(bbox.xmax), str(bbox.ymin), str(bbox.ymax)])
        frame = images.draw_polygon(frame,
                                    np.array([[bbox.xmin, bbox.ymin], [bbox.xmin, bbox.ymax], [bbox.xmax, bbox.ymax], [bbox.xmax, bbox.ymin]], dtype=np.float32), color=images.RED, thickness=2)
        images.save(frame, save_direc+'im.png')

    down_df = down_data.df[down_data.df.x.between(bbox.xmin, bbox.xmax)]
    down_df = down_df[down_df.y.between(bbox.ymin, bbox.ymax)]

    up_df = up_data.df[up_data.df.x.between(bbox.xmin, bbox.xmax)]
    up_df = up_df[up_df.y.between(bbox.ymin, bbox.ymax)]

    # down_duty, down_density = density_duty_mapped(down_df)
    # up_duty, up_density = density_duty_mapped(up_df)

    down_duty, down_density = density_duty_mapped(down_df)
    up_duty, up_density = density_duty_mapped(up_df)

    # np.savetxt(save_direc+'duty_down_rate={}'.format(rate),
    #            down_duty)
    # np.savetxt(save_direc+'density_down_rate={}'.format(rate),
    #            down_density)
    # np.savetxt(save_direc + 'duty_up_rate={}'.format(rate),
    #            up_duty)
    # np.savetxt(save_direc + 'density_up_rate={}'.format(rate),
    #            up_density)

    results = pd.DataFrame.from_dict({'duty_down': down_duty,
                           'density_down': down_density,
                           'duty_up': up_duty,
                           'density_up': up_density}, orient='index').T
    results.to_csv(save_direc+'rate={}'.format(rate))
    #
    # ax.plot(down_duty, down_density, label='down')
    # ax.plot(up_duty, up_density, label=' up ')
    # ax.set_xlabel('Duty')
    # ax.set_ylabel('Density')
    # ax.legend()

plt.show()