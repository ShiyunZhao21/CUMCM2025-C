import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

Q31_df = pd.read_excel("人群数据.xlsx")

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

Q31_fig = plt.figure(figsize=(12, 8))
Q31_ax = Q31_fig.add_subplot(111, projection='3d')

Q31_age_bins = np.arange(20, 40, 2)
Q31_bmi_bins = np.arange(20, 40, 2)
Q31_hist, Q31_xedges, Q31_yedges = np.histogram2d(
    Q31_df['Age'],
    Q31_df['BMI'],
    bins=[Q31_age_bins, Q31_bmi_bins]
)

Q31_xpos, Q31_ypos = np.meshgrid(Q31_xedges[:-1], Q31_yedges[:-1], indexing="ij")
Q31_xpos = Q31_xpos.flatten()
Q31_ypos = Q31_ypos.flatten()
Q31_zpos = np.zeros_like(Q31_xpos)

Q31_dx = np.diff(Q31_xedges)[0] * np.ones_like(Q31_zpos)
Q31_dy = np.diff(Q31_yedges)[0] * np.ones_like(Q31_zpos)
Q31_dz = Q31_hist.flatten()

Q31_ax.bar3d(
    Q31_xpos, Q31_ypos, Q31_zpos,
    Q31_dx, Q31_dy, Q31_dz,
    color='skyblue',
    alpha=0.8,
    edgecolor='gray',
    shade=True
)

Q31_ax.set_xlabel('年龄', fontsize=12, labelpad=10)
Q31_ax.set_ylabel('BMI', fontsize=12, labelpad=10)
Q31_ax.set_zlabel('频率', fontsize=12, labelpad=10)
Q31_ax.set_title('年龄-BMI三维频数分布图', fontsize=16, pad=20)

Q31_ax.xaxis._axinfo["grid"].update({"linewidth": 0.3, "linestyle": ":"})
Q31_ax.yaxis._axinfo["grid"].update({"linewidth": 0.3, "linestyle": ":"})
Q31_ax.zaxis._axinfo["grid"].update({"linewidth": 0.3, "linestyle": ":"})

Q31_ax.view_init(elev=25, azim=45)

Q31_mappable = plt.cm.ScalarMappable(cmap=plt.cm.Blues)
Q31_mappable.set_array(Q31_dz)
Q31_fig.colorbar(Q31_mappable, ax=Q31_ax, shrink=0.5, aspect=5, label='频率')

plt.tight_layout()
plt.show()
