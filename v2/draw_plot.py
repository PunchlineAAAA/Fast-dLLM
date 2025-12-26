import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

mean = np.array([
0.24313188323726875,0.17700271885831712,0.17943704554933376,0.2169294081360968,
0.642216986947571,0.6397014019578507,0.4672615371022757,0.40544724686575334,
0.5005193508536225,0.41329128123621234,0.44813623276271897,0.4704446108540375,
0.43525090853884135,0.33577952922801146,0.4496620025000243,0.43372936486585234,
0.3846424223329528,0.42124842817028824,0.3415575257898251,0.3713023843590417,
0.3947761490270061,0.3860821284046375,0.5043292297570313,0.47342600737652973,
0.480811055022581,0.4093490823513907,0.4641454643171686,0.3507369486936556
])
std = np.array([
0.06480530133247314,0.059989425928329156,0.05186402878209224,0.06359646728092085,
0.08401796105694334,0.07682757898830785,0.09277565309249361,0.08627800136699645,
0.08139296801798743,0.08643953053621947,0.0914485963550847,0.08399364021752316,
0.08551043010244289,0.08768439390781776,0.08347015942521234,0.0794632222209463,
0.09248946857067575,0.10923885577605072,0.08854705767880669,0.0805107481145071,
0.09039331392398196,0.09761440592907061,0.10048422475894228,0.10584008753025083,
0.10726456289830434,0.10171767923273876,0.09526621995356108,0.09871444715224888
])


# 把两行拼成一个矩阵（2 x N）
data = np.vstack([mean, std])

# colormaps
cmap_mean = LinearSegmentedColormap.from_list("mean", ["#ffffff","#fffab0","#e6de70","#9bc53d","#228b22"])
cmap_std  = LinearSegmentedColormap.from_list("std",  ["#ffffff","#d0e7ff","#5fa8d3","#003f88"])

fig, ax = plt.subplots(figsize=(16, 3))

# === 绘制 mean（第 0 行）
ax.imshow(mean[np.newaxis, :], aspect="equal", cmap=cmap_mean,
          vmin=mean.min(), vmax=mean.max(), extent=[0, len(mean), 2, 4])

# === 绘制 std（第 1 行）
ax.imshow(std[np.newaxis, :], aspect="equal", cmap=cmap_std,
          vmin=std.min(), vmax=std.max(), extent=[0, len(std), 0, 2])

ax.set_xlim(0, len(mean))
ax.set_ylim(0, 4)
ax.set_yticks([3, 1])
ax.set_yticklabels(["mean", "std"])
ax.set_xticks(np.arange(len(mean)) + 0.5)
ax.set_xticklabels(np.arange(len(mean)), rotation=90)

# === 数值居中
for i, v in enumerate(mean):
    ax.text(i+0.5, 3, f"{v:.3f}", ha="center", va="center", fontsize=7)

for i, v in enumerate(std):
    ax.text(i+0.5, 1, f"{v:.3f}", ha="center", va="center", fontsize=7)

# === 两个色条
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax1 = divider.append_axes("right", size="2%", pad=0.3)
cax2 = divider.append_axes("right", size="2%", pad=1.0)

plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_mean), cax=cax1, label="mean concentration (green)")
plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_std),  cax=cax2, label="std concentration (blue)")

ax.set_title("Qwen2.5-7B", pad=20)
plt.tight_layout()
plt.savefig("heatmap_single_axis_uniform_height.png", dpi=300)
print("保存完成 heatmap_single_axis_uniform_height.png")
