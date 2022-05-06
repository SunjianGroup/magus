import matplotlib.pyplot as plt 
from magus.parameters import magusParameters 
import numpy as np
import matplotlib.ticker as ticker


def plot_e(ed, er):
    fig = plt.figure()
    # plt.xticks(fontname="Arial", weight='bold')
    plt.title("MTP energy vs DFT energy", fontsize=16)
    ed = ed - np.mean(ed)
    er = er - np.mean(er)
    ax = plt.gca()
    ax.set_aspect(1)
    xmajorLocator = ticker.MaxNLocator(5)
    ymajorLocator = ticker.MaxNLocator(5)
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.yaxis.set_major_locator(ymajorLocator)
    
    ymajorFormatter = ticker.FormatStrFormatter('%.1f') 
    xmajorFormatter = ticker.FormatStrFormatter('%.1f') 
    ax.xaxis.set_major_formatter(xmajorFormatter)
    ax.yaxis.set_major_formatter(ymajorFormatter)
    
    ax.set_xlabel('DFT energy (eV/atom)', fontsize=14)
    ax.set_ylabel('MTP energy (eV/atom)', fontsize=14)
    
    ax.spines['bottom'].set_linewidth(3);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(3);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(3);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(3);####设置上部坐标轴的粗细
    
    ax.tick_params(labelsize=16)

    
    plt.plot([np.min(ed), np.max(ed)], [np.min(er), np.max(er)],
            color='black',linewidth=3,linestyle='--',)
    plt.scatter(ed, er, zorder=200)
    
    m1 = min(np.min(ed), np.min(er))
    m2 = max(np.max(ed), np.max(er))
    ax.set_xlim(m1, m2)
    ax.set_ylim(m1, m2)

    rmse = np.sqrt(np.mean((ed-er)**2))
    plt.text(np.min(ed) * 0.85 + np.max(ed) * 0.15, 
              np.min(er) * 0.15 + np.max(ed) * 0.85,
              "RMSE: {:.3f} eV/atom".format(rmse), fontsize=14)
    return fig


def plot_f(fd, fr):
    fig = plt.figure()
    ax = plt.gca()
    plt.title("MTP forces vs DFT forces", fontsize=16)
    ax.set_aspect(1)
    xmajorLocator = ticker.MaxNLocator(5)
    ymajorLocator = ticker.MaxNLocator(5)
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.yaxis.set_major_locator(ymajorLocator)
    
    ymajorFormatter = ticker.FormatStrFormatter('%.1f') 
    xmajorFormatter = ticker.FormatStrFormatter('%.1f') 
    ax.xaxis.set_major_formatter(xmajorFormatter)
    ax.yaxis.set_major_formatter(ymajorFormatter)
    
    # ax.set_xlabel('DFT forces (eV/A)', fontsize=14)
    # ax.set_ylabel('MTP forces (eV/A)', fontsize=14)
    
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2);####设置上部坐标轴的粗细

    ax.tick_params(labelsize=14)

    ax.set_xlim(np.min(fd), np.max(fd))
    ax.set_ylim(np.min(fr), np.max(fr))

    plt.plot([np.min(fd), np.max(fd)], [np.min(fr), np.max(fr)],
            color='black',linewidth=2,linestyle='--')
    plt.scatter(fd.reshape(-1), fr.reshape(-1), s=2)

    m1 = min(np.min(fd), np.min(fr))
    m2 = max(np.max(fd), np.max(fr))
    ax.set_xlim(m1, m2)
    ax.set_ylim(m1, m2)

    rmse = np.sqrt(np.mean((fd-fr)**2))
    plt.text(np.min(fd) * 0.85 + np.max(fd) * 0.15, 
              np.min(fr) * 0.15 + np.max(fr) * 0.85,
              "RMSE: {:.3f} eV/A".format(rmse), fontsize=14)
    return fig


para = magusParameters('input.yaml') 
ml = para.MLCalculator
a = ml.trainset
b = ml.calc_efs(a)
e1 = np.array([i.info['energy'] / len(i) for i in a])
e2 = np.array([i.info['energy'] / len(i) for i in b])
f1 = np.concatenate([i.info['forces'].reshape(-1) for i in a])
f2 = np.concatenate([i.info['forces'].reshape(-1) for i in b])
n = np.concatenate([i.get_atomic_numbers() for i in a])
fig = plot_e(e1, e2)
fig.savefig('energy.png')
fig = plot_f(f1, f2)
fig.savefig('forces.png')
np.savez('data.npz', e1=e1, e2=e2, f1=f1, f2=f2, n=n)

