import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects
import seaborn as sns


def plot_scatter(vectors_, y_, show_text=False):
    f, ax = plt.subplots(1, 1, figsize=(30, 30))
    colormap = list(set(y_))
    palette = np.array(sns.color_palette("husl", len(colormap)))  # hls
    color_dict = dict(zip(colormap, palette))
    colors = [color_dict[l] for l in y_]
    positions = []
    texts = []
    colors_legend = []
    old_i = 0
    fontsize = max(5, (30 - len(colormap) / 10))
    if len(y_) > 1 and y_[0] != y_[1]:
        positions.append(np.median(vectors_[0:1], axis=0))
        texts.append(y_[0])
        old_i = 1
    for i in range(1, len(y_)):
        if y_[i] != y_[i - 1] or i == len(y_) - 1:
            if i == len(y_) - 1:
                position = np.median(vectors_[old_i:i + 1], axis=0)
            else:
                position = np.median(vectors_[old_i:i], axis=0)
            # If texts overlap, change their y-position
            # for y_pos in y_positions:
            #     if abs(position[1] - y_pos) > fontsize:
            #         position[1] += abs(position[1] - y_pos) * 1.5 *\
            #             np.sign(position[1] - y_pos)
            #         break
            # y_positions.add(position[1])
            positions.append(position)
            print(old_i, i)
            old_i = i
            if i != len(y_) - 1:
                texts.append(y_[i - 1])
                colors_legend.append(colors[i - 1])
            else:
                texts.append(y_[i])
                colors_legend.append(colors[i])
    if show_text:
        txts = []
        for i in range(len(positions)):
            if len(positions[i]) < 2 or np.isnan(positions[i][0]):
                continue
            # Position of each label.
            xtext, ytext = positions[i]
            txt = ax.text(
                xtext, ytext, texts[i], fontsize=fontsize)
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
            txts.append(txt)
    plt.legend()
    patches = []
    for i, c in enumerate(colors_legend):
        patches.append(mpatches.Patch(color=c, label=texts[i]))
    plt.legend(handles=patches)
    # plt.show()
    plt.savefig(" ".join(colormap)[:20] + ".png")


def pca_scatter_plot(vectors, labels=[], tsne_plot=False, my_plot=False):
    vector_lengths = [len(v) for v in vectors]
    if not labels or len(labels) != len(vector_lengths):
        labels = list(range(len(vectors)))
    labels = [str(l) for l in labels]
    scaler = StandardScaler()
    d2v_scaled = scaler.fit_transform(
        np.concatenate(vectors))
    print("fitting pca")
    d2v_pca = PCA().fit(d2v_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    x_values = range(1, d2v_pca.n_components_ + 1)
    ax.plot(
        x_values,
        d2v_pca.explained_variance_ratio_, lw=2, label='explained variance')
    ax.plot(
        x_values, np.cumsum(d2v_pca.explained_variance_ratio_),
        lw=2, label='cumulative explained variance')
    ax.set_title(
        'Doc2vec '
        '(unigram DBOW + trigram DMM) : explained variance of components')
    ax.set_xlabel('principal component')
    ax.set_ylabel('explained variance')
    plt.xticks(np.arange(0, 150, 10))
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    ax.grid(color='r', linestyle='-', linewidth=0.1)
    plt.show()

    vectors = [v[:30] for v in d2v_pca.transform(d2v_scaled)]
    if tsne_plot:
        print("training tsne")
        vectors = TSNE(n_components=2).fit_transform(vectors)
    else:
        vectors = [v[:2] for v in vectors]
        vectors = np.array(vectors)
    y = []
    for l_i, l in enumerate(labels):
        y += [labels[l_i]] * vector_lengths[l_i]
    if my_plot:
        plot_scatter(vectors, y)
    else:
        print("plotting seaborn")
        df = pd.DataFrame(vectors)
        df["source"] = y
        sns.scatterplot(x=0, y=1, hue="source", data=df)
        plt.show()
