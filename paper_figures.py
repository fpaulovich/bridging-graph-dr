import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math

from techniques.t_sne import TSNE
from techniques.metrics import stress, neighborhood_preservation, neighborhood_hit

from util import load_data, draw_graph_with_positions, draw_projection, draw_graph_no_positions

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score

from metrics.local import sortedness
from scipy.stats import weightedtau

import seaborn as sns
import pandas as pd

from t_sne_bridging_dr_gd import tsne_prob_graph, draw_graph_by_tsne, remove_nodes_centrality, remove_nodes_random
from util import draw_graph_forceatlas2, write_graphml

from techniques.knn_gd import knn_graph
from techniques.umap_gd import umap_graph
from techniques.snn_gd import snn_graph
from techniques.pairwise_gd import pairwise_graph

from umap import UMAP


def fig_1():
    # dir_base_graph = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/survey_dr/tsne/'
    # filename_graph = dir_base_graph + 'fashion_mnist-tsne.graphml'

    dir_output = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/paper_figs/'
    dir_base = '/Users/fpaulovich/Documents/data/'
    dataset = 'fashion_mnist'

    perplexity = 15
    nr_neighbors = 15

    # read the dataset
    X, labels = load_data(dir_base + dataset)
    X = MinMaxScaler().fit_transform(X)
    labels = LabelEncoder().fit_transform(labels)

    ##########################################
    # create fully connected
    g = pairwise_graph(X, metric='euclidean', labels=labels)
    filename = dir_output + dataset + '-pairwise_force_atlas_graph.png'
    pos = draw_graph_forceatlas2(X, g, labels, filename=filename)
    filename = dir_output + dataset + '-pairwise_force_atlas_graph.graphml'
    write_graphml(g, pos, filename)

    ##########################################
    # create the graph SNN
    g = snn_graph(X, nr_neighbors, metric='euclidean', labels=labels)
    filename = dir_output + dataset + '-snn_force_atlas_graph.png'
    pos = draw_graph_forceatlas2(X, g, labels, filename=filename)
    filename = dir_output + dataset + '-snn_force_atlas_graph.graphml'
    write_graphml(g, pos, filename)

    ##########################################
    # create the graph kNN
    g = knn_graph(X, nr_neighbors, metric='euclidean', labels=labels)
    filename = dir_output + dataset + '-knn_force_atlas_graph.png'
    pos = draw_graph_forceatlas2(X, g, labels, filename=filename)
    filename = dir_output + dataset + '-knn_force_atlas_graph.graphml'
    write_graphml(g, pos, filename)

    ##########################################
    # create the graph t-SNE
    g = tsne_prob_graph(X, perplexity, metric='euclidean', labels=labels, epsilon=0.9)

    # t-sne probabilities are too small, increasing, so it is possible to use ForceAtlas
    weight = nx.get_edge_attributes(g, 'weight')
    for key in weight.keys():
        weight[key] = weight[key] * 1000
    nx.set_edge_attributes(g, weight, 'weight')

    filename = dir_output + dataset + '-tsne_force_atlas_graph.png'
    pos = draw_graph_forceatlas2(X, g, labels, filename=filename)
    filename = dir_output + dataset + '-tsne_force_atlas_graph.graphml'
    write_graphml(g, pos, filename)

    ##########################################
    # create the graph t-SNE
    g = umap_graph(X, nr_neighbors, metric='euclidean', labels=labels)
    filename = dir_output + dataset + '-umap_force_atlas_graph.png'
    pos = draw_graph_forceatlas2(X, g, labels, filename=filename)
    filename = dir_output + dataset + '-umap_force_atlas_graph.graphml'
    write_graphml(g, pos, filename)

    return


def fig_2():
    dir_output = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/paper_figs/'
    dir_base = '/Users/fpaulovich/Documents/data/'
    dataset = 'fashion_mnist'

    # color SNN by centrality
    filename = dir_output + dataset + '-snn_force_atlas_graph.graphml'
    g = nx.read_graphml(filename)
    centrality = list(map(float, nx.closeness_centrality(g).values()))
    filename = dir_output + dataset + '-snn_force_atlas_graph_centrality.png'
    draw_graph_with_positions(g, labels=centrality, cmap=plt.cm.cividis_r,
                              color_bar_title='closeness centrality', filename=filename)

    # color KNN by centrality
    filename = dir_output + dataset + '-knn_force_atlas_graph.graphml'
    g = nx.read_graphml(filename)
    centrality = list(map(float, nx.closeness_centrality(g).values()))
    filename = dir_output + dataset + '-knn_force_atlas_graph_centrality.png'
    draw_graph_with_positions(g, labels=centrality, cmap=plt.cm.cividis_r,
                              color_bar_title='closeness centrality', filename=filename)

    # color t-SNE by centrality
    filename = dir_output + dataset + '-tsne_force_atlas_graph.graphml'
    g = nx.read_graphml(filename)
    centrality = list(map(float, nx.closeness_centrality(g).values()))
    filename = dir_output + dataset + '-tsne_force_atlas_graph_centrality.png'
    draw_graph_with_positions(g, labels=centrality, cmap=plt.cm.cividis_r,
                              color_bar_title='closeness centrality', filename=filename)

    # color UMAP by centrality
    filename = dir_output + dataset + '-umap_force_atlas_graph.graphml'
    g = nx.read_graphml(filename)
    centrality = list(map(float, nx.closeness_centrality(g).values()))
    filename = dir_output + dataset + '-umap_force_atlas_graph_centrality.png'
    draw_graph_with_positions(g, labels=centrality, cmap=plt.cm.cividis_r,
                              color_bar_title='closeness centrality', filename=filename)

    return


def fig_3():
    dir_output = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/paper_figs/'
    dir_base = '/Users/fpaulovich/Documents/data/'
    dataset = 'fashion_mnist'

    perplexity = 15
    nr_neighbors = 15

    # read the dataset
    X, labels = load_data(dir_base + dataset)
    X = MinMaxScaler().fit_transform(X)
    labels = LabelEncoder().fit_transform(labels)

    # execute t-SNE
    y = TSNE(n_components=2,
             perplexity=perplexity,
             metric='euclidean',
             random_state=42,
             method='barnes_hut',
             init='pca').fit_transform(X)

    filename = dir_output + dataset + '-tsne.png'
    draw_projection(y, labels, cmap=plt.cm.Set1, color_bar_title=None, filename=filename)

    filename = dir_output + dataset + '-tsne_force_atlas_graph.graphml'
    g = nx.read_graphml(filename)
    centrality = list(map(float, nx.closeness_centrality(g).values()))
    filename = dir_output + dataset + '-tsne_force_atlas_centrality.png'
    draw_projection(y, centrality, cmap=plt.cm.cividis_r,
                    color_bar_title='closeness centrality', filename=filename)

    # execute UMAP
    y = UMAP(n_neighbors=nr_neighbors,
             metric='euclidean',
             random_state=42).fit_transform(X)

    filename = dir_output + dataset + '-umap.png'
    draw_projection(y, labels, cmap=plt.cm.Set1, color_bar_title=None, filename=filename)

    filename = dir_output + dataset + '-umap_force_atlas_graph.graphml'
    g = nx.read_graphml(filename)
    centrality = list(map(float, nx.closeness_centrality(g).values()))
    filename = dir_output + dataset + '-umap_force_atlas_centrality.png'
    draw_projection(y, centrality, cmap=plt.cm.cividis_r,
                    color_bar_title='closeness centrality', filename=filename)

    return


def fig_4():
    dir_base_graph = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/paper_figs/'
    filename_graph = dir_base_graph + 'fashion_mnist-tsne_force_atlas_graph.graphml'

    dir_base = '/Users/fpaulovich/Documents/data/'
    dataset = 'fashion_mnist'

    perplexity = 15

    percentages = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]

    metrics_df = pd.DataFrame(columns=['percentage',
                                       'sortedness',
                                       'sortedness_weightedtau',
                                       'trustworthiness',
                                       'stress',
                                       'silhouette_score',
                                       'neighborhood_preservation',
                                       'neighborhood_hit'])

    # read the dataset
    X, label = load_data(dir_base + dataset)
    X = MinMaxScaler().fit_transform(X)

    # read the graph
    g = nx.read_graphml(filename_graph)

    # draw using t-sne
    y = TSNE(n_components=2,
             perplexity=perplexity,
             metric='euclidean',
             random_state=42,
             method='barnes_hut',
             init='pca').fit_transform(X)

    # update graph coordinates
    for node in g.nodes:
        g.nodes[node]['x'] = float(y[int(node)][0])
        g.nodes[node]['y'] = float(y[int(node)][1])

    # draw graph
    filename_fig = dir_base_graph + 'reduced/' + dataset + '[1.00]-reduced_graph_tsne.png'
    draw_graph_with_positions(g, filename=filename_fig)

    centrality = list(map(float, nx.closeness_centrality(g).values()))
    filename_fig = dir_base_graph + 'reduced/' + dataset + '[1.00]-reduced_graph_centrality_tsne.png'
    draw_graph_with_positions(g, labels=centrality, cmap=plt.cm.YlGnBu, color_bar_title='closeness centrality',
                              filename=filename_fig)

    # # draw the graph using t-SNE
    y, label = draw_graph_by_tsne(X, g)

    # save projection image
    filename_fig = dir_base_graph + 'reduced/' + dataset + '[1.00]-reduced_tsne.png'
    draw_projection(y, label, filename=filename_fig)

    filename_fig = dir_base_graph + 'reduced/' + dataset + '[1.00]-reduced_centrality_tsne.png'
    draw_projection(y, centrality, cmap=plt.cm.YlGnBu, color_bar_title='closeness centrality',
                    filename=filename_fig)

    # calculate metrics for the original
    print('---')
    print('>>>original')
    metrics = {}

    dr_metric = sortedness(X, y)
    print('sortedness:', np.average(dr_metric))
    metrics.update({'sortedness': np.average(dr_metric)})

    dr_metric = sortedness(X, y, f=weightedtau)
    print('sortedness (weightedtau):', np.average(dr_metric))
    metrics.update({'sortedness_weightedtau': np.average(dr_metric)})

    dr_metric = trustworthiness(X, y, n_neighbors=7)
    print('trustworthiness:', dr_metric)
    metrics.update({'trustworthiness': np.average(dr_metric)})
    dr_metric = stress(X, y, metric='euclidean')

    print('stress:', dr_metric)
    metrics.update({'stress': np.average(dr_metric)})
    dr_metric = silhouette_score(y, label)

    print('silhouette_score:', dr_metric)
    metrics.update({'silhouette_score': np.average(dr_metric)})
    dr_metric = neighborhood_preservation(X, y, nr_neighbors=7)

    print('neighborhood_preservation:', dr_metric)
    metrics.update({'neighborhood_preservation': np.average(dr_metric)})

    dr_metric = neighborhood_hit(y, label, nr_neighbors=7)
    print('neighborhood_hit:', dr_metric)
    metrics.update({'neighborhood_hit': np.average(dr_metric)})

    metrics_df.loc[len(metrics_df)] = [1.00,
                                       metrics['sortedness'],
                                       metrics['sortedness_weightedtau'],
                                       metrics['trustworthiness'],
                                       metrics['stress'],
                                       metrics['silhouette_score'],
                                       metrics['neighborhood_preservation'],
                                       metrics['neighborhood_hit']]

    for percentage in percentages:
        print('--')
        print('>> percentage: ', percentage)

        # read the graph
        g = nx.read_graphml(filename_graph)

        # remove nodes by centrality
        y_removed, label_removed, g_removed, metrics = remove_nodes_centrality(X,
                                                                               label,
                                                                               g,
                                                                               perplexity,
                                                                               nodes_to_keep=percentage)

        # save reduced graph
        filename_graph_reduced = dir_base_graph + 'reduced/' + dataset + '[' + str(
            percentage) + ']-reduced_tsne.graphml'
        nx.write_graphml(g_removed, filename_graph_reduced, named_key_ids=True)

        # draw reduced graph
        filename_fig_reduced = dir_base_graph + 'reduced/' + dataset + '[' + str(
            percentage) + ']-reduced_graph_tsne.png'
        draw_graph_with_positions(g_removed, filename=filename_fig_reduced)

        centrality = list(map(float, nx.closeness_centrality(g_removed).values()))
        filename_fig_reduced = dir_base_graph + 'reduced/' + dataset + '[' + str(
            percentage) + ']-reduced_graph_centrality_tsne.png'
        draw_graph_with_positions(g_removed, labels=centrality, cmap=plt.cm.cividis_r,
                                  color_bar_title='closeness centrality', filename=filename_fig_reduced)

        metrics_df.loc[len(metrics_df)] = [percentage,
                                           metrics['sortedness'],
                                           metrics['sortedness_weightedtau'],
                                           metrics['trustworthiness'],
                                           metrics['stress'],
                                           metrics['silhouette_score'],
                                           metrics['neighborhood_preservation'],
                                           metrics['neighborhood_hit']]

        # save image
        filename_fig_reduced = dir_base_graph + 'reduced/' + dataset + '[' + \
                               str(percentage) + ']-reduced_tsne.png'
        draw_projection(y_removed, label_removed, filename=filename_fig_reduced)

        filename_fig_reduced = dir_base_graph + 'reduced/' + dataset + '[' + \
                               str(percentage) + ']-reduced_centrality_tsne.png'
        draw_projection(y_removed, centrality, cmap=plt.cm.cividis_r,
                        color_bar_title='closeness centrality', filename=filename_fig_reduced)

    # save CSV
    metrics_df.to_csv(dir_base_graph + 'reduced/' + dataset + '-metrics.csv', sep=',')

    return


def fig_5():
    dir_base_graph = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/paper_figs/'

    metrics = ['stress', 'trustworthiness', 'neighborhood_preservation', 'neighborhood_hit',
               'silhouette_score']
    dataset = 'fashion_mnist'

    filename_metrics = dir_base_graph + 'reduced/' + dataset + '-metrics.csv'
    df_metrics = pd.read_csv(filename_metrics, sep=',')

    # invert stress
    for i in range(len(df_metrics)):
        df_metrics.loc[i, 'stress'] = 1.0 - df_metrics.loc[i]['stress']

    df_metrics_original = df_metrics.copy()

    # normalize between zero and one
    for metric in metrics:
        min_val = df_metrics.min(axis=0)[metric]
        df_metrics[metric] = df_metrics[metric].sub(min_val)

        max_val = df_metrics.max(axis=0)[metric]
        df_metrics[metric] = df_metrics[metric].div(max_val)

    new_df_metrics = pd.DataFrame(columns=['percentage',
                                           'metric',
                                           'score'])

    new_df_metrics_original = pd.DataFrame(columns=['percentage',
                                                    'metric',
                                                    'score'])

    for i in range(len(df_metrics)):
        for metric in metrics:
            perc = int(round(((1.0 - df_metrics_original.loc[i]['percentage']) * 100), 0))
            new_df_metrics_original.loc[len(new_df_metrics_original)] = [f'{perc:02}%',
                                                                         metric,
                                                                         round(
                                                                             float(df_metrics_original.loc[i][metric]),
                                                                             4)]

            perc = int(round(((1.0 - df_metrics.loc[i]['percentage']) * 100), 0))
            new_df_metrics.loc[len(new_df_metrics)] = [f'{perc:02}%',
                                                       metric,
                                                       round(float(df_metrics.loc[i][metric]), 4)]

    new_df_metrics = new_df_metrics.pivot(index="percentage", columns="metric", values="score")
    new_df_metrics_original = new_df_metrics_original.pivot(index="percentage", columns="metric", values="score")

    # color_map = plt.cm.get_cmap('cividis').reversed()

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(new_df_metrics, cbar=False,
                robust=True, annot_kws={"size": 25}, fmt=".4f",
                linewidths=2, linecolor='white', annot=new_df_metrics_original,
                cbar_kws={'orientation': 'vertical'}, cmap=plt.cm.YlGnBu,
                xticklabels=['N.Hit', 'N.Preservation', 'Silhouette', ' 1-Stress', 'Trustworthiness']
                )

    plt.yticks(rotation=0, fontsize=25)
    plt.xticks(rotation=25, fontsize=25, ha='right')
    ax.set(xlabel="", ylabel="")
    # plt.xticks(rotation=20, fontsize=20, ha='right')
    # plt.title(title, fontsize=30)

    filename_fig = dir_base_graph + 'reduced/heatmap_' + dataset + '.png'
    plt.savefig(filename_fig, dpi=300, bbox_inches='tight')
    # plt.show()

    return


def faithfulness_graph_topologies():
    filename_knn = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/paper_figs/' \
                   'fashion_mnist-knn_force_atlas_graph.graphml'
    filename_tsne = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/paper_figs/' \
                    'fashion_mnist-tsne_force_atlas_graph.graphml'
    filename_umap = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/paper_figs/' \
                    'fashion_mnist-umap_force_atlas_graph.graphml'

    g_umap = nx.read_graphml(filename_umap)
    g_tsne = nx.read_graphml(filename_tsne)
    g_knn = nx.read_graphml(filename_knn)

    print("jaccard index (tsne, umap): ",
          len(nx.intersection(g_tsne, g_umap).edges) / len(nx.compose(g_tsne, g_umap).edges))
    print("jaccard index (tsne, knn): ",
          len(nx.intersection(g_tsne, g_knn).edges) / len(nx.compose(g_tsne, g_knn).edges))
    print("jaccard index (umap, knn): ",
          len(nx.intersection(g_umap, g_knn).edges) / len(nx.compose(g_umap, g_knn).edges))

    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # fig_1()
    # fig_2()
    # fig_3()
    # fig_4()
    fig_5()
    # faithfulness_graph_topologies()
