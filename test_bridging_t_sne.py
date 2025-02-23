import math

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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
from util import draw_graph, write_graphml

MACHINE_EPSILON = np.finfo(np.double).eps

datasets = ['bank', 'cifar10', 'cnae9', 'coil20', 'epileptic', 'fashion_mnist', 'fmd', 'har', 'hatespeech',
            'hiva', 'imdb', 'secom', 'seismic', 'sentiment', 'sms', 'spambase', 'svhn']

perplexities = {'bank': 30, 'cifar10': 15, 'cnae9': 5, 'coil20': 50, 'epileptic': 50, 'fashion_mnist': 50,
                'fmd': 50, 'har': 30, 'hatespeech': 30, 'hiva': 50, 'imdb': 50, 'orl': 15, 'secom': 30, 'seismic': 50,
                'sentiment': 15, 'sms': 50, 'spambase': 5, 'svhn': 15}


def run_generate_all_tsne_graphs():
    dir_base = '/Users/fpaulovich/Documents/data/'
    dir_name_output = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/survey_dr/tsne/'

    for dataset in datasets:
        print('>>>processing:', dataset)
        X, y = load_data(dir_base + dataset)
        X = MinMaxScaler().fit_transform(X)
        y = LabelEncoder().fit_transform(y)

        metric = 'euclidean'
        perplexity = int(perplexities[dataset])
        g = tsne_prob_graph(X, perplexity, metric, y, epsilon=0.9)

        filename_fig = dir_name_output + dataset + '-gd_tsne.png'
        filename_graph = dir_name_output + dataset + '-tsne.graphml'

        pos = draw_graph(X, g, y, filename_fig)
        write_graphml(g, pos, filename_graph)

    return


def run_draw_all_graphs_by_tsne():
    dir_data_base = '/Users/fpaulovich/Documents/data/'
    dir_base = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/survey_dr/tsne/'

    for dataset in datasets:
        print('>>>processing:', dataset)
        X, _ = load_data(dir_data_base + dataset)
        X = MinMaxScaler().fit_transform(X)

        filename_graph = dir_base + dataset + '-tsne.graphml'
        filename_fig = dir_base + dataset + '-tsne.png'
        g = nx.read_graphml(filename_graph)
        y, label = draw_graph_by_tsne(X, g)

        plt.figure()
        plt.scatter(y[:, 0], y[:, 1], c=label, cmap='Set1', edgecolors='face', linewidths=0.5, s=4)
        plt.savefig(filename_fig, dpi=400, bbox_inches='tight')
        plt.close()

    return


def run_calculate_metrics_original_techniques():
    dir_base = '/Users/fpaulovich/Documents/data/'

    for dataset in datasets:
        print('>>>processing:', dataset)
        X, label = load_data(dir_base + dataset)
        X = MinMaxScaler().fit_transform(X)
        label = LabelEncoder().fit_transform(label)

        y = TSNE(n_components=2,
                 perplexity=int(perplexities[dataset]),
                 metric='euclidean',
                 random_state=42,
                 method='barnes_hut',
                 init='pca').fit_transform(X)

        dr_metric = sortedness(X, y)
        print('sortedness:', np.average(dr_metric))
        dr_metric = sortedness(X, y, f=weightedtau)
        print('sortedness (weightedtau):', np.average(dr_metric))
        dr_metric = trustworthiness(X, y, n_neighbors=7)
        print('trustworthiness:', dr_metric)
        dr_metric = stress(X, y, metric='euclidean')
        print('stress:', dr_metric)
        dr_metric = silhouette_score(y, label)
        print('silhouette_score:', dr_metric)
        dr_metric = neighborhood_preservation(X, y, nr_neighbors=7)
        print('neighborhood_preservation:', dr_metric)
        dr_metric = neighborhood_hit(y, label, nr_neighbors=7)
        print('neighborhood_hit:', dr_metric)

    return


def run_remove_nodes_centrality():
    dir_base_graph = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/survey_dr/tsne/'
    filename_graph = dir_base_graph + 'fashion_mnist-tsne.graphml'

    dir_base = '/Users/fpaulovich/Documents/data/'
    dataset = 'fashion_mnist'

    perplexity = 50

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
    X, _ = load_data(dir_base + dataset)
    X = MinMaxScaler().fit_transform(X)

    # read the graph
    g = nx.read_graphml(filename_graph)

    # draw graph
    filename_fig = dir_base_graph + 'reduced/' + dataset + '[1.00]-reduced_graph_tsne.png'
    draw_graph_with_positions(g, filename_fig)

    # draw the graph using t-SNE
    y, label = draw_graph_by_tsne(X, g)

    # save image
    filename_fig = dir_base_graph + 'reduced/' + dataset + '[1.00]-reduced_tsne.png'
    draw_projection(y, label, filename=filename_fig)

    # calculate metrics for the original
    print('---')
    print('>>>original')
    metrics = {}

    dr_metric = sortedness(X, y)
    print('sortedness after:', np.average(dr_metric))
    metrics.update({'sortedness': np.average(dr_metric)})

    dr_metric = sortedness(X, y, f=weightedtau)
    print('sortedness after (weightedtau):', np.average(dr_metric))
    metrics.update({'sortedness_weightedtau': np.average(dr_metric)})

    dr_metric = trustworthiness(X, y, n_neighbors=7)
    print('trustworthiness after:', dr_metric)
    metrics.update({'trustworthiness': np.average(dr_metric)})
    dr_metric = stress(X, y, metric='euclidean')

    print('stress after:', dr_metric)
    metrics.update({'stress': np.average(dr_metric)})
    dr_metric = silhouette_score(y, label)

    print('silhouette_score after:', dr_metric)
    metrics.update({'silhouette_score': np.average(dr_metric)})
    dr_metric = neighborhood_preservation(X, y, nr_neighbors=7)

    print('neighborhood_preservation after:', dr_metric)
    metrics.update({'neighborhood_preservation': np.average(dr_metric)})

    dr_metric = neighborhood_hit(y, label, nr_neighbors=7)
    print('neighborhood_hit after:', dr_metric)
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
        draw_graph_with_positions(g_removed, filename_fig_reduced)

        metrics_df.loc[len(metrics_df)] = [percentage,
                                           metrics['sortedness'],
                                           metrics['sortedness_weightedtau'],
                                           metrics['trustworthiness'],
                                           metrics['stress'],
                                           metrics['silhouette_score'],
                                           metrics['neighborhood_preservation'],
                                           metrics['neighborhood_hit']]

        # save image
        filename_fig_reduced = dir_base_graph + 'reduced/' + dataset + '[' + str(percentage) + ']-reduced_tsne.png'
        draw_projection(y_removed, label_removed, filename=filename_fig_reduced)

    # save CSV
    metrics_df.to_csv(dir_base_graph + dataset + '-metrics.csv', sep=',')

    return


def run_remove_nodes_centrality_batch():
    dir_base_dataset = '/Users/fpaulovich/Documents/data/'
    dir_base_graph = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/survey_dr/tsne/'

    datasets = ['cnae9', 'coil20', 'fashion_mnist', 'har', 'spambase']

    percentages = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]

    for dataset in datasets:
        print('--')
        print('--')
        print('>>>processing:', dataset)
        print('--')

        metrics_df = pd.DataFrame(columns=['percentage',
                                           'sortedness',
                                           'sortedness_weightedtau',
                                           'trustworthiness',
                                           'stress',
                                           'silhouette_score',
                                           'neighborhood_preservation',
                                           'neighborhood_hit'])

        # read the dataset
        X, _ = load_data(dir_base_dataset + dataset)
        X = MinMaxScaler().fit_transform(X)

        # read the graph
        filename_graph = dir_base_graph + dataset + '-tsne.graphml'
        g = nx.read_graphml(filename_graph)

        # draw the graph using t-SNE
        y, label = draw_graph_by_tsne(X, g)

        # draw graph
        filename_fig = dir_base_graph + 'reduced/' + dataset + '[1.00]-reduced_graph_tsne.png'
        draw_graph_no_positions(g, y, label, filename=filename_fig)

        # save image
        filename_fig = dir_base_graph + 'reduced/' + dataset + '[1.00]-reduced_tsne.png'
        draw_projection(y, label, filename=filename_fig)

        # calculate metrics for the original
        print('---')
        print('>>>original')
        metrics = {}

        dr_metric = sortedness(X, y)
        print('sortedness after:', np.average(dr_metric))
        metrics.update({'sortedness': np.average(dr_metric)})

        dr_metric = sortedness(X, y, f=weightedtau)
        print('sortedness after (weightedtau):', np.average(dr_metric))
        metrics.update({'sortedness_weightedtau': np.average(dr_metric)})

        dr_metric = trustworthiness(X, y, n_neighbors=7)
        print('trustworthiness after:', dr_metric)
        metrics.update({'trustworthiness': np.average(dr_metric)})
        dr_metric = stress(X, y, metric='euclidean')

        print('stress after:', dr_metric)
        metrics.update({'stress': np.average(dr_metric)})
        dr_metric = silhouette_score(y, label)

        print('silhouette_score after:', dr_metric)
        metrics.update({'silhouette_score': np.average(dr_metric)})
        dr_metric = neighborhood_preservation(X, y, nr_neighbors=7)

        print('neighborhood_preservation after:', dr_metric)
        metrics.update({'neighborhood_preservation': np.average(dr_metric)})

        dr_metric = neighborhood_hit(y, label, nr_neighbors=7)
        print('neighborhood_hit after:', dr_metric)
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
            print('>> percentage: ', percentage)

            # read the graph
            filename_graph = dir_base_graph + dataset + '-tsne.graphml'
            g = nx.read_graphml(filename_graph)

            # remove nodes by centrality
            y_removed, label_removed, g_removed, metrics = remove_nodes_centrality(X,
                                                                                   label,
                                                                                   g,
                                                                                   int(perplexities[dataset]),
                                                                                   nodes_to_keep=percentage)
            # save reduced graph
            filename_graph_reduced = dir_base_graph + 'reduced/' + dataset + '[' + \
                                     str(percentage) + ']-reduced_tsne.graphml'
            nx.write_graphml(g_removed, filename_graph_reduced, named_key_ids=True)

            # draw reduced graph
            filename_fig_reduced = dir_base_graph + 'reduced/' + dataset + '[' + \
                                   str(percentage) + ']-reduced_graph_tsne.png'
            draw_graph_with_positions(g_removed, filename_fig_reduced)

            metrics_df.loc[len(metrics_df)] = [percentage,
                                               metrics['sortedness'],
                                               metrics['sortedness_weightedtau'],
                                               metrics['trustworthiness'],
                                               metrics['stress'],
                                               metrics['silhouette_score'],
                                               metrics['neighborhood_preservation'],
                                               metrics['neighborhood_hit']]

            # save image
            filename_fig_reduced = dir_base_graph + 'reduced/' + dataset + '[' + str(percentage) + ']-reduced_tsne.png'
            draw_projection(y_removed, label_removed, filename=filename_fig_reduced)

        filename_metrics = dir_base_graph + 'reduced/' + dataset + '-metrics.csv'
        metrics_df.to_csv(filename_metrics, sep=',')

    return


def run_remove_nodes_random_batch():
    dir_base_dataset = '/Users/fpaulovich/Documents/data/'
    dir_base_graph = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/survey_dr/tsne/'

    datasets = ['cnae9', 'coil20', 'fashion_mnist', 'har', 'spambase']

    percentages = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]

    for dataset in datasets:
        print('--')
        print('--')
        print('>>>processing:', dataset)
        print('--')

        metrics_df = pd.DataFrame(columns=['percentage',
                                           'sortedness',
                                           'sortedness_weightedtau',
                                           'trustworthiness',
                                           'stress',
                                           'silhouette_score',
                                           'neighborhood_preservation',
                                           'neighborhood_hit'])

        # read the dataset
        X, _ = load_data(dir_base_dataset + dataset)
        X = MinMaxScaler().fit_transform(X)

        # read the graph
        filename_graph = dir_base_graph + dataset + '-tsne.graphml'
        g = nx.read_graphml(filename_graph)

        # draw the graph using t-SNE
        y, label = draw_graph_by_tsne(X, g)

        # draw graph
        filename_fig = dir_base_graph + 'reduced_random/' + dataset + '[1.00]-reduced_graph_tsne.png'
        draw_graph_no_positions(g, y, label, filename=filename_fig)

        # save image
        filename_fig = dir_base_graph + 'reduced_random/' + dataset + '[1.00]-reduced_tsne.png'
        draw_projection(y, label, filename=filename_fig)

        # calculate metrics for the original
        print('---')
        print('>>>original')
        metrics = {}

        dr_metric = sortedness(X, y)
        print('sortedness after:', np.average(dr_metric))
        metrics.update({'sortedness': np.average(dr_metric)})

        dr_metric = sortedness(X, y, f=weightedtau)
        print('sortedness after (weightedtau):', np.average(dr_metric))
        metrics.update({'sortedness_weightedtau': np.average(dr_metric)})

        dr_metric = trustworthiness(X, y, n_neighbors=7)
        print('trustworthiness after:', dr_metric)
        metrics.update({'trustworthiness': np.average(dr_metric)})
        dr_metric = stress(X, y, metric='euclidean')

        print('stress after:', dr_metric)
        metrics.update({'stress': np.average(dr_metric)})
        dr_metric = silhouette_score(y, label)

        print('silhouette_score after:', dr_metric)
        metrics.update({'silhouette_score': np.average(dr_metric)})
        dr_metric = neighborhood_preservation(X, y, nr_neighbors=7)

        print('neighborhood_preservation after:', dr_metric)
        metrics.update({'neighborhood_preservation': np.average(dr_metric)})

        dr_metric = neighborhood_hit(y, label, nr_neighbors=7)
        print('neighborhood_hit after:', dr_metric)
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
            print('>> percentage: ', percentage)

            # read the graph
            filename_graph = dir_base_graph + dataset + '-tsne.graphml'
            g = nx.read_graphml(filename_graph)

            # remove nodes randomly
            y_removed, label_removed, g_removed, metrics = remove_nodes_random(X,
                                                                               label,
                                                                               g,
                                                                               int(perplexities[dataset]),
                                                                               nodes_to_keep=percentage)

            # save reduced graph
            filename_graph_reduced = dir_base_graph + 'reduced_random/' + dataset + '[' + \
                                     str(percentage) + ']-reduced_tsne.graphml'
            nx.write_graphml(g_removed, filename_graph_reduced, named_key_ids=True)

            # draw reduced graph
            filename_fig_reduced = dir_base_graph + 'reduced_random/' + dataset + '[' + \
                                   str(percentage) + ']-reduced_graph_tsne.png'
            draw_graph_with_positions(g_removed, filename_fig_reduced)

            metrics_df.loc[len(metrics_df)] = [percentage,
                                               metrics['sortedness'],
                                               metrics['sortedness_weightedtau'],
                                               metrics['trustworthiness'],
                                               metrics['stress'],
                                               metrics['silhouette_score'],
                                               metrics['neighborhood_preservation'],
                                               metrics['neighborhood_hit']]

            # save image
            filename_fig_reduced = dir_base_graph + 'reduced_random/' + dataset + '[' + \
                                   str(percentage) + ']-reduced_tsne.png'
            draw_projection(y_removed, label_removed, filename=filename_fig_reduced)

        filename_metrics = dir_base_graph + 'reduced_random/' + dataset + '-metrics.csv'
        metrics_df.to_csv(filename_metrics, sep=',')

    return


def draw_line_graph():
    metrics = ['sortedness', 'sortedness_weightedtau', 'trustworthiness', 'stress',
               'silhouette_score', 'neighborhood_preservation', 'neighborhood_hit']

    datasets = ['fashion_mnist']  # ['cnae9', 'coil20', 'fashion_mnist', 'har', 'spambase']

    dir_base_graph = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/survey_dr/tsne/'

    plt.figure()
    for metric in metrics:
        for dataset in datasets:
            filename_metrics = dir_base_graph + 'reduced_random/' + dataset + '-metrics.csv'
            df_metrics = pd.read_csv(filename_metrics, sep=',')
            plt.plot(df_metrics['percentage'], df_metrics[metric], label=dataset)

        plt.xlabel('percentage')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.ylabel(metric)
        plt.gca().invert_xaxis()

        filename_fig = dir_base_graph + 'reduced_random/metric_' + metric + '.png'
        plt.savefig(filename_fig, dpi=300, bbox_inches='tight')
        plt.close()


def heatmap():
    # metrics = ['sortedness', 'sortedness_weightedtau', 'trustworthiness', 'stress',
    #            'silhouette_score', 'neighborhood_preservation', 'neighborhood_hit']

    metrics = ['trustworthiness', 'silhouette_score', 'neighborhood_preservation', 'neighborhood_hit']

    datasets = ['fashion_mnist', 'cnae9', 'coil20', 'fashion_mnist', 'har', 'spambase']

    dir_base_graph = '/Users/fpaulovich/OneDrive - TU Eindhoven/Dropbox/papers/2024/bridging_dr_graph/survey_dr/tsne/'

    for dataset in datasets:
        filename_metrics = dir_base_graph + 'reduced/' + dataset + '-metrics.csv'
        df_metrics = pd.read_csv(filename_metrics, sep=',')
        df_metrics_original = df_metrics.copy()

        new_df_metrics = pd.DataFrame(columns=['percentage',
                                               'metric',
                                               'score'])

        new_df_metrics_original = pd.DataFrame(columns=['percentage',
                                                        'metric',
                                                        'score'])

        for i in range(len(df_metrics)):
            for metric in metrics:
                new_df_metrics_original.loc[len(new_df_metrics_original)] = [df_metrics_original.loc[i]['percentage'],
                                                                             metric,
                                                                             float(df_metrics_original.loc[i][metric])]

                min_val = df_metrics.min(axis=0)[metric]
                max_val = df_metrics.max(axis=0)[metric]

                df_metrics[metric] = df_metrics[metric].sub(min_val)
                df_metrics[metric] = df_metrics[metric].div(max_val)

                new_df_metrics.loc[len(new_df_metrics)] = [df_metrics.loc[i]['percentage'],
                                                           metric,
                                                           float(df_metrics.loc[i][metric])]

        new_df_metrics = new_df_metrics.pivot(index="metric", columns="percentage", values="score")
        new_df_metrics_original = new_df_metrics_original.pivot(index="metric", columns="percentage", values="score")

        color_map = plt.cm.get_cmap('cividis')  # .reversed()

        fig, ax = plt.subplots(figsize=(16, 14))
        sns.heatmap(new_df_metrics, cbar=False,
                    robust=True, annot_kws={"size": 25}, fmt=".3f",
                    linewidths=2, linecolor='white', annot=new_df_metrics_original,
                    cbar_kws={'orientation': 'vertical'}, cmap=color_map,
                    yticklabels=['N.Hit', 'N.Preservation', 'Silhouette', 'Trustworthiness'])

        plt.yticks(rotation=0, fontsize=25)
        plt.xticks(rotation=0, fontsize=25)
        ax.set(xlabel="", ylabel="")
        # plt.xticks(rotation=20, fontsize=20, ha='right')
        # plt.title(title, fontsize=30)

        filename_fig = dir_base_graph + 'reduced/heatmap_' + dataset + '.png'
        plt.savefig(filename_fig, dpi=300, bbox_inches='tight')
        # plt.show()

    return


if __name__ == '__main__':
    # run_generate_all_tsne_graphs()
    # run_draw_all_graphs_by_tsne()

    # run_remove_nodes_centrality_batch()
    # run_remove_nodes_random_batch()

    # run_remove_nodes_centrality()
    # draw_line_graph()

    heatmap()
