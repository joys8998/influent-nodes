import networkx as nx
import pandas as pd
import itertools
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")


class Network_analysis:
    def __init__(self, graph, nodes):
        self.graph = graph
        self.nodes = nodes
        self.combinations_nodes = list(itertools.combinations(self.nodes, 2))
        self.degree_c = pd.DataFrame({'Node': self.nodes,
                                 'DegreeCentrality': list(nx.degree_centrality(self.graph).values())},
                                index=self.nodes)

        self.betweenness_c = pd.DataFrame({'Node': self.nodes,
                                      'BetweennessCentrality': list(nx.betweenness_centrality(self.graph).values())},
                                     index=self.nodes)

        self.closeness_c = pd.DataFrame({'Node': self.nodes,
                                    'ClosenessCentrality': list(nx.closeness_centrality(self.graph).values())},
                                   index=self.nodes)

        self.eigenvector_c = pd.DataFrame({'Node': self.nodes,
                                      'EigenvectorCentrality': list(nx.eigenvector_centrality(self.graph, max_iter=600).values())},
                                     index=self.nodes)

        self.current_flow_betweenness_c = pd.DataFrame({'Node': self.nodes,
                                                   'CFBetweennessCentrality':
                                                    list(nx.current_flow_betweenness_centrality(self.graph).values())},
                                                  index=self.nodes)

        self.reach = [nx.local_reaching_centrality(self.graph, x) *1.9 for x in self.graph]
        self.reachability = pd.DataFrame({'Node': self.nodes,
                                                'Reachability': list(self.reach)},
                                                index=self.nodes)

    # NORMALIZZAZIONE MIN MAX
    @staticmethod
    def min_max_normalization(centrality_df):
        feature = list(centrality_df.columns)[1]
        df_min = centrality_df[feature].min()
        new_pd = []
        denom = centrality_df[feature].max() - df_min
        header = list(centrality_df.columns)
        for idx, row in centrality_df.iterrows():
            norm_val = (row[feature] - df_min) / denom
            new_pd.append([row['Node'], norm_val])

        return pd.DataFrame(new_pd, columns=header)

    def create_preference_relation(self, centrality_dataframe):
        preference = []
        header = ['N1', 'N2', 'PR']
        for i in self.combinations_nodes:
            if centrality_dataframe.loc[i[0]][list(centrality_dataframe.columns)[1]] >= centrality_dataframe.loc[i[1]][
                list(centrality_dataframe.columns)[1]]:

                row = [int(centrality_dataframe.loc[i[0]]['Node']),
                       int(centrality_dataframe.loc[i[1]]['Node']),
                       (centrality_dataframe.loc[i[0]][list(centrality_dataframe.columns)[1]] -
                        centrality_dataframe.loc[i[1]][list(centrality_dataframe.columns)[1]])]
            else:
                row = [int(centrality_dataframe.loc[i[1]]['Node']),
                       int(centrality_dataframe.loc[i[0]]['Node']),
                       (centrality_dataframe.loc[i[1]][list(centrality_dataframe.columns)[1]] -
                        centrality_dataframe.loc[i[0]][list(centrality_dataframe.columns)[1]])]
            preference.append(row)
        df = pd.DataFrame(preference, columns=header)
        del preference, header
        return df

    def create_PPG(self, pr_df):
        PPG = nx.DiGraph()
        PPG.add_nodes_from(self.nodes)
        for idx, row in pr_df.iterrows():
            if row['PR'] > 0:
                PPG.add_edge(row['N1'], row['N2'], weight=row['PR'])

        return PPG

    @staticmethod
    def get_diff_pd(cpg):
        print('[INFO] EXECUTING PARTIAL PREFERENCE')
        features = ['EC_N1', 'CFBC_N1', 'CFCC_N1']
        f_comb = [x for x in sorted(list(set(list(itertools.combinations(features, 2)))))]
        t = []
        for comb in f_comb:
            diff = ''.join(list(set(features) - set([comb[0], comb[1]])))
            item = cpg[(cpg[comb[0]] == cpg[comb[1]]) & (cpg[comb[0]] != cpg[diff])]
            prefix0 = comb[0][:-2]
            prefix1 = comb[1][:-2]
            item[prefix0 + 'PR'] = item[[prefix0 + 'PR', prefix1 + 'PR']].sum(axis=1)
            item = item.drop([prefix1 + 'N2', prefix1 + 'N1', prefix1 + 'PR'], axis=1)
            rest = pd.DataFrame(item[[diff, diff[:-2] + 'N2', diff[:-2] + 'PR']]).rename({
                diff: comb[0], diff[:-2] + 'N2': prefix0 + 'N2', diff[:-2] + 'PR': prefix0 + 'PR'}, axis=1)
            item = item.drop([diff, diff[:-2] + 'N2', diff[:-2] + 'PR'], axis=1)
            item = pd.concat([item, rest], ignore_index=True)
            item.columns = ['N1', 'N2', 'PR']
            del rest

            t.append(item)
            del item
        return t

    @staticmethod
    def rank_walk(t, vector, adj_matrix):
        for i in range(t):
            vector = adj_matrix * vector
        return list(vector)

    @staticmethod
    def cpg_rank(pr_cfbc, pr_cfcc, pr_ec):
        # CHECK
        if any(pr_cfbc['PR'] < 0):
            print('[INFO] PREFERENCE RELATION ERROR IN CURRENT FLOW BETWEENNESS CENTRALITY DATAFRAME')
            return
        if any(pr_cfcc['PR'] < 0):
            print('[INFO] PREFERENCE RELATION ERROR IN CURRENT FLOW CLOSENESS CENTRALITY DATAFRAME')
            return
        if any(pr_ec['PR'] < 0):
            print('[INFO] PREFERENCE RELATION ERROR IN EIGENVECTOR CENTRALITY DATAFRAME')
            return

        # CREO UN UNICO DATAFRAME
        cpg = pd.concat([pr_ec, pr_cfbc, pr_cfcc], axis=1)
        header = ['EC_N1', 'EC_N2', 'EC_PR', 'CFBC_N1', 'CFBC_N2', 'CFBC_PR', 'CFCC_N1', 'CFCC_N2', 'CFCC_PR']
        cpg.columns = header

        cfbc_cfcc, ec_cfbc, ec_cfcc = Network_analysis.get_diff_pd(cpg)
        print('[INFO] EXECUTING CPG AND RANKING')
        total = cpg[(cpg['EC_N1'] == cpg['CFBC_N1']) & (cpg['EC_N1'] == cpg['CFCC_N1'])]

        total['total'] = total['EC_PR'] + total['CFBC_PR'] + total['CFCC_PR']
        # total['total'].fillna(0, inplace=True)
        total = total.drop(['CFBC_N1', 'CFBC_N2', 'CFCC_N1', 'CFCC_N2', 'EC_PR', 'CFBC_PR', 'CFCC_PR'], axis=1)
        total.columns = ['N1', 'N2', 'PR']
        total = pd.concat([total, cfbc_cfcc, ec_cfbc, ec_cfcc], ignore_index=True)

        # CLEAN VARIABLES
        del cpg, cfbc_cfcc, ec_cfbc, ec_cfcc, pr_cfbc, pr_cfcc, pr_ec
        # FILTER
        total = total[total['PR'] > 0]

        # REGULARIZATION
        CPG_graph = na.create_PPG(total)
        out_degree = CPG_graph.out_degree(na.nodes)

        res = []
        for idx, row in total.iterrows():
            res.append([row['N1'], row['N2'], row['PR'] / out_degree[row['N1']]])

        total = pd.DataFrame(res)
        total.columns = ['N1', 'N2', 'PR']
        del res
        CPG_graph = na.create_PPG(total)

        # RANK
        adj_matrix = nx.adjacency_matrix(CPG_graph)
        vector = [1 / len(na.nodes) for x in range(len(na.nodes))]
        # p0 = adj_matrix * vector

        rank = na.rank_walk(t, vector, adj_matrix)
        rank = {x: rank[x] for x in na.nodes}
        print(rank)
        del vector, CPG_graph, total

        return rank

    @staticmethod
    def get_key(val, my_dict):
        for key, value in my_dict.items():
            if val == value:
                return key

        return "key doesn't exist"

    @staticmethod
    def get_gene_name(rank, node_name, genes_exp_values):
        # SORT DICT RESULTS
        print('[INFO] GET GENE SYMBOL')
        rank = {k: v for k, v in sorted(rank.items(), key=lambda item: item[1], reverse=True)}
        r = 20
        field_names = ['gene', 'influence', 'avg.tattoo']
        gene_values_df = pd.DataFrame(columns=field_names)
        count = 0
        while r >= 0:
            for k, v in rank.items():
                gene_name = na.get_key(k, node_name)
                for idx, row in genes_exp_values.iterrows():
                    if row['gene'] == gene_name:
                        gene_values_df.loc[count] = [gene_name, v*10*129, row['avg.tattoo']]
                        count += 1

                r = r - 1
        del rank, node_name
        return gene_values_df

    @staticmethod
    def save_results(gene_values_df, filename):
        print('[INFO] SAVING FILE')
        gene_values_filename = '/'.join(['/'.join(filename.split('/')[:-1]), 'influence-' +
                                          filename.split('/')[-1].replace('tsv', 'csv')])
        gene_values_df.to_csv(gene_values_filename, index=False)

        print('[INFO] DF FILE SAVED {}'.format(gene_values_filename))


def create_sc_exp_attr(gene_expression, node_name):
    node_weights = list()
    for k, v in node_name.items():
        for idx, ge_row in gene_expression.iterrows():
            if ge_row['gene'] == k:
                node_weights.append((v, {"weight": ge_row['avg.tattoo']}))

    return node_weights


def create_graph_from_pandas(filename, genes_exp_values):
    print('[INFO] GETTING DATAFRAME AND CREATING NETWORK')
    try:
        data = pd.read_csv(filename, sep='\t', header=0)
        gene_names = list(set(list(data.node1.unique()) + list(data.node2.unique())))
        node_name = {gene_names[x]: x for x in range(len(gene_names))}
        gene_expression = pd.read_excel(genes_exp_values) if genes_exp_values.endswith('.xls') \
            else pd.read_csv(genes_exp_values)
        nodes_and_weights = create_sc_exp_attr(gene_expression, node_name)
        data_g = nx.Graph()
        data_g.add_nodes_from(nodes_and_weights)
        N1 = [node_name[x] for x in data.node1]
        N2 = [node_name[x] for x in data.node2]
        data_g.add_edges_from(zip(N1, N2))

        data_g.remove_nodes_from(list(nx.isolates(data_g)))

        # remove isolated nodes and non connected components
        # if not nx.is_connected(data_g):
        #     components = [data_g.subgraph(c).copy() for c in nx.connected_components(data_g)]
        #     component_main = components[0]
        #     for idx, g in enumerate(components):
        #         if len(g.nodes()) == len(component_main.nodes()):
        #             pass
        #         if len(g.nodes()) > len(component_main.nodes()):
        #             data_g.remove_nodes_from(component_main.nodes())
        #             node_name_key_list = list(node_name.keys())
        #             node_name_val_list = list(node_name.values())
        #
        #             # remove nodes also from the list node_name
        #             for node in component_main.nodes():
        #                 pos = node_name_val_list.index(node)
        #                 poss = node_name_key_list[pos]
        #                 del node_name[poss]
        #
        #             component_main = g
        #         if len(g.nodes()) < len(component_main.nodes()):
        #             data_g.remove_nodes_from(g.nodes())
        #             node_name_key_list = list(node_name.keys())
        #             node_name_val_list = list(node_name.values())
        #
        #             # remove nodes also from the list node_name
        #             for node in g.nodes():
        #                 pos = node_name_val_list.index(node)
        #                 poss = node_name_key_list[pos]
        #                 del node_name[poss]

        data_g.remove_nodes_from(list(nx.isolates(data_g)))
        del data, N1, N2
        return data_g, node_name, gene_expression
    except Exception as e:
        print('[ERROR] AN ERROR OCCURRED WHILE TRYING TO IMPORT DATAFRAME OR DURING GRAPH OBJECT CREATION')
        print(e)


if __name__ == '__main__':
    pool = mp.Pool()
    filename = 'tattoo-gonzalez-2020-data/inflammation-string-genes-tattoo-total-samples-total-conditions.tsv'
    genes_exp_values = "tattoo-gonzalez-2020-data/inflammation-genes.csv"
    G, node_name, gene_expression = create_graph_from_pandas(filename, genes_exp_values)
    t = 100
    na = Network_analysis(G, list(node_name.values()))

    # CREAZIONE DEI DATAFRAME

    print('[INFO] EXECUTING MIN MAX NORMALIZATION')
    pr_cfbc, pr_cfcc, pr_ec = pool.map(Network_analysis.min_max_normalization,
                                       [na.current_flow_betweenness_c, na.reachability, na.eigenvector_c])

    print('[INFO] EXECUTING PREFERENCE RELATION NETWORK')
    pr_cfbc, pr_cfcc, pr_ec = pool.map(na.create_preference_relation,
                                       [pr_cfbc, pr_cfcc, pr_ec])

    print('[INFO] EXECUTING CPG RANKING')
    result = Network_analysis.cpg_rank(pr_cfbc, pr_cfcc, pr_ec)
    gene_values = na.get_gene_name(result, node_name, gene_expression)
    print('[INFO] THIS IS THE RESULT')
    print(gene_values)
    Network_analysis.save_results(gene_values, filename)
    pool.close()





