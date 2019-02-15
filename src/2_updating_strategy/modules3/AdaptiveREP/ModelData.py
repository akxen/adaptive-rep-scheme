"""Load raw data and organise into format useful for model"""

import os
import pickle

import numpy as np
import pandas as pd


class RawData:
    """Load raw data to be used in model"""

    def __init__(self, data_dir, scenarios_dir, seed=10):

        # Paths to directories
        # --------------------
        # Core data directory
        self.data_dir = data_dir

        # Network data
        # ------------
        # Nodes
        self.df_n = pd.read_csv(os.path.join(self.data_dir,
                                             'egrimod-nem-dataset-v1.3',
                                             'akxen-egrimod-nem-dataset-4806603',
                                             'network',
                                             'network_nodes.csv'), index_col='NODE_ID')

        # AC edges
        self.df_e = pd.read_csv(os.path.join(self.data_dir,
                                             'egrimod-nem-dataset-v1.3',
                                             'akxen-egrimod-nem-dataset-4806603',
                                             'network', 'network_edges.csv'), index_col='LINE_ID')

        # HVDC links
        self.df_hvdc_links = pd.read_csv(os.path.join(self.data_dir,
                                                      'egrimod-nem-dataset-v1.3',
                                                      'akxen-egrimod-nem-dataset-4806603',
                                                      'network',
                                                      'network_hvdc_links.csv'), index_col='HVDC_LINK_ID')

        # AC interconnector links
        self.df_ac_i_links = pd.read_csv(os.path.join(self.data_dir,
                                                      'egrimod-nem-dataset-v1.3',
                                                      'akxen-egrimod-nem-dataset-4806603',
                                                      'network',
                                                      'network_ac_interconnector_links.csv'), index_col='INTERCONNECTOR_ID')

        # AC interconnector flow limits
        self.df_ac_i_limits = pd.read_csv(os.path.join(self.data_dir,
                                                       'egrimod-nem-dataset-v1.3',
                                                       'akxen-egrimod-nem-dataset-4806603',
                                                       'network',
                                                       'network_ac_interconnector_flow_limits.csv'), index_col='INTERCONNECTOR_ID')

        # Generators
        # ----------
        # Generating unit information
        self.df_g = pd.read_csv(os.path.join(self.data_dir,
                                             'egrimod-nem-dataset-v1.3',
                                             'akxen-egrimod-nem-dataset-4806603',
                                             'generators',
                                             'generators.csv'), index_col='DUID', dtype={'NODE': int})

        # Perturb short-run marginal costs (SRMCs) so all are unique.
        # (add uniformly distributed random number between 0 and 2 to each SRMC. Set seed so this randomness
        # can be reproduced)
        np.random.seed(seed)
        self.df_g['SRMC_2016-17'] = self.df_g['SRMC_2016-17'] + np.random.uniform(0, 2, self.df_g.shape[0])

        # Load scenario data
        # ------------------
        with open(os.path.join(scenarios_dir, 'weekly_scenarios.pickle'), 'rb') as f:
            self.df_scenarios = pickle.load(f)


class OrganisedData(RawData):
    """Organise data to be used in mathematical program"""

    def __init__(self, data_dir, scenarios_dir):
        # Load model data
        super().__init__(data_dir, scenarios_dir)

    def get_admittance_matrix(self):
        """Construct admittance matrix for network"""

        # Initialise dataframe
        df_Y = pd.DataFrame(data=0j, index=self.df_n.index, columns=self.df_n.index)

        # Off-diagonal elements
        for index, row in self.df_e.iterrows():
            fn, tn = row['FROM_NODE'], row['TO_NODE']
            df_Y.loc[fn, tn] += - (1 / (row['R_PU'] + 1j * row['X_PU'])) * row['NUM_LINES']
            df_Y.loc[tn, fn] += - (1 / (row['R_PU'] + 1j * row['X_PU'])) * row['NUM_LINES']

        # Diagonal elements
        for i in self.df_n.index:
            df_Y.loc[i, i] = - df_Y.loc[i, :].sum()

        # Add shunt susceptance to diagonal elements
        for index, row in self.df_e.iterrows():
            fn, tn = row['FROM_NODE'], row['TO_NODE']
            df_Y.loc[fn, fn] += (row['B_PU'] / 2) * row['NUM_LINES']
            df_Y.loc[tn, tn] += (row['B_PU'] / 2) * row['NUM_LINES']

        return df_Y

    def get_HVDC_incidence_matrix(self):
        """Incidence matrix for HVDC links

        Returns
        -------
        df : pandas DataFrame
            DataFrame describing HVDC connections between nodes
        """

        # Incidence matrix for HVDC links
        df = pd.DataFrame(index=self.df_n.index, columns=self.df_hvdc_links.index, data=0)

        # Loop through links, update incidence matrix
        for index, row in self.df_hvdc_links.iterrows():
            # From nodes assigned a value of 1
            df.loc[row['FROM_NODE'], index] = 1

            # To nodes assigned a value of -1
            df.loc[row['TO_NODE'], index] = -1

        return df

    def get_all_ac_edges(self):
        """Tuples defining from and to nodes for all AC edges (forward and reverse)

        Returns
        -------
        edge_set : set
            Set of AC edges within network
        """

        # Set of all AC edges
        edge_set = set()

        # Loop through edges, add forward and reverse direction indice tuples to set
        for index, row in self.df_e.iterrows():
            edge_set.add((row['FROM_NODE'], row['TO_NODE']))
            edge_set.add((row['TO_NODE'], row['FROM_NODE']))

        return edge_set

    def get_network_graph(self):
        """Graph containing connections between all network nodes

        Returns
        -------
        network_graph : dict
            Key is node, value is set of all nodes directly connected to the 'key' node
        """

        # Initialise map between all nodes directly connected to each node
        network_graph = {n: set() for n in self.df_n.index}

        # Loop through AC edges, update map
        for index, row in self.df_e.iterrows():
            network_graph[row['FROM_NODE']].add(row['TO_NODE'])
            network_graph[row['TO_NODE']].add(row['FROM_NODE'])

        return network_graph

    def get_all_dispatchable_fossil_generator_duids(self):
        """Fossil dispatch generator DUIDs"""

        # Filter - keeping only fossil and scheduled generators
        mask = (self.df_g['FUEL_CAT'] == 'Fossil') & (self.df_g['SCHEDULE_TYPE'] == 'SCHEDULED')

        # All fossil / dispatchable generators
        dispatchable_fossil_generator_duids = self.df_g[mask].index

        return dispatchable_fossil_generator_duids

    def get_reference_nodes(self):
        """Get reference node IDs

        Returns
        -------
        reference_node_ids : pandas Index
            Node IDs corresponding to reference nodes in each AC network
        """

        # Filter Regional Reference Nodes (RRNs) in Tasmania and Victoria.
        mask = (self.df_n['RRN'] == 1) & (self.df_n['NEM_REGION'].isin(['TAS1', 'VIC1']))
        reference_node_ids = self.df_n[mask].index

        return reference_node_ids

    def get_generator_node_map(self, generators):
        """Get set of generators connected to each node

        Returns
        -------
        generator_node_map : pandas DataFrame
            Map between nodes and all generators connected to each node
        """

        # Generators connected to each node
        generator_node_map = (self.df_g.reindex(index=generators)
                              .reset_index()
                              .rename(columns={'OMEGA_G': 'DUID'})
                              .groupby('NODE').agg(lambda x: set(x))['DUID']
                              .reindex(self.df_n.index, fill_value=set()))

        return generator_node_map

    def get_ac_interconnector_summary(self):
        """Summarise aggregate flow limit information for AC interconnectors

        Returns
        -------
        df_interconnector_limits : pandas DataFrame
            Summary of AC interconnectors and the individual branches that constitute each
        """

        # Create dictionary containing collections of AC branches for which interconnectors are defined.
        # These collections are for both forward and reverse directions.
        interconnector_limits = {}

        # Loop through each AC interconnector
        for index, row in self.df_ac_i_limits.iterrows():
            # Forward limit
            interconnector_limits[index + '-FORWARD'] = {'FROM_REGION': row['FROM_REGION'], 'TO_REGION': row['TO_REGION'], 'LIMIT': row['FORWARD_LIMIT_MW']}

            # Reverse limit
            interconnector_limits[index + '-REVERSE'] = {'FROM_REGION': row['TO_REGION'], 'TO_REGION': row['FROM_REGION'], 'LIMIT': row['REVERSE_LIMIT_MW']}

        # Convert to DataFrame
        df_interconnector_limits = pd.DataFrame(interconnector_limits).T

        # Find all branches that consitute each interconnector - order is important.
        # First element is 'from' node, second is 'to' node
        branch_collections = {b: {'branches': list()} for b in df_interconnector_limits.index}

        for index, row in self.df_ac_i_links.iterrows():
            # For a given branch, find the interconnector index to which it belongs. This will either be the forward or
            # reverse direction as defined in the interconnector links DataFrame. If the forward direction, 'FROM_REGION'
            # will match between DataFrames, else it indicates the link is in the reverse direction.

            # Assign branch to forward interconnector limit ID
            mask_forward = (df_interconnector_limits.index.str.contains(index)
                            & (df_interconnector_limits['FROM_REGION'] == row['FROM_REGION'])
                            & (df_interconnector_limits['TO_REGION'] == row['TO_REGION']))

            # Interconnector ID corresponding to branch
            branch_index_forward = df_interconnector_limits.loc[mask_forward].index[0]

            # Add branch tuple to branch collection
            branch_collections[branch_index_forward]['branches'].append((row['FROM_NODE'], row['TO_NODE']))

            # Assign branch to reverse interconnector limit ID
            mask_reverse = (df_interconnector_limits.index.str.contains(index)
                            & (df_interconnector_limits['FROM_REGION'] == row['TO_REGION'])
                            & (df_interconnector_limits['TO_REGION'] == row['FROM_REGION']))

            # Interconnector ID corresponding to branch
            branch_index_reverse = df_interconnector_limits.loc[mask_reverse].index[0]

            # Add branch tuple to branch collection
            branch_collections[branch_index_reverse]['branches'].append((row['TO_NODE'], row['FROM_NODE']))

        # Append branch collections to interconnector limits DataFrame
        df_interconnector_limits['branches'] = pd.DataFrame(branch_collections).T['branches']

        return df_interconnector_limits
