import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import warnings
import itertools
import plotly.graph_objects as go

from sklearn.metrics import pairwise_distances
from sklearn import datasets
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import AgglomerativeClustering

from gtda.plotting import plot_point_cloud
from ripser import ripser
from persim import PersistenceImager, plot_diagrams
import gudhi as gd
from ot import fused_gromov_wasserstein2
import ot

import trimesh


"""
Main Class
"""

class DecoratedReebGraph:

    """
    Creates a Reeb Graph from a Euclidean point cloud or finite metric space.

    Reeb Graphs can be 'decorated' with higher-dimensional homological data by the 'fit_barcode' method.
    The result is a 'Decorated Reeb Graph (DRG)'.
    """

    def __init__(self,
                data = None,
                function = None,
                distance_matrix = False):

        """
        'data' should be an np.array corresponding to a Euclidean pointcloud, with shape = samples-by-features
        OR a square distance matrix. In the latter case, 'distance_matrix' should be set to True

        'function' should be one of the following:
        - a 1d np.array corresponding to function values on the data points
        - the string 'eigenfunction', in which case a Laplacian eigenfunction of the VR 1-skeleton will be used
        - the string 'PCA', in which case projection onto the first principal direction will be used.
          This option is only valid for Euclidean point cloud input data.
        """

        self.data = data
        self.function = function
        self.distance_matrix = distance_matrix
        self.metadata = {}
        self.VRComplex = None
        self.MapperGraph = None
        self.coords = None

        if self.data.shape[0]==self.data.shape[1]:
            if not self.distance_matrix:
                warnings.warn("Input data is a square matrix. If it is a distance matrix, set 'distance_matrix == True'.")
        else:
            if self.distance_matrix:
                raise TypeError("If distance_matrix flag is True, input data must be a square matrix.")

        if self.distance_matrix:
            self.distances = self.data
        else:
            D = pairwise_distances(self.data)
            self.distances = D

        min_rad = ripser(D, distance_matrix=True, maxdim = 0)['dgms'][0][-2][1]
        self.metadata['Connected VR complex radius'] = min_rad

        if not distance_matrix:
            if data.shape[1]==2:
                self.coords = data
            elif data.shape[1]==3:
                self.coords = data
            else:
                pca = PCA(n_components=3)
                self.coords = pca.fit_transform(data)


    """
    Creating a Reeb Graph
    """

    def fit_Vietoris_Rips(self,
                         sparse = 0.5,
                         min_rad_factor = 2,
                         max_dim = 2,
                         with_weights = True):

        """
        Constructs the 2-skeleton of a Vietoris-Rips complex with chosen radius.

        Input:
        sparse: We use sparse approximations of VR for speed, following [1]. This is a sparsity parameter.
        min_rad_factor: the radius for the VR construction is min_rad_factor*min_rad,
                        where min_rad is the smallest radius at which VR is connected.
                        This method makes the choice of scale easier, because it is relative to data scale.
        max_dim: max dimension of simplices included in VR complex
        with_weights: flag to output a weighted graph representing the 1-skeleton of the VR complex

        Creates:
        self.VRComplex: simplex tree for Vietoris-Rips complex
        self.VRSkeleta: lists of 1 and 2-simplices in VR complex
        self.VRGraph: networkx graph representing 1-skeleton of VR complex

        [1] Donald R. Sheehy. Linear-size approximations to the Vietoris-Rips filtration.
        Discrete & Computational Geometry, 49(4):778–796, 2013.
        """

        D = self.distances
        min_rad = self.metadata['Connected VR complex radius']
        function = self.function

        Rips_complex, skeleta = Vietoris_Rips(D,min_rad_factor = min_rad_factor,
                                              max_dim = max_dim,sparse = sparse,min_rad = min_rad)

        H = VR_to_graph(skeleta, function, with_weights = with_weights)

        self.metadata['Vietoris-Rips radius'] = min_rad_factor*min_rad
        self.VRComplex = Rips_complex
        self.VRSkeleta = skeleta
        self.VRGraph = H

    def fit_Reeb(self,
                   n_bins = 10,
                   density_factor = 0.0,
                   add_coords = True,
                   sparse = 0.5,
                   min_rad_factor = 2,
                   max_dim = 2,
                   with_weights = True):

        """
        Constructs an approximation of the Reeb graph for the VR complex using a Mapper-like construction [2].

        Input:
        n_bins: number of bins to use when partitioning the range of the function
        density_factor: Setting above 0 will ignore small clusters in the Reeb graph construction.
        add_coords: add coordinates to the nodes of the Reeb graph, for plotting.
        Remaining arguments: Same as in fit_Vietoris_Rips.

        Creates:
        self.ReebGraph: Approximate Reeb graph of the VR Complex.

        [2] Singh G, Mémoli F, Carlsson GE.
        Topological methods for the analysis of high dimensional data sets and 3d object recognition.
        PBG@ Eurographics. 2007 Sep 2;2:091-100.
        """

        D = self.distances
        min_rad = self.metadata['Connected VR complex radius']
        function = self.function
        distance_matrix = self.distance_matrix

        # Need to fit VR complex before mapper.
        # This will be done automatically if it has not been done already.
        if self.VRComplex is None:

            Rips_complex, skeleta = Vietoris_Rips(D,min_rad_factor = min_rad_factor,
                                                  max_dim = max_dim,sparse = sparse,min_rad = min_rad)

            H = VR_to_graph(skeleta, function, with_weights = with_weights)

            self.metadata['Vietoris-Rips radius'] = min_rad_factor*min_rad
            self.VRComplex = Rips_complex
            self.VRSkeleta = skeleta
            self.VRGraph = H

        H = self.VRGraph

        G, colors = Reeb_approx_graph(H, function, n_bins,
                                      return_embedding_data = True, density_factor = density_factor)

        if add_coords and not distance_matrix:
            coords = self.coords
            if coords.shape[1]==2:
                G = add_2d_node_positions(G,coords)
            elif coords.shape[1]==3:
                G = add_3d_node_positions(G,coords)

        self.ReebGraph = G

    def fit_diagrams(self,
                         homology_dimension = 1,
                         persistence_images = False,
                         persistence_statistics = True,
                         pixel_size = 0.2,
                         birth_range = None,
                         pers_range = None,
                         kernel_params = {'sigma': [[0.25, 0.0], [0.0, 0.25]]}):

            """
            Adds persistence diagrams (and persistence images) to each node, by computing VR persistent
            homology of the set of points which map to the node in the Reeb graph construction.

            Input:
            homology_dimension: number of bins to use when partitioning the range of the function
            persistence_images: Each diagram can be converted to a persistence image.
            persistence_statistics: Each diagram can be converted to a statistical summary.
            Remaining arguments: Hyperparameters for constructing persistence images.

            Updates:
            self.ReebGraph: Adds features to the nodes for diagrams (and persistence images).

            [3] Adams H, Emerson T, Kirby M, Neville R, Peterson C, Shipman P, Chepushtanova S, Hanson E, Motta F, Ziegelmeier L.
            Persistence images: A stable vector representation of persistent homology.
            Journal of Machine Learning Research. 2017 Jan;18.
            """

            G = self.ReebGraph
            data = self.data
            distance_matrix = self.distance_matrix

            if distance_matrix:
                G = decorate_Reeb_Graph_metric_space(G,data,homology_dimension = homology_dimension, return_stats = False)
            else:
                G = decorate_Reeb_Graph(G,data,homology_dimension = homology_dimension, return_stats = False)

            if persistence_images:

                G = fit_persistence_images(G,
                                          pixel_size = pixel_size,
                                          birth_range = birth_range,
                                          pers_range = pers_range,
                                          kernel_params = kernel_params)

            if persistence_statistics:

                G = get_persistence_statistics(G)


            self.ReebGraph = G
    
    def fit_diagrams_global(self,
                            homology_dimension = 1,
                         persistence_images = False,
                         persistence_statistics = True,
                         truncate_inf = True,
                         truncation_factor = 5,
                         pixel_size = 0.2,
                         birth_range = None,
                         pers_range = None,
                         kernel_params = {'sigma': [[0.25, 0.0], [0.0, 0.25]]}):
        
        G = self.ReebGraph
        data = self.data
        VRComplex = self.VRComplex
        VRSkeleta = self.VRSkeleta
        
        G = decorate_Reeb_Graph_global(G,VRComplex,VRSkeleta,homology_dimension = homology_dimension,
                                      truncate_inf=truncate_inf, truncation_factor=truncation_factor)
        
        if persistence_statistics:
            G = get_persistence_statistics_global(G)
            
        if persistence_images:

            G = fit_persistence_images(G,
                                  pixel_size = pixel_size,
                                  birth_range = birth_range,
                                  pers_range = pers_range,
                                  kernel_params = kernel_params,
                                      diagram_type = 'global_diagram')
        
        self.ReebGraph = G
        
        

    def draw_Reeb(self):

        """
        Draws a 2 or 3d plot of Reeb graph, informed by original point cloud coordinates.
        Only valid for Euclidean input data.
        """

        coords = self.coords
        G = self.ReebGraph

        if coords is not None:
            if coords.shape[1]==2:
                nx.draw(G,pos = coords)
            if coords.shape[1]==3:
                plot_3d_graph(G)


"""
Helper Functions
"""


def get_skeleta(rips_list):

    zero_skeleton = [spx[0][0] for spx in rips_list if len(spx[0])==1]
    one_skeleton = [spx[0] for spx in rips_list if len(spx[0])==2]
    two_skeleton = [spx[0] for spx in rips_list if len(spx[0])==3]

    skeleta = [zero_skeleton,one_skeleton,two_skeleton]

    return skeleta

def Vietoris_Rips(D,
                  min_rad_factor = 1.5,
                  max_dim = 2,
                  sparse = 0.5,
                  min_rad = None):

        """
        Constructs the 2-skeleton of a Vietoris-Rips complex with chosen radius.
        Input:
        D is a distance matrix
        min_rad_factor: the radius for the VR construction is min_rad_factor*min_rad,
                        where min_rad is the smallest radius at which VR is connected.
                        This method makes the choice of scale easier, because it is relative to data scale.
        max_dim: max dimension of simplices included in VR complex
        sparse: We use sparse approximations of VR for speed, following [27]. This is a sparsity parameter.
        min_rad: if min_rad is precomputed, it can be passed to the function.
                Otherwise, it is computed as a subroutine.

        Output:
        Rips_complex: gudhi simplex tree representation of VR complex
        skeleta: list of all simplices inf the VR complex

        [27] Donald R. Sheehy. Linear-size approximations to the Vietoris-Rips filtration. Discrete & Computational Geometry, 49(4):778–796, 2013.
        """

        # Compute the minimum radius at which VR Complex becomes connected
        if min_rad is None:
            min_rad = ripser(D, distance_matrix=True, maxdim = 0)['dgms'][0][-2][1]

        # Extract radius for VR complex
        radius = min_rad_factor*min_rad

        # Construct Vietoris-Rips complex
        skeleton = gd.RipsComplex(distance_matrix = D,
                                  max_edge_length = radius,
                                  sparse = sparse)
        Rips_complex = skeleton.create_simplex_tree(max_dimension = max_dim)
        rips_filtration = Rips_complex.get_filtration()
        rips_list = list(rips_filtration)
        skeleta = get_skeleta(rips_list)

        return Rips_complex, skeleta

def VR_to_graph(VRSkeleta,function,with_weights = False):

    skeleta = VRSkeleta
    nodes = skeleta[0]

    if with_weights:
        edges = [(edge[0],edge[1],np.abs(function[edge[0]]-function[edge[1]])) for edge in skeleta[1]]
    else:
        edges = [(edge[0],edge[1]) for edge in skeleta[1]]

    VRGraph = nx.Graph()

    VRGraph.add_nodes_from(nodes)

    if with_weights:
        VRGraph.add_weighted_edges_from(edges)
    else:
        VRGraph.add_edges_from(edges)

    return VRGraph

def Reeb_approx_graph(H, function, n_bins, return_embedding_data = True, density_factor = 0.1):

    """
    Input H is a graph, function is a list of node values
    """


    # Partition the range of the function
    bin_range = [np.min(function),np.max(function)+1e-6]
    endpoints = np.linspace(bin_range[0],bin_range[1],n_bins)
    bins = np.array([[endpoints[i],endpoints[i+1]] for i in range(len(endpoints)-1)])

    # Induce partition on the nodes
    subsets_idx = [np.where((current_bin[0] <= function)*(function < current_bin[1]))[0] for current_bin in bins]

    # Construct the Mapper graph...
    G = nx.Graph()

    # ...First Level
    current_bin = bins[0]
    subset_current = subsets_idx[0]
    H_current = H.subgraph(subset_current)
    H_current_components = [list(conn_cmp) for conn_cmp in nx.connected_components(H_current) if len(conn_cmp) > len(H)/n_bins*density_factor]

    attrs = {}
    colors = []

    for k, H0 in enumerate(H_current_components):
        G.add_node((0,k))
        func_value = (current_bin[1]+current_bin[0])/2
        attrs[(0,k)] = {'component indices':H0, 'function value':func_value}
        if return_embedding_data:
            colors.append(func_value)

    nx.set_node_attributes(G, attrs)

    H_prev_components = H_current_components

    # ...Iterate Remaining Levels
    for j in range(1,len(subsets_idx)):

        current_bin = bins[j]
        subset_current = subsets_idx[j]
        H_current = H.subgraph(subset_current)
        H_current_components = [list(conn_cmp) for conn_cmp in nx.connected_components(H_current) if len(conn_cmp) > len(H)/n_bins*density_factor]

        attrs = {}

        for k, H0 in enumerate(H_current_components):

            G.add_node((j,k))
            func_value = (current_bin[1]+current_bin[0])/2
            attrs[(j,k)] = {'component indices':H0, 'function value':func_value}

            if return_embedding_data:
                colors.append(func_value)

            for ll, H1 in enumerate(H_prev_components):
                if nx.is_connected(H.subgraph(H0+H1)):
                    wt = np.mean(function[H0]) - np.mean(function[H1])
                    G.add_edge((j,k),(j-1,ll), weight = wt)

        nx.set_node_attributes(G, attrs)

        H_prev_components = H_current_components

    if return_embedding_data:
        return G, colors
    else:
        return G

def add_3d_node_positions(G,data):

    attrs = {}

    for node in G.nodes:
        pos = list(np.mean(data[G.nodes[node]['component indices']],axis = 0))
        attrs[node] = {'3d pos':pos}

    nx.set_node_attributes(G,attrs)

    return G

def add_2d_node_positions(G,data):

    attrs = {}

    for node in G.nodes:
        pos = list(np.mean(data[G.nodes[node]['component indices']],axis = 0))
        attrs[node] = {'2d pos':pos}

    nx.set_node_attributes(G,attrs)

    return G

def plot_3d_graph(G):

    """
    This code is tailored to graphs which come from 3d point clouds,
    using the Vietoris-Rips+Mapper Construction
    """

    fig = go.Figure()

    # Add nodes
    for node in G.nodes():
        pos = G.nodes[node]['3d pos']
        fig.add_trace(go.Scatter3d(x=[pos[0]], y=[pos[1]], z=[pos[2]], mode='markers', marker=dict(size=10, color='blue')))

    # Add edges to the figure
    for edge in G.edges():
        pos0 = G.nodes[edge[0]]['3d pos']
        pos1 = G.nodes[edge[1]]['3d pos']
        fig.add_trace(go.Scatter3d(x=[pos0[0], pos1[0]], y=[pos0[1], pos1[1]], z=[pos0[2], pos1[2]], mode='lines', line=dict(width=1, color='black')))

    # Set the layout of the figure
    fig.update_layout(width=800, height=800, showlegend=False, margin=dict(l=0, r=0, b=0, t=0))

    # Show the figure
    fig.show()

    return

def decorate_Reeb_Graph(G,data,homology_dimension = 1,return_stats=False,simplify = False,truncation_value = 1.0):

    attr = {}

    Gnew = G.copy()

    births = []
    persistences = []

    for node in Gnew:

        X0_idx = Gnew.nodes[node]['component indices']
        X0 = data[X0_idx]
        dgm = ripser(X0,maxdim=homology_dimension,distance_matrix=False)['dgms'][-1]
        if return_stats:
            births += list(dgm[:,0])
            persistences += list(dgm[:,1] - dgm[:,0])
            
        if simplify:
            dgm = np.array([bar for bar in dgm if bar[1]-bar[0] > truncation_value])
            
        attr[node] = {'diagram':dgm}

    nx.set_node_attributes(Gnew,attr)

    if return_stats:
        return Gnew, births, persistences
    else:
        return Gnew

def decorate_Reeb_Graph_metric_space(G,D,homology_dimension = 1, return_stats = True):

    attr = {}

    Gnew = G.copy()

    births = []
    persistences = []

    for node in Gnew:

        X0_idx = Gnew.nodes[node]['component indices']
        X0 = D[np.ix_(X0_idx,X0_idx)]
        dgm = ripser(X0,maxdim=homology_dimension,distance_matrix=True)['dgms'][-1]
        if return_stats:
            births += list(dgm[:,0])
            persistences += list(dgm[:,1] - dgm[:,0])
        attr[node] = {'diagram':dgm}

    nx.set_node_attributes(Gnew,attr)

    if return_stats:
        return Gnew, births, persistences
    else:
        return Gnew

def fit_persistence_images(G,pixel_size = 0.2,birth_range = None,
                           pers_range = None,kernel_params = {'sigma': [[0.25, 0.0], [0.0, 0.25]]},
                          diagram_type = 'diagram'):

    if birth_range is None:
        raise Exception("Must set a birth_range = (a,b).")

    if pers_range is None:
        pers_range = birth_range

    Gnew = G.copy()

    attr = {}

    pimgr = PersistenceImager(pixel_size=pixel_size, birth_range=birth_range,
                              pers_range = pers_range, kernel_params = kernel_params)

    if diagram_type == 'diagram':
        for node in Gnew:
            dgm = Gnew.nodes[node]['diagram']
            img = pimgr.transform(dgm, skew = True).T
            attr[node] = {'persistence_image':img}

        nx.set_node_attributes(Gnew,attr)
        
    elif diagram_type == 'global_diagram':
        for node in Gnew:
            dgm = Gnew.nodes[node]['global_diagram']
            img = pimgr.transform(dgm, skew = True).T
            attr[node] = {'persistence_image_global':img}

        nx.set_node_attributes(Gnew,attr)

    return Gnew

"""
Distance Functions
"""


def get_attribution_distances(G1,G2,attribute):

    # attribute should be a string from {'diagram','persistence_image','persistence_statistics','persistence_image_global'}

    len1 = len(G1)
    len2 = len(G2)

    M = np.zeros([len1,len2])

    if attribute == 'diagram':
        for i,node1 in enumerate(G1):
            dgm1 = G1.nodes[node1]['diagram']
            for j,node2 in enumerate(G2):
                dgm2 = G2.nodes[node2]['diagram']
                M[i,j] = bottleneck(dgm1,dgm2)
    elif attribute == 'persistence_image':
        for i,node1 in enumerate(G1):
            PI1 = G1.nodes[node1]['persistence_image']
            for j,node2 in enumerate(G2):
                PI2 = G2.nodes[node2]['persistence_image']
                M[i,j] = np.linalg.norm(PI1-PI2)
    elif attribute == 'persistence_image_global':
        for i,node1 in enumerate(G1):
            PI1 = G1.nodes[node1]['persistence_image_global']
            for j,node2 in enumerate(G2):
                PI2 = G2.nodes[node2]['persistence_image_global']
                M[i,j] = np.linalg.norm(PI1-PI2)                
    elif attribute == 'persistence_statistics':
        for i,node1 in enumerate(G1):
            stats1 = G1.nodes[node1]['persistence_statistics']
            for j,node2 in enumerate(G2):
                stats2 = G2.nodes[node2]['persistence_statistics']
                M[i,j] = np.linalg.norm(stats1 - stats2)
    else:
        raise Exception("attribute variable should one of the strings 'diagram' or 'persistence_image' or 'persistence_statistics'.")

    return M

def DRG_to_measure_network(G,matrix_style = 'distance'):

    """
    matrix_style = 'distance' or 'adjacency'
    """

    if matrix_style == 'distance':
        C = np.array(nx.floyd_warshall_numpy(G))
    elif matrix_style == 'adjacency':
        C = np.array(nx.adjacency_matrix(G).todense())
    else:
        raise Exception("matrix_style should one of the strings 'distance' or 'adjacency'.")
    p_init = np.ones(len(G))
    p = p_init/sum(p_init)

    return C,p

def DRG_distance(G1,G2,attribute,matrix_style = 'distance',alpha = 0.5):

    C1,p1 = DRG_to_measure_network(G1, matrix_style = matrix_style)
    C2,p2 = DRG_to_measure_network(G2, matrix_style = matrix_style)

    if attribute == 'none':
        M = np.zeros([len(C1),len(C2)])
        dist = fused_gromov_wasserstein2(M,C1,C2,p1,p2,alpha = 1)

    else:
        M = get_attribution_distances(G1,G2,attribute)**2
        dist = fused_gromov_wasserstein2(M,C1,C2,p1,p2,alpha = alpha)

    return dist

"""
For persistence statistics:
"""

def entropy(dgm):

    if len(dgm) == 0:
        ent = 0
    else:
        lifespans = dgm[:,1]-dgm[:,0]
        L = np.sum(lifespans)
        
    ent = -np.sum(lifespans/L*np.log(lifespans/L))

    return ent

def persistence_statistics(dgm):

    """
    Input: dgm should be an np.array of size Nx2
    Output: vector of summary statistics for the diagram
    """

    if len(dgm) == 0:
        summary_stats = np.zeros(30)

    else:
        percentiles = [10,25,50,75,90]
        summary_stats = []

        births = dgm[:,0]
        deaths = dgm[:,1]
        lifespans = dgm[:,1]-dgm[:,0]
        midpoints = (dgm[:,1]+dgm[:,0])/2

        types = [births,deaths,lifespans,midpoints]

        for T in types:
            summary_stats += [np.mean(T),np.std(T)] + [np.percentile(T,p,method = 'midpoint') for p in percentiles]

        summary_stats.append(len(dgm))
        summary_stats.append(entropy(dgm))

        summary_stats = np.array(summary_stats)

    return summary_stats

def get_persistence_statistics(G):

    Gnew = G.copy()

    attr = {}

    for node in Gnew:
        dgm = Gnew.nodes[node]['diagram']
        attr[node] = {'persistence_statistics':persistence_statistics(dgm)}

    nx.set_node_attributes(Gnew,attr)

    return Gnew

"""
Plotting
"""

def total_persistence(dgm):
    
    persistences = dgm[:,1] - dgm[:,0]
    
    return np.sum(persistences)

def max_persistence(dgm):
    
    persistences = dgm[:,1] - dgm[:,0]
    if len(persistences)==0:
        return 0
    else:
        return np.max(persistences)

def plot_Reeb_3d(G,box = (-3,3),no_tick_labels = True, node_color = 'max', elev = 10, azim = 20):
    
    if node_color == 'total':
        colors = [total_persistence(G.nodes[node]['diagram']) for node in G.nodes]
    elif node_color == 'max':
        colors = [max_persistence(G.nodes[node]['diagram']) for node in G.nodes]
    elif node_color == 'births_global':
        colors = [np.mean(G.nodes[node]['global_diagram'][:,0]) for node in G.nodes]

        
    Reeb_graph_node_pos = np.array([G.nodes[node]['3d pos'] for node in G.nodes])

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(Reeb_graph_node_pos[:,0],Reeb_graph_node_pos[:,1],Reeb_graph_node_pos[:,2], 
               c = colors, cmap = plt.cm.copper,alpha = 0.8, s = 80)
    ax.set_xlim3d(box[0],box[1])
    ax.set_ylim3d(box[0],box[1])
    ax.set_zlim3d(box[0],box[1])
    for edge in G.edges:
        edge_source_pos = G.nodes[edge[0]]['3d pos']
        edge_target_pos = G.nodes[edge[1]]['3d pos']
        segment = np.vstack([edge_source_pos,edge_target_pos])
        ax.plot(segment[:,0],segment[:,1],segment[:,2],c = 'black')

    if no_tick_labels:        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        
    ax.view_init(elev=elev, azim=azim)
    
    plt.show()
    
def plot_3d_point_cloud(data,box = (-3,3),no_tick_labels = True, function = None, elev = 10, azim = 20):

    if function is None:
        function = data[:,2]
        
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data[:,0],data[:,1],data[:,2], c = function, cmap = plt.cm.copper,alpha = 0.8)
    ax.set_xlim3d(box[0],box[1])
    ax.set_ylim3d(box[0],box[1])
    ax.set_zlim3d(box[0],box[1])
    
    if no_tick_labels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        
    ax.view_init(elev=elev, azim=azim)

    plt.savefig('pointCloudDRGExample.png')
    plt.show()
    
    return

"""
For Barcode Transforms
"""

def decorate_Reeb_Graph_global(G,VRComplex,VRSkeleta,homology_dimension = 1,return_stats=True,
                              truncate_inf = False, truncation_factor = 1.5):

    attr = {}

    Gnew = G.copy()
    
    node_dict = {node:j for j,node in enumerate(G.nodes)}
    Reeb_dists = np.array(nx.floyd_warshall_numpy(G))
    diam = np.max(Reeb_dists)
    
    point_cloud_to_reeb_dict = {}
    for node in G.nodes:
        for v in G.nodes[node]['component indices']:
            point_cloud_to_reeb_dict[v] = node_dict[node]

    for node in Gnew:

        node_idx = node_dict[node]
        filter_function = Reeb_dists[node_idx,:]
        
        Rips_complex = VRComplex
        rips_filtration = Rips_complex.get_filtration()

        for v in VRSkeleta[0]:
            Rips_complex.assign_filtration([v],filtration = filter_function[point_cloud_to_reeb_dict[v]])

        Rips_complex.make_filtration_non_decreasing()

        rips_filtration = Rips_complex.get_filtration()

        BarCodes = Rips_complex.persistence()

        BC1 = Rips_complex.persistence_intervals_in_dimension(homology_dimension)
        
        BC1new = []

        if truncate_inf:
            for bar in BC1:
                if bar[1] == np.inf:
                    bar = [bar[0],truncation_factor*diam]
                BC1new.append(bar)

        BC1new = np.array(BC1new)

        attr[node] = {'global_diagram':BC1new}

    nx.set_node_attributes(Gnew,attr)

    return Gnew

def get_persistence_statistics_global(G):

    Gnew = G.copy()

    attr = {}

    for node in Gnew:
        dgm = Gnew.nodes[node]['global_diagram']
        attr[node] = {'persistence_statistics_global':persistence_statistics(dgm)}

    nx.set_node_attributes(Gnew,attr)

    return Gnew