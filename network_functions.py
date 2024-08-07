import networkx as nx

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


import json
import csv


from scipy import stats
from scipy.spatial.distance import *


twenty_distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0',\
                          '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8',\
                          '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff',\
                          '#000000']

def depth_first_search(matrix,query_index,max_depth=1,depth=1,vertices=[],visited=[]):
    """
    Recursive function to visit all vertices that are reachable from a query vertex.
    param matrix: The adjacency matrix representation of a graph
    param query_index: The row/column index which defines the query vertex
    param max_depth: How deep to go (for a bipartite graph the maximum is 2)
    param depth: Keeps track of how deep we have gone
    param vertices: Store found vertices
    param visited: Store visited vertices so we have a terminating condition
    return list of vertices
    """
    visited.append(query_index)
    # Index row - find connected head vertices in the query index row. In other words,
    # find the vertices that the query vertex point to
    vertices.extend([i for i,v in enumerate(matrix[query_index]) if v > 0 and not i in visited])
    if depth < max_depth:
        for i in vertices:
            if i in visited:
                continue
            vertices = depth_first_search(matrix,i,max_depth=1,depth=1,vertices=vertices,visited=visited)
    return vertices

def adjacency_from_biadjacency(data_dict):
    """
    Build the full adjacency matrix from the binary part which represents bipartite 
    agreements-actor graph. The full adjacency is needed for DFS and by network packages.
    Rows and columns of the adjacency matrix are identical and
    are constructed from the binary-valued matrix in row-column order.
    The number of rows (and columns) in the adjacency matrix is therefore:
    binary_matix.shape[0] +  binary_matix.shape[1]
    param data_dict: A data dictionary
    return adjacency matrix and list of vertex labels. The latter is the concatenated lists of
    agreement and actor vertex labels
    """    
    binary_matrix = data_dict['matrix']
    size = binary_matrix.shape[0] + binary_matrix.shape[1]
    adjacency_matrix = np.zeros((size,size))
    
    # Get the range of the bin matrix rows to generate the upper triangle
    # of the adjacency matrix
    row_index = 0
    col_index = binary_matrix.shape[0]
    adjacency_matrix[row_index:row_index + binary_matrix.shape[0],\
           col_index:col_index + binary_matrix.shape[1]] = binary_matrix
    # Add in the lower triangle
    adjacency_matrix = adjacency_matrix + adjacency_matrix.T    
    adj_vertices = []
    adj_vertices.extend(data_dict['agreement_ids'])
    adj_vertices.extend(data_dict['actor_ids'])

    return adjacency_matrix,adj_vertices

def get_query_matrix(query_indices,matrix,max_depth=1,operator='OR'):    
    """
    Query an adjacency matrix using depth-first search
    param query_indices: The indices of the query vertices
    param matrix: The adjacency matrix we are querying
    param max_depth: Max depth of the search. Defaults to 1. Agreement-actor graphs are bipartite so
    the maximum depth is 2.
    param operator: Boolean operator to use on found vertices. AND restricts the results to entities
    that have an edge to all query vertices.
    return: An adjacency matrix for the set of found vertices and the indices of the found vertices
    """    
    found_indices = []
    for i,query_index in enumerate(query_indices):
        vertices = depth_first_search(matrix,query_index,max_depth=max_depth,vertices=[],visited=[])
        if i == 0:
            found_indices.extend(vertices)
        else:
            if operator == 'OR':
                found_indices = list(set(found_indices).union(set(vertices)))
            else:
                found_indices = list(set(found_indices).intersection(set(vertices)))
    # Add the query vertex to the found vertices
    found_indices.extend(query_indices)    
    found_indices = sorted(found_indices)
    # Extract the sub-matrix containing only the found vertices
    query_matrix = matrix[np.ix_(found_indices,found_indices)]
    return query_matrix,found_indices

def display_networkx_graph(matrix,vertex_ids,data_dict):
    node_labels = {i:label for i,label in enumerate(vertex_ids)}
    node_colors = []
    for vertex_id in vertex_ids:
        if vertex_id in data_dict['agreement_ids']:
            vertex_type = data_dict['agreements_dict'][vertex_id]['type']
        else:
            vertex_type = data_dict['actors_dict'][vertex_id]['type']
        node_colors.append(data_dict['color_map'][vertex_type])
    graph = nx.from_numpy_array(matrix, create_using=nx.Graph)
    f = plt.figure(figsize=(16,16))
    pos = nx.spring_layout(graph) 
    nx.draw_networkx(graph,pos,labels=node_labels,node_color=node_colors,node_size=400,font_size=12,alpha=0.6)
    plt.grid(False)
    plt.show()
    
def display_ts_networkx_graph(actor_ids,query_matrix,vertex_ids,data_dict,title='',file=''):
    node_labels = {}
    node_sizes = []
    for i,vertex_id in enumerate(vertex_ids):
        node_labels[i] = vertex_id
        if len(actor_ids) == 2:
            if vertex_id == actor_ids[0]:
                right_node = i
                node_sizes.append(1600)
            elif vertex_id == actor_ids[1]:
                left_node = i
                node_sizes.append(1600)
            else:
                node_sizes.append(400)
        else:
            if vertex_id == actor_ids[0]:
                right_node = i
                node_sizes.append(1600)
            else:
                node_sizes.append(400)
            
        
    node_colors = []
    for vertex_id in vertex_ids:
        if vertex_id in data_dict['agreement_ids']:
            vertex_type = data_dict['agreements_dict'][vertex_id]['type']
        else:
            vertex_type = data_dict['actors_dict'][vertex_id]['type']
        node_colors.append(data_dict['color_map'][vertex_type])

    graph = nx.from_numpy_array(query_matrix, create_using=nx.Graph)
    f = plt.figure(figsize=(8,8))
    pos = nx.circular_layout(graph)
    
    if len(actor_ids) == 2:
        pos[left_node] = np.array([-2,0.1])
    pos[right_node] = np.array([2,0])
        

    nx.draw_networkx(graph,pos,labels=node_labels,node_color=node_colors,node_size=node_sizes,\
                     font_size=12,alpha=0.6)
    plt.grid(False)
    plt.title(title,fontsize='xx-large')
    #if len(file) > 0:
    #    plt.savefig('../../outputs/' + file + '.png', bbox_inches='tight')
    plt.show()


def display_comatrix_as_networkx_graph(co_matrix,vertex_indices,vertex_list,data_dict,title=''):
    """
    Create and display a networkx graph of a co-occurrence matrix. Includes the display of singletons — vertices that have no co-occurrences.
    param co_matrix: Co-occurence matrix (only the uppoer triangle is used to avoid self loops and edge duplication)
    param vertex_indices: The indices of the vertices in the occurrence matrix. These are indices into the complete set of vertices of which the
    co-occurrence vertices may be a subset
    param vertex_list: Complete list of vertex identifiers. The indices in vertex_indices locate the identifiers for the co-occurring vertices.
    param data_dict: The application data dictionary.
    param title: Optional title.
    """
    co_matrix = np.triu(co_matrix,k=1)
    node_labels = {i:vertex_list[index] for i,index in enumerate(vertex_indices)}
    node_colors = [data_dict['color_map'][v.split('_')[0]] for _,v in node_labels.items()]
    graph = nx.from_numpy_array(co_matrix, create_using=nx.Graph)
    f = plt.figure(figsize=(16,16))
    pos = nx.circular_layout(graph) 
    nx.draw_networkx(graph,pos,labels=node_labels,node_color=node_colors,node_size=400,alpha=0.6)
    # Get the edge labels
    rc = np.nonzero(co_matrix) # Row and column indices of non-zero pairs
    z = list(zip(list(rc[0]),list(rc[1])))
    edge_labels = {t:co_matrix[t[0]][t[1]] for t in z}
    nx.draw_networkx_edge_labels(graph, pos,edge_labels,font_color='red',font_size=12)
    plt.grid(False)
    plt.title(title)
    plt.show()

def get_peace_processes(data_dict):
    """
    Get list of peace process names 
    param data_dict: The application's data dictionary obtained from load_agreement_actor_data()
    return: list of process names in alpha order
    """
    processes = [agreement_data['pp_name'].strip() for _,agreement_data in\
                 data_dict['agreements_dict'].items()]
    return sorted(list(set(processes)))

def get_peace_process_data(process_name,data_dict):
    
    # Get all agreements for a peace process
    agreement_ids = [agreement_id for agreement_id,agreement_data in data_dict['agreements_dict'].items()\
                        if agreement_data['pp_name'].strip()==process_name]
    print(len(agreement_ids))
    agreement_indices = [data_dict['agreement_ids'].index(agreement_id) for\
                            agreement_id in agreement_ids]
    
    sub_matrix = data_dict['matrix'][np.ix_\
                            (agreement_indices,range(0,data_dict['matrix'].shape[1]))]
    actor_indices = list(set([i for row in sub_matrix for i,v in enumerate(row) if v > 0]))
    actor_ids = [data_dict['actor_ids'][index] for index in actor_indices]
    
    matrix = sub_matrix[np.ix_(range(0,len(agreement_indices)),actor_indices)]
    matrix = np.array(matrix)
    pp_data_dict = {}
    pp_data_dict['name'] = process_name
    pp_data_dict['actor_ids'] = actor_ids
    pp_data_dict['agreement_ids'] = agreement_ids
    pp_data_dict['matrix'] = matrix    
    return pp_data_dict

def get_cooccurrence_matrices(matrix):
    # Actor-actor co-occurence matrix for a peace process
    V = np.matmul(matrix.T,matrix)
    # Agreement-agreement co-occurence matrix
    W = np.matmul(matrix,matrix.T)
    return (V,W)

def load_agreement_actor_data(actors_file,signatories_file,agreements_file,countries_file,data_path):
    # Stash data in a dictionary
    data_dict = {}
    
    # Read the CSVs
    # Actor data
    with open(data_path + actors_file, encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        actors_header = next(reader)
        actors_data = [row for row in reader]
        f.close()
    #print(actors_header)
    #print()
    
    # Links agreeements to actors
    with open(data_path + signatories_file, encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        signatories_header = next(reader)
        signatories_data = [row for row in reader]
        f.close()
    #print(signatories_header)
    #print()

    # Agreements
    with open(data_path + agreements_file, encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        agreements_header = next(reader)
        agreements_data = [row for row in reader]
        f.close()
    #print(agreements_header)
    #print()
    
    # Countries - links agreements to countries
    with open(data_path + countries_file, encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        countries_header = next(reader)
        countries_data = [row for row in reader]
        f.close()
    #print(countries_header)
    #print()
    
    # Using id_paax as the definitive actor identifier
    # Collect all actors from the actor data
    actor_ids = [row[actors_header.index('id_paax')].strip() for row in actors_data]
    actor_ids = sorted(list(set(actor_ids)))
    
    # Collect all actors from the signatory data and compare to actor_ids
    sig_actor_ids = [row[signatories_header.index('id_paax')].strip() for row in signatories_data]
    sig_actor_ids = sorted(list(set(sig_actor_ids)))
    
    # Check all the signatory actor IDs are in the full set of IDs
    assert(set(sig_actor_ids) <= set(actor_ids))
    
    # Got past the assert so our actor ids are OK
    
    
    # Using the AgtId as the definitive agreement identifier
    # Collect all agreements from the actor data
    agreement_ids = [row[agreements_header.index('AgtId')].strip() for row in agreements_data]
    agreement_ids = sorted(list(set(agreement_ids)))
    
    # Collect all agreements from the signatory data and compare to agreement_ids
    sig_agreement_ids = [row[signatories_header.index('AgtId')].strip() for row in signatories_data]
    sig_agreement_ids = sorted(list(set(sig_agreement_ids)))

    # Check all the signatory agreement IDs are in the full set of agreement IDs
    assert(set(sig_agreement_ids) <= set(agreement_ids))
    
    # Got past the assert so our agreement ids are OK
    
    # Check we have country data for all our signatory agreements
    country_agreement_ids = [row[countries_header.index('AgtId')].strip() for row in countries_data]
    country_agreement_ids = sorted(list(set(country_agreement_ids)))
    
    # Check all the signatory agreements have country data
    assert(set(sig_agreement_ids) <= set(country_agreement_ids))
    
    
    # Build the dictionaries
    countries_dict = {}
    for row in countries_data:
        agreement_id = row[countries_header.index('AgtId')].strip()
        if not agreement_id in sig_agreement_ids:
            continue
        agreement_id = 'AGT_' + agreement_id
        countries_dict[agreement_id] = {}
        countries_dict[agreement_id]['country'] = row[countries_header.index('Country_entity')].strip()
        countries_dict[agreement_id]['region'] = row[countries_header.index('Region')].strip()
        countries_dict[agreement_id]['iso'] = row[countries_header.index('ISO code')].strip()
        countries_dict[agreement_id]['gwno'] = row[countries_header.index('GWNO')].strip()
        
    actors_dict = {}
    for row in actors_data:
        actor_id = row[actors_header.index('id_paax')].strip()
        if not actor_id in sig_actor_ids:
            continue
        actors_dict[actor_id] = {}
        # This is the integer actor ID but not using it because want to use actor labels
        # e.g., CON_20 as actor_id
        actors_dict[actor_id]['id'] = row[actors_header.index('actor_id')].strip()
        actors_dict[actor_id]['name'] = row[actors_header.index('actor_name')].strip()
        actors_dict[actor_id]['region'] = row[actors_header.index('region_pax')].strip()
        actors_dict[actor_id]['type'] = row[actors_header.index('type')].strip()
        actors_dict[actor_id]['type_name'] = row[actors_header.index('actor_type')].strip()
        actors_dict[actor_id]['un_type'] = row[actors_header.index('un_type')].strip()
        actors_dict[actor_id]['ucdp_id'] = row[actors_header.index('ucdp_id')]
        actors_dict[actor_id]['ucdp_name'] = row[actors_header.index('ucdp_name')].strip()
        actors_dict[actor_id]['acled_name'] = row[actors_header.index('acled_name')].strip()
        
    agreements_dict = {}
    dates_dict = {}
    for row in agreements_data:
        agreement_id = row[agreements_header.index('AgtId')].strip()
        if not agreement_id in sig_agreement_ids:
            continue
        agreement_id = 'AGT_' + agreement_id
        agreements_dict[agreement_id] = {}
        agreements_dict[agreement_id]['type'] = 'AGT'
        agreements_dict[agreement_id]['date'] = row[agreements_header.index('Dat')].strip()
        agreements_dict[agreement_id]['pp_id'] = row[agreements_header.index('PP')].strip()
        agreements_dict[agreement_id]['pp_name'] = row[agreements_header.index('PPName')].strip()
        agreements_dict[agreement_id]['stage'] = row[agreements_header.index('Stage')].strip()
        agreements_dict[agreement_id]['stage_sub'] = row[agreements_header.index('StageSub')].strip()
        agreements_dict[agreement_id]['stage_label'] = row[agreements_header.index('stage_label')].strip()
        date = row[agreements_header.index('Dat')].strip().split('-')
        dates_dict[agreement_id] = int(''.join(date))
    
    # Build the agreement-actor biadjacency matrix - the core data structure
    matrix = np.zeros((len(agreements_dict),len(actors_dict)))
    for row in signatories_data:
        agreement_id = row[signatories_header.index('AgtId')].strip()
        agreement_index = sig_agreement_ids.index(agreement_id)
        actor_id = row[signatories_header.index('id_paax')].strip()
        actor_index = sig_actor_ids.index(actor_id)
        
        edge_dict = populate_edge_dict(row,signatories_header)
        matrix[agreement_index,actor_index] = get_edge_weight(edge_dict)
    matrix = np.array(matrix)
    
    # Assign colors to types
    vertex_types = []
    for _,actor_data in actors_dict.items():
        vertex_types.append(actor_data['type'])
    vertex_types.append('AGT')
    vertex_types = sorted(list(set(vertex_types)))

    # Build a colour map for types
    color_map = {type_:twenty_distinct_colors[i] for\
                 i,type_ in enumerate(vertex_types)}
    
    data_dict['actors_header'] = sig_actor_ids
    data_dict['agreements_header'] = sig_agreement_ids
    data_dict['countries_header'] = countries_dict
    data_dict['actor_ids'] = sig_actor_ids
    data_dict['agreement_ids'] = ['AGT_' + agreement_id for agreement_id in sig_agreement_ids]
    data_dict['countries_dict'] = countries_dict
    data_dict['actors_dict'] = actors_dict
    data_dict['agreements_dict'] = agreements_dict
    data_dict['dates_dict'] = dates_dict
    data_dict['color_map'] = color_map
    data_dict['matrix'] = matrix

    return data_dict

def get_empty_edge_dict():
    """
    Dictionary for storing signatory edge properties.
    The values of the properties form a binary string
    These strings are converted to integers for use a biadjacency cells values which are edge weights
    return dictionary of properties
    """
    edge_dict = {}
    edge_dict['is_party'] = 0
    edge_dict['is_third_party'] = 0
    edge_dict['other'] = 0
    return edge_dict

def populate_edge_dict(row,header):
    """
    Populate an edge dictionary from a signatory row for a given actor
    param row: Mediation data row
    param header: Mediation row header
    return dictionary of mediation-actor properties
    """
    edge_dict = get_empty_edge_dict()
    
    signatory_type = row[header.index('signatory_type')].strip()
    if signatory_type == 'party':
        edge_dict['is_party'] = 1
    elif signatory_type == 'third party':
        edge_dict['is_third_party'] = 1
    else:
        edge_dict['other'] = 1
    return edge_dict

def get_edge_weight(edge_dict):
    """
    Convert edge property values into an integer
    param edge_dict: dictionary containing set of Boolean valued edge properties
    return edge_weight: an integer encoding the binary string of edge properties
    """
    s = ''
    for k,v in edge_dict.items():
        s += str(v)
    return int(''.join(c for c in s),2)

def recover_edge_dict(edge_weight,props_length):
    """
    Recover the edge dictionary from an edge weight
    param edge_weight: integer value
    param props_length: length of properties binary string
    return edge_dict: dictionary containing set of Boolean valued edge properties
    """
    formatter = '0' + str(props_length) + 'b'
    # Convert edge weight integer to binary string
    b = format(edge_weight, formatter)
    edge_dict = {}
    edge_dict['is_party'] = b[0]
    edge_dict['is_third_party'] = b[1]
    edge_dict['other'] = b[2]
    return edge_dict

def get_agreement_cosignatories(agreement_ids,data_dict):
    """
    Given a list of agreements get the signatories in common
    Works within a peace process only
    param agreement_ids: List of agreement IDs
    param data_dict: A data dictionary
    return: List of actor IDs who a signatories to all the agreements in agreement_ids
    """
    if len(agreement_ids) < 2:        
        return []
    for agreement_id in agreement_ids:
        if not agreement_id in data_dict['agreement_ids']:
            return []
    agreement_indices = [data_dict['agreement_ids'].index(agreement_id) for\
                         agreement_id in agreement_ids]
    for i,agreement_index in enumerate(agreement_indices):
        row = data_dict['matrix'][agreement_index]
        if i == 0:
            actors_bitset = row
        else:
            actors_bitset = np.bitwise_and(actors_bitset,row)
    actor_ids = []
    for index,value in enumerate(actors_bitset): 
        if value == 1:
            actor_ids.append(data_dict['actor_ids'][index])
    return actor_ids

def get_consignatory_agreements_from_data_dict(actor_ids,data_dict):
    """
    Given a list of actors get the agreements in common form the entire data set
    param actor_ids: List of actor IDs
    param data_dict: Data dictionary
    return: List of agreements to which the actors in actor_ids are cosignatories
    """
    # Given a list of actors get the agreements in common
    if len(actor_ids) < 2:        
        return []
    for actor_id in actor_ids:
        if not actor_id in data_dict['actor_vertices']:
            return []
    actor_indices = [data_dict['actor_vertices'].index(actor_id) for actor_id in actor_ids]
    for i,actor_index in enumerate(actor_indices):
        row = data_dict['matrix'].T[actor_index]
        if i == 0:
            agreements_bitset = row
        else:
            agreements_bitset = np.bitwise_and(agreements_bitset,row)
    agreement_ids = []
    for index,value in enumerate(agreements_bitset): 
        if value == 1:
            agreement_ids.append(data_dict['agreement_vertices'][index])
    return agreement_ids


def get_consignatory_agreements(actor_ids,data_dict):
    """
    Given a list of actors get the agreements in common
    Works within a peace process only
    param actor_ids: List of actor IDs
    param data_dict: Data dictionary
    return: List of agreements to which the actors in actor_ids are cosignatories
    """
    # Given a list of actors get the agreements in common
    # Binarise to discard edge weights
    matrix = (data_dict['matrix'].T > 0).astype(np.int64)
    if len(actor_ids) < 1:        
        return []
    for actor_id in actor_ids:
        if not actor_id in data_dict['actor_ids']:
            return []
    actor_indices = [data_dict['actor_ids'].index(actor_id) for actor_id in actor_ids]
    for i,actor_index in enumerate(actor_indices):
        row = matrix[actor_index]
        if i == 0:
            agreements_bitset = row
        else:
            agreements_bitset = np.bitwise_and(agreements_bitset,row)
    agreement_ids = []
    for index,value in enumerate(agreements_bitset): 
        if value == 1:
            agreement_ids.append(data_dict['agreement_ids'][index])
    return agreement_ids

def get_consignatories(actor_id,data_dict):
    """
    Get the cosignatories of an actor
    Works within a peace process only
    param actor_id: Actor ID
    param data_dict: A data dictionary
    return: List of actors who are cosignatories with the actor in actor_id
    """
    co_matrices = get_cooccurrence_matrices(data_dict['matrix'])
    actor_index = data_dict['actor_ids'].index(actor_id)
    cosign_ids = [data_dict['actor_ids'][i] for i,v in enumerate(co_matrices[0][actor_index]) if v > 0]
    return cosign_ids

def get_coagreements(agreement_id,data_dict):
    """
    Get the coagreements of an agreement, i.e., the agreements that have signatories in 
    common with the agreement in agreement_id
    Works within a peace process only
    param agreement_id: agreement ID
    param data_dict: A data dictionary
    return: List of agreements with actors in common with the agreement in agreement_id
    """
    co_matrices = get_cooccurrence_matrices(data_dict['pmatrix'])
    agreement_index = data_dict['agreement_ids'].index(agreement_id)
    coagree_ids = [data_dict['agreement_ids'][i] for\
                   i,v in enumerate(co_matrices[1][agreement_index]) if v > 0]
    return coagree_ids

def get_agreements(actor_id,data_dict):
    """
    Get the agreements to which an actor is a signatory
    Works within a peace process only
    param actor_id: Actor ID
    param data_dict: A data dictionary
    return: List of agreements to which the actor in actor_id is a signatory
    """
    actor_index = data_dict['actor_ids'].index(actor_id)
    agreement_ids = [data_dict['agreement_ids'][i] for\
                     i,v in enumerate(data_dict['matrix'][:,actor_index]) if v > 0]
    return agreement_ids

def get_actors(agreement_id,data_dict):
    """
    Get the actors who are signatories to the agreement in agreement_id
    Works within a peace process only
    param agreement_id: agreement ID
    param data_dict: A data dictionary
    return: List of actors who a signatories to the agreement in agreement_id
    """
    agreement_index = data_dict['agreement_ids'].index(agreement_id)
    actor_ids = [data_dict['actor_ids'][i] for\
                     i,v in enumerate(data_dict['matrix'][agreement_index]) if v > 0]
    return actor_ids

def get_actor_name(actor_id,data_dict):
    """
    Get the name of an actor
    param actor_id: actor ID
    param data_dict: Global data dictionary
    return: Name of actor
    """
    return data_dict['vertices_dict'][actor_id][data_dict['nodes_header'].index('node_name')]

def get_agreement_name(agreement_id,data_dict):
    """
    Get the name of an agreement
    param agreement_id: agreement ID
    param data_dict: Global data dictionary
    return: Name of agreement
    """
    return data_dict['vertices_dict'][agreement_id][data_dict['nodes_header'].index('node_name')]

def get_agreement_date(agreement_id,data_dict):
    """
    Get the date of an agreement
    param agreement_id: agreement ID
    param data_dict: Global data dictionary
    return: Name of agreement
    """
    return data_dict['vertices_dict'][agreement_id][data_dict['nodes_header'].index('date')]
