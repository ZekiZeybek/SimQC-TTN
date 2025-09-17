from .graph_ten import TNode, TNetwork
import numpy as np
import copy
import opt_einsum as oe
from . import graph_ten_util as g_util
import simqc_ttn.simulation.linalg_utils as ut

def btree_layers(chi_ls,node_id_str):
    """
    Generate layers of nodes for a binary tree to be used for forming TTN network with appropriate edge formations
    """
    depth = len(chi_ls)
    node_ls = []
    for layer in range(depth):
        if layer == 0:
            node_ls.append([TNode(2, '{}_l{}_b{}'.format(node_id_str,layer,branch), chi_ls[layer], False) for branch in range(int(2**layer))])
        else:
            node_ls.append([TNode(3, '{}_l{}_b{}'.format(node_id_str,layer,branch), chi_ls[layer], False) for branch in range(int(2**layer))])

    return node_ls

def balanced_tree_layers(layer_ls=[1,2,4],chi_ls=[16,4,2],rank_ls=[2,3,3],node_id_str='tree'):
    depth = len(chi_ls)
    node_ls = []
    num_of_qbts = layer_ls[-1]*(rank_ls[-1]-1)
    print(num_of_qbts)
    node_ls.append([TNode(rank_ls[0], '{}_l{}_b{}'.format(node_id_str,0,branch), [chi_ls[0] for k in range(rank_ls[0])], False) for branch in range(layer_ls[0])])

    for i in range(1,depth):
        rank = rank_ls[i]
        chi_parent = chi_ls[i-1]
        chi_child = [chi_ls[i]]*(rank-1)
        leg_dim = [chi_parent]
        leg_dim.extend(chi_child)
        node_ls.append([TNode(rank, '{}_l{}_b{}'.format(node_id_str,i,branch), leg_dim, False) for branch in range(layer_ls[i])])

    return node_ls

def btree_rand_tensors(chi_ls):
    """Orthogonality center is the top node!"""
    depth = len(chi_ls)
    tens_ls = []
    for layer in range(depth):
        if layer == 0:
            tens_ls.append([ut.rand_norm_ten(chi_ls[layer]) for branch in range(int(2**layer))])
        else:
            tens_ls.append([ut.iso_ten(chi_ls[layer],0) for branch in range(int(2**layer))])
    return tens_ls

def rand_iso_btree_layers(tree_layers):
    for layer_id,layer in enumerate(tree_layers):
        if layer_id == 0:
            for node in layer:
                node.tensor = np.conjugate(ut.rand_norm_ten(node.leg_dims))
        else:
            for node in layer:
                node.tensor = np.conjugate(ut.iso_ten(node.leg_dims,m=0))
    return tree_layers   

def down_state_btree_layers(tree_layers):
    num_layers = len(tree_layers)

    for layer_id, layer in enumerate(tree_layers):
        if layer_id == 0:
            for node in layer:
                node.tensor = np.ones(node.leg_dims)

        elif layer_id == num_layers-1:
            for node in layer:
                node.tensor = np.zeros(node.leg_dims)
                node.tensor[0,0,0] = 1
        else:
            for node in layer:
                node.tensor = np.ones(node.leg_dims)
    
    return tree_layers


def bttn_netw(ket_bttn_layers):
    netw = TNetwork()
    netw.add_nodes(ket_bttn_layers)
    netw.add_to_nodes_dic()
    for layer_id in range(len(ket_bttn_layers)-1):
        num_edge_nodes = [i for i in range(int(2**(layer_id+1)))]
        if layer_id == 0:
            for node_id, node in enumerate(ket_bttn_layers[layer_id]):
                idx = int(2*node_id)
                for i in num_edge_nodes[idx:idx+2]:    
                    netw.form_edge(node,[i],ket_bttn_layers[layer_id+1][i],[0])       
        else:
            for node_id, node in enumerate(ket_bttn_layers[layer_id]):
                idx = int(2*node_id)
                for i in num_edge_nodes[idx:idx+2]:  
                    leg = int(i%2+1)  
                    netw.form_edge(node,[leg],ket_bttn_layers[layer_id+1][i],[0])
    
    netw.einsum_dic = netw.gen_ctr_str_static()
    
    return netw

def calc_norm_bttn(netw):
    """Using the ket_network (netw) to form a bra nodes and layers, then forming a composite network (bra-ket) by adding bra nodes to ket_netw (netw)
    Quick fix: To do that I deep copy the initial netw object such that outside of the function it still stays as a ket_netw rather than turning into a bra-ket netw
    """

    tmp_TN = copy.deepcopy(netw)

    chi_ls = []
    for layer in tmp_TN.nodes_ls[0]:
        chi_ls.append(layer[0].leg_dims)

    bra_tree_layer = btree_layers(chi_ls,'bra')

    for bra_layer, ket_layer in zip(bra_tree_layer,tmp_TN.nodes_ls[0]):
        for node_bra,node_ket in zip(bra_layer,ket_layer):
            node_bra.tensor = np.conjugate(node_ket.tensor)
    
    bra_netw = bttn_netw(bra_tree_layer)

    tmp_TN.nodes_ls = [tmp_TN.nodes_ls]
    tmp_TN.add_nodes(bra_tree_layer)
    tmp_TN.edge_net.update(bra_netw.edge_net)
    tmp_TN.nodes_dic.update(bra_netw.nodes_dic)
    

    for phys_node_ket,phys_node_bra in zip(tmp_TN.nodes_ls[0][0][-1],tmp_TN.nodes_ls[1][-1]):
        tmp_TN.form_edge(phys_node_ket,[1,2],phys_node_bra,[1,2])

    tmp_dic = tmp_TN.gen_ctr_str_static()
    
    einsum_parts = []
    tens_ls = []
    for key, indices in tmp_dic.items():
        einsum_parts.append(indices)
        tens_ls.append(tmp_TN.nodes_dic[key].tensor)
    
    einsum_string = ','.join(einsum_parts) + '->'
    norm = oe.contract(einsum_string,*tens_ls)
    
    return norm

def trace_bttn(netw):
    ket_tensors = []
    ket_tensors_conj = []
    for layer in netw.nodes_ls: 
        tmp_layer=[]
        tmp_layer_conj=[]
        for node in layer:
            tmp_layer.append(copy.deepcopy(node.tensor))
            tmp_layer_conj.append(copy.deepcopy(node.tensor.conj()))  
        ket_tensors.append(tmp_layer)
        ket_tensors_conj.append(tmp_layer_conj)
    
    cpy_einsum_dic = copy.deepcopy(netw.einsum_dic)
    ket_tensors = list(g_util.flatten(ket_tensors))
    ket_tensors_conj = list(g_util.flatten(ket_tensors_conj))

    einsum_parts = []
    for key, indices in cpy_einsum_dic.items():
        einsum_parts.append(indices)    
    
    einsum_string = f"{','.join(einsum_parts)},{','.join(einsum_parts)}->"
    trace = oe.contract(einsum_string,*ket_tensors,*ket_tensors_conj)

    return trace

def trace_tensor(tensor):
    return np.sum(tensor * tensor.conj())

def trace_ovlp_tensor(tensor1,tensor2):
    return np.sum(tensor1 * tensor2.conj())