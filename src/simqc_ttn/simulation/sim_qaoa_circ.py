import numpy as np
import simqc_ttn.tree_networks.binary_tree_netw as bttn
import simqc_ttn.simulation.dmrg_compress_circ as dmrg
import simqc_ttn.tree_networks.binary_tree_netw as bttn
import simqc_ttn.circuits.qaoa_circs as qaoa
import networkx as nx

def chi_bra_layers(chi_layer):
    chi_bra_list = []
    num_layers = len(chi_layer)
    for i in range(num_layers):
        if i == 0:
            chi_bra_list.append([chi_layer[i], chi_layer[i]])
        else:
            chi_bra_list.append([chi_layer[i-1], chi_layer[i], chi_layer[i]])
    return chi_bra_list

def generate_3_regular_graph(num_qbts,seed=1):
    G = nx.random_regular_graph(3, num_qbts,seed=seed)
    graph_dict = {node: list(G.neighbors(node)) for node in G.nodes()}
    edge_list = list(G.edges())
    edges = [sorted(list(edge)) for edge in edge_list]
    return G,edges

def generate_linear_graph(num_qbts):
    G = nx.path_graph(num_qbts)
    graph_dict = {node: list(G.neighbors(node)) for node in G.nodes()}
    edge_list = list(G.edges())
    return G, edge_list

def sim_qaoa_flexible(edges, num_QAOA_layer=1, num_qbts=8, chi_bra=None, N2g_per_comp=8, sweep_per_comp=4):
    assert N2g_per_comp < len(edges), "N2g_per_comp should be less than the total number of 2-qubit gates in a layer"
    
    QAOA = qaoa.QAOACircuit(num_qbts, num_QAOA_layer, chi_bra, 'flexible', edges)
    ket_nodes = bttn.down_state_btree_layers(bttn.btree_layers(QAOA.chi_ket, 'ket'))
    bra_nodes = bttn.rand_iso_btree_layers(bttn.btree_layers(QAOA.chi_bra, 'bra'))

    QAOA_layer_fid = []
    for i in range(num_QAOA_layer):
        print("QAOA Layer: ", i + 1)
        compress_fid = []
        if i == 0:
            circ_chunks = QAOA.build_layer_chunks(edges, num_qbts, N2g_per_comp, num_QAOA_layer, include_hadamard=True)
            for comp_id, (circ_nodes, edge_chunk) in enumerate(circ_chunks):
                if comp_id == 0:
                    print("Initial compressive step with Hadamard and first ZZ gates")
                    netw, bra_paths = QAOA.form_init_QAOA_circ(edge_chunk, circ_nodes, ket_nodes, bra_nodes)
                elif comp_id == len(circ_chunks) - 1:
                    print("Final compressive step with last ZZ and X rotation gates")
                    netw, bra_paths = QAOA.form_final_QAOA_circ(edge_chunk, circ_nodes, ket_nodes, bra_nodes)
                else:
                    print("Intermediate compressive step with only ZZ gates")
                    netw, bra_paths = QAOA.form_intermed_QAOA_circ(edge_chunk, circ_nodes, ket_nodes, bra_nodes)

                compressor = dmrg.DMRGCompressor(netw, bra_paths)
                comp_fid = compressor.sweep(num_sweeps=sweep_per_comp)
                netw.nullify_edge_relats(0)
                netw.nullify_edge_relats(-1)
                QAOA.update_QAOA_netw(netw)
                compress_fid.append(comp_fid.real)
                print(f"Compressive step {comp_id + 1}/{len(circ_chunks)} fidelity: {comp_fid}")
            
        else:
            circ_chunks = QAOA.build_layer_chunks(edges, num_qbts, N2g_per_comp, num_QAOA_layer, include_hadamard=False)
            for comp_id, (circ_nodes, edge_chunk) in enumerate(circ_chunks):
                if comp_id == len(circ_chunks) - 1:
                    print("Final compressive step with last ZZ and X rotation gates")
                    netw, bra_paths = QAOA.form_final_QAOA_circ(edge_chunk, circ_nodes, ket_nodes, bra_nodes)
                else:
                    print("Intermediate compressive step with only ZZ gates")
                    netw, bra_paths = QAOA.form_intermed_QAOA_circ(edge_chunk, circ_nodes, ket_nodes, bra_nodes)

                compressor = dmrg.DMRGCompressor(netw, bra_paths)
                comp_fid = compressor.sweep(num_sweeps=sweep_per_comp)  
                netw.nullify_edge_relats(0)
                netw.nullify_edge_relats(-1)
                QAOA.update_QAOA_netw(netw)
                compress_fid.append(comp_fid.real)
                print(f"Compressive step {comp_id + 1}/{len(circ_chunks)} fidelity: {comp_fid}")
    
        QAOA_layer_fid.append(np.prod(compress_fid))
    
    return netw, QAOA_layer_fid

def sim_qaoa_native(edges,circ_depth=1,num_qbts=8,chi_bra=None,sweep_per_comp=4):
    
    layer_fids = []
    for i in range(circ_depth):
        print('LAYER', i)
        if i == 0:
            QAOA = qaoa.QAOACircuit(num_qbts, circ_depth, chi_bra, 'standard',edges)
            circ_TN,bra_paths = QAOA.circ_network, QAOA.circ_paths
            compressor = dmrg.DMRGCompressor(circ_TN, bra_paths)
            layer_fid= compressor.sweep(num_sweeps=sweep_per_comp)
            layer_fids.append(layer_fid.real)
        else:
            QAOA.update_QAOA_netw(circ_TN)
            layer_fid= compressor.sweep(num_sweeps=sweep_per_comp)
            layer_fids.append(layer_fid.real)

    return circ_TN, np.array(layer_fids)



if __name__ == "__main__":
    
  
    chi_bra = chi_bra_layers([32,16,4,2])
    num_of_qbts = 16
    G, edge_list = generate_3_regular_graph(num_qbts=num_of_qbts,seed=2)
    # G, edge_list = generate_linear_graph(num_qbts=num_of_qbts)
    num_2qbt_gate = len(edge_list)
    print('Num. of 2qbt gates aka num. of edge: ', num_2qbt_gate)
    num_QAOA_layers = 2
    sweeps_per_compress = 6
    N2g_per_comp = 6
    
    dmrg_circ, layer_fids = sim_qaoa_flexible(edges=edge_list,
                                               num_QAOA_layer=num_QAOA_layers,
                                               num_qbts=num_of_qbts,
                                               chi_bra=chi_bra,
                                               N2g_per_comp=N2g_per_comp,
                                               sweep_per_comp=sweeps_per_compress)
    

    
    F_tilde = np.prod(layer_fids)
    total_num_2qbt_gates = num_2qbt_gate * num_QAOA_layers
    print('F_tilde: ',F_tilde)
    


    # dmrg_circ, layer_fids = sim_qaoa_native(edges=edge_list,
    #                                         circ_depth=num_QAOA_layers,
    #                                         num_qbts=num_of_qbts,
    #                                         chi_bra=chi_bra,
    #                                         sweep_per_comp=sweeps_per_compress)
    

    # F_tilde = np.prod(layer_fids)
    # total_num_2qbt_gates = num_2qbt_gate * num_QAOA_layers
    # print('F_tilde: ',F_tilde)
