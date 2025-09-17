import numpy as np
import string
import opt_einsum as oe
from . import linalg_utils as ut
import simqc_ttn.tree_networks.binary_tree_netw as bttn
import simqc_ttn.simulation.dmrg_compress_circ as dmrg
import simqc_ttn.circuits.rand_circ as rcs


def sim_rand(circ_depth=1,num_of_layers=2,num_of_qbts=8,chi_bra=None,num_sweeps=4):
    """
    One full random circuit depth = 2 layers = 1 odd layer + 1 even layer of random two-qubit gates
    Odd-even ordering is used
    """
    layer_fids = []
    for i in range(circ_depth):
        print("Random circuit depth: ",i+1)
        if i == 0:
            RC = rcs.RandomCircuit(num_of_qbts, circ_depth, num_of_layers, chi_bra)
            circ_TN,bra_paths = RC.circ_network,RC.circ_paths
            compressor = dmrg.DMRGCompressor(circ_TN, bra_paths)
            layer_fid= compressor.sweep(num_sweeps=num_sweeps)
            print('Depth fidelity: ',layer_fid)
            layer_fids.append(layer_fid.real)
        else:
            RC.update_rand_netw(circ_TN)
            layer_fid=compressor.sweep(num_sweeps=num_sweeps)
            print("Depth fidelity: ",layer_fid)
            layer_fids.append(layer_fid.real)

    return circ_TN, np.array(layer_fids)

def chi_bra_layers(chi_layer):
    chi_bra_list = []
    num_layers = len(chi_layer)
    for i in range(num_layers):
        if i == 0:
            chi_bra_list.append([chi_layer[i], chi_layer[i]])
        else:
            chi_bra_list.append([chi_layer[i-1], chi_layer[i], chi_layer[i]])
    return chi_bra_list


if __name__ == "__main__":
    

    chi_bra =chi_bra_layers([32,16,4,2])
    num_of_qbts = 16
    circ_depth =8
    num_sweeps = 6


    dmrg_state,layer_fids = sim_rand(circ_depth=circ_depth,
                                     num_of_qbts=num_of_qbts,
                                     chi_bra=chi_bra,
                                     num_sweeps=num_sweeps)
   
    F_tilde = np.prod(layer_fids)
    print('F_tilde: ',F_tilde)




























    raise SystemExit("Stopping execution here")

    dmrg_state_vec = state_vector_DMRG_TTN_Circ(dmrg_state)
    print(exact_state_vec.shape,dmrg_state_vec.shape)
    
    letters = string.ascii_lowercase
    ovl_letters = letters[:len(exact_state_vec.shape)]
    ovl_letters = f"{ovl_letters},{ovl_letters}->"

    ovlp = oe.contract(ovl_letters,exact_state_vec,dmrg_state_vec.conj(),optimize= 'auto')
    print(abs(ovlp))

    raise SystemExit("Stopping execution here")



    # raise SystemExit("Stopping execution here")
    print("This line will NOT be executed")
    ket_bttn_layers = rand_iso_btree_layers(btree_layers(chi_bra,'ket'))
    KET = bttn_netw(ket_bttn_layers)
   
    
    
    # print(oe.contract('ab,ab->',KET.nodes_dic['ket_l0_b0'].tensor,KET.nodes_dic['ket_l0_b0'].tensor.conj()))


    all_paths = all_shortest_paths(KET.edge_net)
    print(trace_bttn(KET))
    print(trace_tensor(KET.nodes_dic['ket_l0_b0'].tensor))
    
    iso_towards(KET,KET.nodes_dic['ket_l0_b0'],KET.nodes_dic['ket_l2_b0'])
    print(trace_bttn(KET))
    print(trace_tensor(KET.nodes_dic['ket_l0_b0'].tensor))
    print(trace_tensor(KET.nodes_dic['ket_l2_b0'].tensor))



    iso_towards(KET,KET.nodes_dic['ket_l2_b0'],KET.nodes_dic['ket_l2_b1'])
    print(trace_bttn(KET))
    print(trace_tensor(KET.nodes_dic['ket_l2_b0'].tensor))
    print(trace_tensor(KET.nodes_dic['ket_l2_b1'].tensor))



    # is_Q_isom(KET.nodes_dic['ket_l2_b1'].tensor,m=2)






























    # bra_tensors = []
    # ket_tensors = []
    # for layer in bra_bttn_layers:
    #     tmp_layer=[]
    #     for node in layer:
    #         tmp_layer.append(copy.deepcopy(node.tensor))
    #     bra_tensors.append(tmp_layer)

    # for layer in ket_bttn_layers:
    #     tmp_layer=[]
    #     for node in layer:
    #         tmp_layer.append(copy.deepcopy(node.tensor))
    #     ket_tensors.append(tmp_layer)


    # ein = copy.deepcopy(TN.einsum_dic)
    # node_ten = TN.nodes_dic['ket_l0_b0'].tensor
    # node_ten = rand_gate()
    # TN.nodes_dic['ket_l0_b0'].tensor = rand_gate()
    # print(node_ten.shape)
    # print(TN.nodes_dic['ket_l0_b0'].tensor.shape)
    # del ein['ket_l0_b0'] 
    # print(ein)
    # print(TN.einsum_dic) if it is not list it does not refer to the same ndarray no need to deep copy