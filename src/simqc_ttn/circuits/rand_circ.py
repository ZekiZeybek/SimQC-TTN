import simqc_ttn.tree_networks.binary_tree_netw as bttn
from simqc_ttn.tree_networks.graph_ten import TNode, TNetwork
import simqc_ttn.tree_networks.graph_ten_util as g_util
import simqc_ttn.simulation.linalg_utils as ut
from .circuit import Circuit
import numpy as np

class RandomCircuit(Circuit):
    def __init__(self, num_qubits, depth, num_layers,chi_bra):
        super().__init__(num_qubits, depth,chi_bra)
        self.num_layers = num_layers
        self.circ_network = None
        self.circ_paths = None
        self.build()
       
    def build(self):
        ket_layers = bttn.down_state_btree_layers(bttn.btree_layers(self.chi_ket, 'ket'))
        bra_layers = bttn.rand_iso_btree_layers(bttn.btree_layers(self.chi_bra, 'bra'))
        circ_layers = self.form_circ_layers(self.num_layers, self.num_qubits)
        self.circ_network, self.circ_paths = self.rand_circ_btree_netw(circ_layers, ket_layers, bra_layers)

    def form_circ_layers(self,num_of_layers, num_of_qbts):
        circ_layers = []
        for layer in range(num_of_layers):
            if layer % 2 == 0:
                circ_layers.append([TNode(4, f'c_l{layer}_{i}', [2,2,2,2], False) for i in range(num_of_qbts//2)])
            else:
                circ_layers.append([TNode(4, f'c_l{layer}_{i}', [2,2,2,2], False) for i in range(num_of_qbts//2 - 1)])
        for layer in circ_layers:
            for gate_node in layer:
                gate_node.tensor = ut.rand_gate()
        return circ_layers
    
    def connect_circuit_layers(self, netw, circ):
        for layer_id in range(len(circ)-1):
            if layer_id % 2 == 0:
                if layer_id + 2 < len(circ):
                    netw.form_edge(circ[layer_id][0], [2], circ[layer_id+2][0], [0])
                    netw.form_edge(circ[layer_id][-1], [3], circ[layer_id+2][-1], [1])
                    for gate_id, gate in enumerate(circ[layer_id+1]):
                        netw.form_edge(gate, [0], circ[layer_id][gate_id], [3])
                        netw.form_edge(gate, [1], circ[layer_id][gate_id+1], [2])
                else:
                    for gate_id, gate in enumerate(circ[layer_id+1]):
                        netw.form_edge(gate, [0], circ[layer_id][gate_id], [3])
                        netw.form_edge(gate, [1], circ[layer_id][gate_id+1], [2])            
            else:
                for gate_id, gate in enumerate(circ[layer_id]):
                    netw.form_edge(gate, [2], circ[layer_id+1][gate_id], [1])
                    netw.form_edge(gate, [3], circ[layer_id+1][gate_id+1], [0])
    
    def connect_ket_to_circuit(self, netw, ket_bttn_layers, circ):
        for qbt_node, qbt_gate in zip(ket_bttn_layers[-1], circ[0]):
            if qbt_node.rank == 2:
                netw.form_edge(qbt_node, [0,1], qbt_gate, [0,1])
            else:
                netw.form_edge(qbt_node, [1,2], qbt_gate, [0,1])

    def connect_bra_to_circuit(self, netw, bra_bttn_layers, circ, circuit_layer_size): 
        if circuit_layer_size%2 == 0:
            for idx_qbt_node,qbt_node in enumerate(bra_bttn_layers[-1][::-1]):
                if idx_qbt_node == 0:
                    netw.form_edge(qbt_node,[1],circ[-2][0],[2])
                    netw.form_edge(qbt_node,[2],circ[-1][0],[2])
                    
                elif idx_qbt_node == len(bra_bttn_layers[-1])-1:
                    netw.form_edge(qbt_node,[2],circ[-2][-1],[3])
                    netw.form_edge(qbt_node,[1],circ[-1][-1],[3])

                else:
                    netw.form_edge(qbt_node,[1],circ[-1][idx_qbt_node-1],[3])
                    netw.form_edge(qbt_node,[2],circ[-1][idx_qbt_node],[2])
        
        else:
            for qbt_node,qbt_gate in zip(bra_bttn_layers[-1],circ[-1]):
                if qbt_node.rank == 2:
                    netw.form_edge(qbt_node,[0,1],qbt_gate,[2,3])
                else:
                    netw.form_edge(qbt_node,[1,2],qbt_gate,[2,3])

    def rand_circ_btree_netw(self, circ, ket_bttn_layers, bra_bttn_layers):
        netw = TNetwork()
        netw.add_nodes(ket_bttn_layers)
        netw.add_nodes(circ)
        netw.add_nodes(bra_bttn_layers)
        netw.add_to_nodes_dic()
        circuit_layer_size = len(circ)

        self.connect_circuit_layers(netw, circ)
        self.connect_ket_layers(netw, ket_bttn_layers)
        self.connect_bra_layers(netw, bra_bttn_layers)
        all_bra_paths = self.collect_bra_paths(netw, bra_bttn_layers)
        self.connect_ket_to_circuit(netw, ket_bttn_layers, circ)
        self.connect_bra_to_circuit(netw, bra_bttn_layers, circ, circuit_layer_size)

        netw.einsum_dic = netw.gen_ctr_str_static()
        return netw, all_bra_paths

    def update_rand_netw(self,netw): 
        for layer_ket, layer_bra in zip(netw.nodes_ls[0],netw.nodes_ls[-1]):
            for node_ket, node_bra in zip(layer_ket,layer_bra):
                node_ket.tensor = np.conjugate(node_bra.tensor)

        for layer in netw.nodes_ls[1]:
            for gate in layer:
                gate.tensor = ut.rand_gate()   

def main():
    num_qubits = 4
    depth = 1
    num_layers = 1
    chi_bra = [[4,4],[4,2,2]]  

    rc = RandomCircuit(num_qubits, depth, num_layers, chi_bra)

    network = rc.get_network()
    bra_paths = rc.get_paths()

    print("Network built:", network)
    print("Bra paths:", bra_paths)

if __name__ == "__main__":
    main()