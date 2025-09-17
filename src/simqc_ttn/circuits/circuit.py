import numpy as np
import simqc_ttn.tree_networks.graph_ten_util as g_util
import simqc_ttn.simulation.linalg_utils as ut

class Circuit:
    def __init__(self, num_qubits, depth, chi_bra):
        self.num_qubits = num_qubits
        self.depth = depth
        self.chi_bra = chi_bra
        self.chi_ket = self.form_chi_ket(num_qubits)
        self.circ_network = None
        self.circ_paths = None

    def build(self):
        """
        Build the tensor network for the circuit; actually only implemented in subclasses.
        This is more of a placeholder here. I might play with abstract base class stuff, maybe it is more appropriate
        """
        raise NotImplementedError

    def get_network(self):
        return self.circ_network
    
    def get_paths(self):
        return self.circ_paths
    
    @staticmethod
    def form_chi_ket(num_qubits):
        num_tree_layers = int(np.log2(num_qubits))
        num_mid_tree_layers = num_tree_layers - 2
        chi_ket = [[1,1]] + [[1,1,1] for _ in range(num_mid_tree_layers)] + [[1,2,2]]
        return chi_ket
    
    def connect_ket_layers(self, netw, ket_bttn_layers):   
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

    def connect_bra_layers(self, netw, bra_bttn_layers):
        for layer_id in range(len(bra_bttn_layers)-1):
            num_edge_nodes = [i for i in range(int(2**(layer_id+1)))]
            if layer_id == 0:
                for node_id, node in enumerate(bra_bttn_layers[layer_id]):
                    idx = int(2*node_id)
                    for i in num_edge_nodes[idx:idx+2]:    
                        netw.form_edge(node,[i],bra_bttn_layers[layer_id+1][i],[0])       
            else:
                for node_id, node in enumerate(bra_bttn_layers[layer_id]):
                    idx = int(2*node_id)
                    for i in num_edge_nodes[idx:idx+2]:  
                        leg = int(i%2+1)  
                        netw.form_edge(node,[leg],bra_bttn_layers[layer_id+1][i],[0])

    def collect_bra_paths(self, netw, bra_bttn_layers):
        bra_keys = []
        bra_vals = []
        for node in list(g_util.flatten(bra_bttn_layers)):
            bra_keys.append(node.node_id)
            bra_vals.append(netw.edge_net[node.node_id])
        my_dict = dict(zip(bra_keys, bra_vals))
        all_bra_paths = ut.all_shortest_paths(my_dict)
        return all_bra_paths
    

    

