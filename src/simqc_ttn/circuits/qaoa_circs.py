import simqc_ttn.tree_networks.binary_tree_netw as bttn
from simqc_ttn.tree_networks.graph_ten import TNode, TNetwork
from .circuit import Circuit
import numpy as np
import math

class QAOACircuit(Circuit):
    '''
    QAOA Circuit class inheriting from the base Circuit class.
    It constructs a QAOA circuit tensor network based on the provided graph edges and parameters.
    The circuit can be built in two modes: 'standard' and 'flexible'.
    'standard' mode constructs a fixed QAOA layer with random angles.
    'flexible' mode allows for dynamic updates of the circuit layers.
    As of now only Ising type of cost function is implemented.
    Attributes:
    - num_qubits: Number of qubits in the circuit.
    - depth: Depth of the circuit.
    - chi_bra: Bond dimension for the bra layer.
    - mode: Mode of the circuit ('standard' or 'flexible').
    - graph_edges: Edges of the graph for the QAOA circuit (two-qubit gate connectivity).
    - circ_network: Tensor network representation of the circuit.
    - circ_paths: Paths for the bra layer in the circuit.
    '''

    def __init__(self, num_qubits, depth, chi_bra, mode='standard', graph_edges=None):
        super().__init__(num_qubits, depth, chi_bra)
        self.mode = mode
        self.graph_edges = graph_edges
        self.circ_network = None
        self.circ_paths = None
        self.build()
    
    def form_circuit_layers(self,seed=42):
        ''' Forming a single QAOA layer with random angles
        '''
        # np.random.seed(seed)

        weight = [1]*len(self.graph_edges) # Setting all weights, beta and gamma to 1
        beta = [1]*self.num_qubits
        gamma = [1]*len(self.graph_edges)

        # weight = [1]*len(self.graph_edges) # Setting all weights to 1
        # beta = np.random.uniform(0, np.pi, self.num_qubits)
        # gamma = np.random.uniform(0, 2*np.pi, len(self.graph_edges))

        ZZ_nodes = self.ZiZj_nodes(edges=self.graph_edges,weights=weight,gammas=gamma)
        X_nodes = self.Xj_nodes(num_qbts=self.num_qubits,beta=beta)
        Had_nodes = self.H_nodes(num_qbts=self.num_qubits)
        circ = [Had_nodes,ZZ_nodes,X_nodes]   

        return circ

    def build(self):
        if self.mode == 'standard':
            ket_layers = bttn.down_state_btree_layers(bttn.btree_layers(self.chi_ket, 'ket'))
            bra_layers = bttn.rand_iso_btree_layers(bttn.btree_layers(self.chi_bra, 'bra'))
            circ_layers = self.form_circuit_layers()
            self.circ_network, self.circ_paths = self.qaoa_circ_btree_netw(self.graph_edges, circ_layers, ket_layers, bra_layers)
        
        elif self.mode == 'flexible':
           print("Flexible mode selected. Circuit layers will be updated dynamically.")
        else:
            raise ValueError("Mode not recognized. Use 'standard' or 'flexible'.")

    def qaoa_circ_btree_netw(self, edges, circ_nodes, ket_nodes, bra_nodes):
        netw = TNetwork()
        netw.add_nodes(ket_nodes)
        netw.add_nodes(circ_nodes)
        netw.add_nodes(bra_nodes)
        netw.add_to_nodes_dic()

        ZZ_nodes = circ_nodes[1]
        X_nodes = circ_nodes[2]
        Had_nodes = circ_nodes[0]
        H_len = len(Had_nodes)
        X_len = len(X_nodes)

        edges = self.prepare_edges(edges)
        first_occurrence_indices = self.get_first_occurrences(edges)

        self.connect_ket_layers(netw, ket_nodes)
        self.connect_bra_layers(netw, bra_nodes)
        all_bra_paths = self.collect_bra_paths(netw, bra_nodes)

        self.connect_ket_to_hadamard(netw, ket_nodes, Had_nodes, H_len)
        self.connect_ZZ_H_X(netw, edges, ZZ_nodes, Had_nodes, X_nodes, first_occurrence_indices)
        self.connect_bra_to_X(netw, bra_nodes, X_nodes, H_len, X_len)

        netw.einsum_dic = netw.gen_ctr_str_static()
        return netw, all_bra_paths

    def update_QAOA_netw(self,netw,seed=42):
        # np.random.seed(seed)

        # for ZZ_gate in netw.nodes_ls[1][1]:
        #     gamma  =  np.random.uniform(0, 2*np.pi, 1).item() 
        #     weight = 1
        #     ZZ_gate.tensor = self.ZiZj(gamma=gamma,weight=weight)

        # for X_gate in netw.nodes_ls[1][2]:
        #     beta = np.random.uniform(0, np.pi, 1).item() 
        #     X_gate.tensor = self.Pauli_X(beta=beta)
        for layer_ket, layer_bra in zip(netw.nodes_ls[0],netw.nodes_ls[-1]):
                for node_ket, node_bra in zip(layer_ket,layer_bra):
                    node_ket.tensor = np.conjugate(node_bra.tensor)

        if self.mode == 'standard':     
            for had_node in netw.nodes_ls[1][0]:
                had_node.tensor = np.eye(2)
        
    @staticmethod
    def prepare_edges(edges):
        edges = np.array(edges)
        return edges[edges[:, 0].argsort()]

    @staticmethod
    def get_first_occurrences(edges):
        _, first_occurrence_indices = np.unique(edges[:, 1], return_index=True)
        return sorted(first_occurrence_indices.tolist())

    def connect_ket_to_hadamard(self, netw, ket_nodes, Had_nodes, H_len):
        for i in range(H_len // 2):
            if H_len == 2:
                print("Too small circuit forget about it")
            else:
                netw.form_edge(ket_nodes[-1][i], [1], Had_nodes[2 * i], [0])
                netw.form_edge(ket_nodes[-1][i], [2], Had_nodes[2 * i + 1], [0])

    def connect_ZZ_H_X(self, netw, edges, ZZ_nodes, Had_nodes, X_nodes, first_occurrence_indices):
        for i in range(edges.shape[0]):
            # Hadamard connections
            if i == 0:
                netw.form_edge(ZZ_nodes[i], [0], Had_nodes[edges[0, 0]], [1])
            else:
                if edges[i, 0] != edges[i - 1, 0] and 0 in ZZ_nodes[i].open_legs:
                    netw.form_edge(ZZ_nodes[i], [0], Had_nodes[edges[i, 0]], [1])
            # Second qubit Hadamard
            if i in first_occurrence_indices:
                netw.form_edge(ZZ_nodes[i], [1], Had_nodes[edges[i, 1]], [1])
                first_occurrence_indices.remove(i)
            # X connections
            if i == edges.shape[0] - 1:
                netw.form_edge(ZZ_nodes[i], [2], X_nodes[edges[i, 0]], [0])
                netw.form_edge(ZZ_nodes[i], [3], X_nodes[edges[i, 1]], [0])
            else:
                if edges[i, 0] == edges[i + 1, 0]:
                    netw.form_edge(ZZ_nodes[i], [2], ZZ_nodes[i + 1], [0])
                else:
                    netw.form_edge(ZZ_nodes[i], [2], X_nodes[edges[i, 0]], [0])
                for j in range(i + 1, edges.shape[0]):
                    if edges[i, 1] == edges[j, 1]:
                        netw.form_edge(ZZ_nodes[i], [3], ZZ_nodes[j], [1])
                        break
                    elif edges[i, 1] == edges[j, 0]:
                        netw.form_edge(ZZ_nodes[i], [3], ZZ_nodes[j], [0])
                        break
                    elif j == edges.shape[0] - 1:
                        netw.form_edge(ZZ_nodes[i], [3], X_nodes[edges[i, 1]], [0])
                        break

    def connect_bra_to_X(self, netw, bra_nodes, X_nodes, H_len, X_len):

        for i in range(X_len // 2):
            if H_len == 2:
                print("Too small circuit forget about it")
            else:
                netw.form_edge(bra_nodes[-1][i], [1], X_nodes[2 * i], [1])
                netw.form_edge(bra_nodes[-1][i], [2], X_nodes[2 * i + 1], [1])

    @staticmethod
    def ZiZj(gamma=1, weight=0):
        Pauli_ZZ = np.zeros((2, 2, 2, 2), dtype=complex)
        for i in range(2):
            for j in range(2):
                phase = np.exp(-1j * weight * gamma * ((-1) ** (i + j)))
                Pauli_ZZ[i, j, i, j] = phase
        return Pauli_ZZ

    @staticmethod
    def ZiZj_nodes(edges, weights, gammas):
        nodes = []
        for edge_id, edge in enumerate(edges):
            node = TNode(rank=4, node_id='Z{}Z{}'.format(edge[0], edge[1]), leg_dims=[2, 2, 2, 2])
            node.tensor = QAOACircuit.ZiZj(gamma=gammas[edge_id], weight=weights[edge_id])
            nodes.append(node)
        return nodes

    @staticmethod
    def Pauli_X(beta=1):
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        phase = np.exp(-0.5j * beta)
        return phase * sigma_x

    @staticmethod
    def Xj_nodes(num_qbts, beta):
        nodes = []
        for i in range(num_qbts):
            ang = beta[i]
            node = TNode(rank=2, node_id='X{}'.format(i), leg_dims=[2, 2])
            node.tensor = QAOACircuit.Pauli_X(beta=ang)
            nodes.append(node)
        return nodes

    @staticmethod
    def Hadamard():
        Had = np.array([[1, 1], [1, -1]], dtype=complex)
        return (1 / np.sqrt(2)) * Had

    @staticmethod
    def H_nodes(num_qbts):
        nodes = []
        for i in range(num_qbts):
            node = TNode(rank=2, node_id='H{}'.format(i), leg_dims=[2, 2])
            node.tensor = QAOACircuit.Hadamard()
            nodes.append(node)
        return nodes
    
  
    def form_init_QAOA_circ(self, edges, circ_nodes, ket_nodes, bra_nodes):
        """
        Build the initial QAOA circuit tensor network.
        """
        netw = TNetwork()
        netw.add_nodes(ket_nodes)
        netw.add_nodes(circ_nodes)
        netw.add_nodes(bra_nodes)
        ZZ_nodes = circ_nodes[1]
        Had_nodes = circ_nodes[0]
        netw.add_to_nodes_dic()
        H_len = int(len(Had_nodes))

        edges = self.prepare_edges(edges)
        first_occurrence_indices = self.get_first_occurrences(edges)

        self.connect_ket_layers(netw, ket_nodes)
        self.connect_bra_layers(netw, bra_nodes)
        all_bra_paths = self.collect_bra_paths(netw, bra_nodes)

        for i in range(int(H_len/2)):
            if H_len == 2:
                print("Too small circuit forget about it")
            else:
                netw.form_edge(ket_nodes[-1][i],[1],Had_nodes[2*i],[0])
                netw.form_edge(ket_nodes[-1][i],[2],Had_nodes[2*i+1],[0])

        
        for i in range(edges.shape[0]):
            if i == 0:
                netw.form_edge(ZZ_nodes[i], [0], Had_nodes[edges[0, 0]], [1])
                netw.form_edge(ZZ_nodes[i], [1], Had_nodes[edges[0, 1]], [1])
            else:
                if edges[i, 0] != edges[i-1, 0] and 0 in ZZ_nodes[i].open_legs:
                    netw.form_edge(ZZ_nodes[i], [0], Had_nodes[edges[i, 0]], [1])
        
            if i in first_occurrence_indices and i!=0:
                netw.form_edge(ZZ_nodes[i], [1], Had_nodes[edges[i, 1]], [1])
                first_occurrence_indices.remove(i)

    
            if i == edges.shape[0] - 1:        
                qbt_idx_2 = edges[i, 0]
                bra_leg_2 = 1 + (qbt_idx_2 % 2)
                qbt_idx_3 = edges[i, 1]
                bra_leg_3 = 1 + (qbt_idx_3 % 2)
                netw.form_edge(ZZ_nodes[i], [2], bra_nodes[-1][math.floor(qbt_idx_2/2)], [bra_leg_2])
                netw.form_edge(ZZ_nodes[i], [3], bra_nodes[-1][math.floor(qbt_idx_3/2)], [bra_leg_3])

            else:
                if edges[i, 0] == edges[i+1, 0]:
                    netw.form_edge(ZZ_nodes[i], [2], ZZ_nodes[i+1], [0])
                else:
                    qbt_idx = edges[i, 0]
                    bra_leg = 1 + (qbt_idx % 2)
                    netw.form_edge(ZZ_nodes[i], [2], bra_nodes[-1][math.floor(qbt_idx/2)], [bra_leg])

                for j in range(i+1, edges.shape[0]):
                    if edges[i, 1] == edges[j, 1]:
                        netw.form_edge(ZZ_nodes[i], [3], ZZ_nodes[j], [1])
                        break
                    elif edges[i, 1] == edges[j, 0]:
                        netw.form_edge(ZZ_nodes[i], [3], ZZ_nodes[j], [0])
                        break
                    elif j == edges.shape[0] - 1:
                        qbt_idx = edges[i, 1]
                        bra_leg = 1 + (qbt_idx % 2)
                        netw.form_edge(ZZ_nodes[i], [3], bra_nodes[-1][math.floor(qbt_idx/2)], [bra_leg])
                        break

        for Had_id,Had in enumerate(Had_nodes): 
            if Had.open_legs:
                qbt_idx = Had_id
                bra_leg = 1 + (qbt_idx % 2) 
                netw.form_edge(Had, [1], bra_nodes[-1][math.floor(qbt_idx/2)], [bra_leg])

        netw.merge_duplicate_edges()
        netw.einsum_dic = netw.gen_ctr_str_static()

        return netw,all_bra_paths

    def form_intermed_QAOA_circ(self, edges, circ_nodes, ket_nodes, bra_nodes):

        netw = TNetwork()
        netw.add_nodes(ket_nodes)
        netw.add_nodes(circ_nodes)
        netw.add_nodes(bra_nodes)
        ZZ_nodes = circ_nodes[0]
        netw.add_to_nodes_dic()

        edges = self.prepare_edges(edges)
        first_occurrence_indices = self.get_first_occurrences(edges)

        self.connect_ket_layers(netw, ket_nodes)
        self.connect_bra_layers(netw, bra_nodes)
        all_bra_paths = self.collect_bra_paths(netw, bra_nodes)

        for i in range(edges.shape[0]):
            if i == 0:
                qbt_idx_0 = edges[i, 0]
                ket_leg_0 = 1 + (qbt_idx_0 % 2)
                qbt_idx_1 = edges[i, 1]
                ket_leg_1 = 1 + (qbt_idx_1 % 2)
                netw.form_edge(ZZ_nodes[i], [0], ket_nodes[-1][math.floor(qbt_idx_0/2)], [ket_leg_0])
                netw.form_edge(ZZ_nodes[i], [1], ket_nodes[-1][math.floor(qbt_idx_1/2)], [ket_leg_1])

            else:
                if edges[i, 0] != edges[i-1, 0] and 0 in ZZ_nodes[i].open_legs:
                    qbt_idx = edges[i, 0]
                    ket_leg = 1 + (qbt_idx % 2)
                    netw.form_edge(ZZ_nodes[i], [0], ket_nodes[-1][math.floor(qbt_idx/2)], [ket_leg])
        
            if i in first_occurrence_indices and i!=0:
                qbt_idx = edges[i, 1]
                ket_leg = 1 + (qbt_idx % 2)
                netw.form_edge(ZZ_nodes[i], [1], ket_nodes[-1][math.floor(qbt_idx/2)], [ket_leg])
                first_occurrence_indices.remove(i)
            
            if i == edges.shape[0] - 1:
                qbt_idx_2 = edges[i, 0]
                bra_leg_2 = 1 + (qbt_idx_2 % 2)
                qbt_idx_3 = edges[i, 1]
                bra_leg_3 = 1 + (qbt_idx_3 % 2)
                netw.form_edge(ZZ_nodes[i], [2], bra_nodes[-1][math.floor(qbt_idx_2/2)], [bra_leg_2])
                netw.form_edge(ZZ_nodes[i], [3], bra_nodes[-1][math.floor(qbt_idx_3/2)], [bra_leg_3])

            else:
                if edges[i, 0] == edges[i+1, 0]:
                    netw.form_edge(ZZ_nodes[i], [2], ZZ_nodes[i+1], [0])
                else:
                    qbt_idx = edges[i, 0]
                    bra_leg = 1 + (qbt_idx % 2)
                    netw.form_edge(ZZ_nodes[i], [2], bra_nodes[-1][math.floor(qbt_idx/2)], [bra_leg])

                for j in range(i+1, edges.shape[0]):
                    if edges[i, 1] == edges[j, 1]:
                        netw.form_edge(ZZ_nodes[i], [3], ZZ_nodes[j], [1])
                        break
                    elif edges[i, 1] == edges[j, 0]:
                        netw.form_edge(ZZ_nodes[i], [3], ZZ_nodes[j], [0])
                        break
                    elif j == edges.shape[0] - 1:
                        qbt_idx = edges[i, 1]
                        bra_leg = 1 + (qbt_idx % 2)
                        netw.form_edge(ZZ_nodes[i], [3], bra_nodes[-1][math.floor(qbt_idx/2)], [bra_leg])
                        break
        
        for ket_qbt_node, bra_qbt_node in zip(ket_nodes[-1],bra_nodes[-1]):
            if ket_qbt_node.open_legs:
                netw.form_edge(ket_qbt_node, ket_qbt_node.open_legs, bra_qbt_node,  bra_qbt_node.open_legs)
        
        netw.merge_duplicate_edges()
        netw.einsum_dic = netw.gen_ctr_str_static()
    
        return netw,all_bra_paths
   
    def form_final_QAOA_circ(self, edges, circ_nodes, ket_nodes, bra_nodes):
        netw = TNetwork()
        netw.add_nodes(ket_nodes)
        netw.add_nodes(circ_nodes)
        netw.add_nodes(bra_nodes)
        ZZ_nodes = circ_nodes[0]
        X_nodes =  circ_nodes[1]
        X_len = int(len(X_nodes))
        netw.add_to_nodes_dic()

        edges = self.prepare_edges(edges)
        first_occurrence_indices = self.get_first_occurrences(edges)

        self.connect_ket_layers(netw, ket_nodes)
        self.connect_bra_layers(netw, bra_nodes)
        all_bra_paths = self.collect_bra_paths(netw, bra_nodes)
        
        for i in range(edges.shape[0]):
            if i == 0:
                qbt_idx_0 = edges[i, 0]
                ket_leg_0 = 1 + (qbt_idx_0 % 2)
                qbt_idx_1 = edges[i, 1]
                ket_leg_1 = 1 + (qbt_idx_1 % 2)
                netw.form_edge(ZZ_nodes[i], [0], ket_nodes[-1][math.floor(qbt_idx_0/2)], [ket_leg_0])
                netw.form_edge(ZZ_nodes[i], [1], ket_nodes[-1][math.floor(qbt_idx_1/2)], [ket_leg_1])
            
            else:
                if edges[i, 0] != edges[i-1, 0] and 0 in ZZ_nodes[i].open_legs:
                    qbt_idx = edges[i, 0]
                    ket_leg = 1 + (qbt_idx % 2)
                    netw.form_edge(ZZ_nodes[i], [0], ket_nodes[-1][math.floor(qbt_idx/2)], [ket_leg])
        
            if i in first_occurrence_indices and i!=0:
                qbt_idx = edges[i, 1]
                ket_leg = 1 + (qbt_idx % 2)
                netw.form_edge(ZZ_nodes[i], [1], ket_nodes[-1][math.floor(qbt_idx/2)], [ket_leg])
                first_occurrence_indices.remove(i)
            
            if i == edges.shape[0] - 1:
                netw.form_edge(ZZ_nodes[i], [2], X_nodes[edges[i, 0]], [0])
                netw.form_edge(ZZ_nodes[i], [3], X_nodes[edges[i, 1]], [0])

            else:
                if edges[i, 0] == edges[i+1, 0]:
                    netw.form_edge(ZZ_nodes[i], [2], ZZ_nodes[i+1], [0])
                else:
                    netw.form_edge(ZZ_nodes[i], [2], X_nodes[edges[i, 0]], [0])

                for j in range(i+1, edges.shape[0]):
                    if edges[i, 1] == edges[j, 1]:
                        netw.form_edge(ZZ_nodes[i], [3], ZZ_nodes[j], [1])
                        break
                    elif edges[i, 1] == edges[j, 0]:
                        netw.form_edge(ZZ_nodes[i], [3], ZZ_nodes[j], [0])
                        break
                    elif j == edges.shape[0] - 1:
                        netw.form_edge(ZZ_nodes[i], [3], X_nodes[edges[i, 1]], [0])
                        break
        
        for i in range(int(X_len/2)):
            if X_len == 2:
                print("Too small circuit forget about it")
            else:
                netw.form_edge(bra_nodes[-1][i],[1],X_nodes[2*i],[1])
                netw.form_edge(bra_nodes[-1][i],[2],X_nodes[2*i+1],[1])


        for X_gate_id, X_gate_node in enumerate(X_nodes):
            if X_gate_node.open_legs:
                if X_gate_id%2 == 0:
                    k = int(X_gate_id/2)
                    netw.form_edge(ket_nodes[-1][k],[1],X_nodes[X_gate_id],[0])
                else:
                    k = int((X_gate_id-1)/2)
                    netw.form_edge(ket_nodes[-1][k],[2],X_nodes[X_gate_id],[0])

        netw.merge_duplicate_edges()
        netw.einsum_dic = netw.gen_ctr_str_static()

        return netw,all_bra_paths


    def build_flexible_layers(self, edges, num_qbts, N2g_per_comp, seed=42):
        """
        Partition the QAOA circuit into chunks for flexible construction.
        Returns: 
            - list of (circ_nodes, edge_chunk) for each chunk in each layer
        """
        np.random.seed(seed)
        weight = [1] * len(edges)
        beta = np.random.uniform(0, np.pi, num_qbts)
        gamma = np.random.uniform(0, 2 * np.pi, len(edges))

        ZZ_nodes = self.ZiZj_nodes(edges=edges, weights=weight, gammas=gamma)
        X_nodes = self.Xj_nodes(num_qbts=num_qbts, beta=beta)
        Had_nodes = self.H_nodes(num_qbts=num_qbts)
        N2g = len(ZZ_nodes)
        num_comp_part = math.floor(N2g / N2g_per_comp)
        resid = N2g - (num_comp_part * N2g_per_comp)

        if resid == 0:
            ZZ_parts = [ZZ_nodes[i * N2g_per_comp:N2g_per_comp * (i + 1)] for i in range(num_comp_part)]
            edge_parts = [edges[i * N2g_per_comp:N2g_per_comp * (i + 1)] for i in range(num_comp_part)]
        else:
            ZZ_parts = [ZZ_nodes[i * N2g_per_comp:N2g_per_comp * (i + 1)] for i in range(num_comp_part)]
            ZZ_parts.append(ZZ_nodes[-resid:])
            edge_parts = [edges[i * N2g_per_comp:N2g_per_comp * (i + 1)] for i in range(num_comp_part)]
            edge_parts.append(edges[-resid:])

        circ_chunks = []
        for i in range(len(ZZ_parts)):
            if i == 0:
                circ_chunks.append(([Had_nodes, ZZ_parts[i]], edge_parts[i]))
            elif i == len(ZZ_parts) - 1:
                circ_chunks.append(([ZZ_parts[i], X_nodes], edge_parts[i]))
            else:
                circ_chunks.append(([ZZ_parts[i]], edge_parts[i]))
        return circ_chunks

    def build_layer_chunks(self, edges, num_qbts, N2g_per_comp, num_QAOA_layer, include_hadamard=False, seed=42):
        """
        Build multiple QAOA layers with flexible chunking.
        Returns:
            - list of layers, each containing (circ_nodes, edge_chunk) for each chunk
        """
        # np.random.seed(42)
        # weight = [1]*len(self.graph_edges) # Setting all weights to 1
        # beta = np.random.uniform(0, np.pi, self.num_qubits)
        # gamma = np.random.uniform(0, 2*np.pi, len(self.graph_edges))

        weight = [1]*len(edges) # Setting all weights, beta and gamma to 1
        beta = [1]*num_qbts
        gamma = [1]*len(edges)
        ZZ_nodes = self.ZiZj_nodes(edges=edges,weights=weight,gammas=gamma)
        X_nodes = self.Xj_nodes(num_qbts=num_qbts,beta=beta)
        Had_nodes = self.H_nodes(num_qbts=num_qbts)

        N2g = len(ZZ_nodes)
        num_comp_part= math.floor(N2g/N2g_per_comp)
        resid = N2g-(num_comp_part*N2g_per_comp)

        if resid == 0:
            ZZ_parts= [ZZ_nodes[i*N2g_per_comp:N2g_per_comp*(i+1)] for i in range(num_comp_part)]
            edge_parts = [edges[i*N2g_per_comp:N2g_per_comp*(i+1)] for i in range(num_comp_part)]
        else:
            ZZ_parts= [ZZ_nodes[i*N2g_per_comp:N2g_per_comp*(i+1)] for i in range(num_comp_part)]
            res = ZZ_nodes[-resid:]
            ZZ_parts.append(res)
            edge_parts = [edges[i*N2g_per_comp:N2g_per_comp*(i+1)] for i in range(num_comp_part)]
            res_edges = edges[-resid:]
            edge_parts.append(res_edges)

        circ_chunks = []
        for i in range(len(ZZ_parts)):
            if i == 0 and include_hadamard:
                circ_chunks.append(([Had_nodes, ZZ_parts[i]], edge_parts[i]))
            elif i == len(ZZ_parts) - 1:
                circ_chunks.append(([ZZ_parts[i], X_nodes], edge_parts[i]))
            else:
                circ_chunks.append(([ZZ_parts[i]], edge_parts[i]))
        
        return circ_chunks

        
