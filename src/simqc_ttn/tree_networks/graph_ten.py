import numpy as np
import copy
from opt_einsum import parser
from . import graph_ten_util as g_util
from collections import deque
 
class TNode:
    def __init__(self, rank=2, node_id=None, leg_dims=[], random=False):
        self.rank = rank
        self.leg_dims = leg_dims
        self.random = random
        self.node_id = node_id
        self.edge_nodes = []
        self.open_legs = [i for i in range(rank)]
        self.edged_legs = []
        self._tensor = None
        self.leg_indcs = [i for i in range(rank)]

    def __str__(self):
        return f"TNode(node_id={self.node_id}, rank={self.rank}, leg_dim={self.leg_dims})"

    def __repr__(self):
        return f"TNode(node_id={self.node_id}, rank={self.rank}, leg_dim={self.leg_dims})"

    @property
    def tensor(self):
        return self._tensor
    
    @tensor.setter
    def tensor(self, new_tensor):
        if new_tensor is not None and not isinstance(new_tensor, np.ndarray):
            raise TypeError("Tensor must be of NumPy array type")
        
        self._tensor = new_tensor
        if new_tensor is not None:
            self.rank = new_tensor.ndim
            self.leg_dims = list(new_tensor.shape)
            self.leg_indcs = list(range(self.rank)) 

    def remove_edge(self, edge_node):
        self.edge_nodes = [edge for edge in self.edge_nodes if edge is not edge_node]
     
    def show_edges(self):
        pass


class TNetwork:
    def __init__(self,nodes_ls=None):
        self.nodes_ls = nodes_ls if nodes_ls is not None else []
        self.nodes_dic = {}
        self.edge_net = {}
        self.einsum_dic = {}

    def is_edged(self,node1,node2):
        return node2.node_id in self.edge_net[node1.node_id]
   
    def add_to_nodes_dic(self):

        nested_list = self.nodes_ls
        flattened_list = list(g_util.flatten(nested_list))
        for node in flattened_list:
            self.nodes_dic.update({'{}'.format(node.node_id): node})

    def add_to_edge_net(self,node,node2):
        
        if node.node_id in self.edge_net:
            self.edge_net[node.node_id].append(node2.node_id)
        else:
            self.edge_net.setdefault('{}'.format(node.node_id), [])
            self.edge_net[node.node_id].append(node2.node_id)

    def remove_node_from_edge_net(self,node_to_remove):
        # Removes node_to_remove from the edge network dictionary and also removes it from other node's edge lists. 
        node_to_remove_edges=self.edge_net[node_to_remove.node_id]
        del self.edge_net[node_to_remove.node_id]
        for node in node_to_remove_edges:
            self.edge_net[node].remove(node_to_remove.node_id)
            if not self.edge_net[node] : del self.edge_net[node]

    def show_edged_legs(self,node1,node2):
        leg_curr_idx = self.edge_net[node1.node_id].index(node2.node_id)
        leg_curr_node = node1.edged_legs[leg_curr_idx]
        return leg_curr_node

    def check_leg_dims(self,p_node, p_leg, c_node, c_leg):
        p_leg_dims = [p_node.leg_dims[i] for i in p_leg]
        c_leg_dims = [c_node.leg_dims[i] for i in c_leg]

        return p_leg_dims==c_leg_dims

    def form_edge(self, p_node, p_leg, c_node, c_leg):
    
        assert self.check_leg_dims(p_node, p_leg, c_node, c_leg),"Non-equal leg dimensions!"
        assert set(p_leg).issubset(p_node.open_legs), f"Open legs {p_node.open_legs} of the first node ({p_node.node_id}) does not have the leg {p_leg} to be edged with {c_leg} of {c_node.node_id} "
        assert set(c_leg).issubset(c_node.open_legs), f"Open legs {c_node.open_legs} of the first node ({c_node.node_id}) does not have the leg {c_leg} to be edged with {p_leg} of {p_node.node_id}"
    
        p_node.edge_nodes.append(c_node)
        c_node.edge_nodes.append(p_node)

        c_node.open_legs= [i for i in c_node.open_legs if i not in c_leg ]
        c_node.edged_legs.append(c_leg)
        
        p_node.open_legs= [i for i in p_node.open_legs if i not in p_leg ]
        p_node.edged_legs.append(p_leg)
        
        self.add_to_edge_net(p_node,c_node)
        self.add_to_edge_net(c_node,p_node)
        
    def add_nodes(self, node):
        self.nodes_ls.append(node)
    
    def gen_ctr_str_static(self):

        nested_list = self.nodes_ls
        flattened_list = list(g_util.flatten(nested_list))
        netw_idx = {}
     
        for idx,node in enumerate(flattened_list):
            if not netw_idx:
                index_str = "".join([parser.get_symbol(i) for i in range(node.rank)])
                netw_idx.update({'{}'.format(node.node_id):index_str })
            else:
                accum_idx_str = "".join([netw_idx[flattened_list[i].node_id] for i in range(idx)])
                index_str = "".join(parser.gen_unused_symbols(accum_idx_str, node.rank))
                netw_idx.update({'{}'.format(node.node_id):index_str })


        cpy_edge_net = copy.deepcopy(self.edge_net)
        for node in self.nodes_dic:
            if node in cpy_edge_net:                    # IF EDGE NODES have the same edge twice with different legs (contraction over 2 legs) since dictionary is updated after one letter assignments it is deleted and no chance fpr takig
                                                                    #care of the second leg generalzie or tidy up the edge net dic
                for edge_node in cpy_edge_net[node]:
                    
                    leg1_idx = self.edge_net[node].index(edge_node)
                    leg2_idx = self.edge_net[edge_node].index(node)
                    leg_1 = self.nodes_dic[node].edged_legs[leg1_idx]
                    leg_2 = self.nodes_dic[edge_node].edged_legs[leg2_idx]
                   
                    assert self.check_leg_dims(self.nodes_dic[node], leg_1,self.nodes_dic[edge_node], leg_2), "Non-equal leg dimensions!"
                    
                    node_str=list(netw_idx[node])
                    edge_node_str = list(netw_idx[edge_node])
                    g_util.swap_list_vals(node_str,leg_1,edge_node_str,leg_2)
                    netw_idx[node] = "".join(node_str)
                    netw_idx[edge_node] = "".join(edge_node_str)
            
                g_util.remove_key_from_nest_dic(cpy_edge_net,self.nodes_dic[node])

        return netw_idx
    
    def contract_node(self, node1, node2):
        
        leg1_idx = self.edge_net[node1.node_id].index(node2.node_id)
        leg2_idx = self.edge_net[node2.node_id].index(node1.node_id)
        leg_1 = node1.edged_legs[leg1_idx]
        leg_2 = node2.edged_legs[leg2_idx]
     
        assert self.check_leg_dims(node1, leg_1, node2, leg_2), "Non-equal leg dimensions!"

    
        num_of_legs_contrctd = len(leg_1)
        rank_new = node1.rank-num_of_legs_contrctd + node2.rank-num_of_legs_contrctd

        cnode = TNode(rank=rank_new)
        cnode.node_id = '{}_{}'.format(node1.node_id,node2.node_id)


        # index order after the contraction
        # remaining leg dimensions of node1 after removing its legs edged with node2 due to cntrc
        # remaining leg dimensions of node2 after removing its legs edged with node1 due to cntrc
        
        # A_abcdef B_gbhcij = (AB)_adefghij np.tensordot convention is used
        # A.leg_dims = [dim(a),dim(b),dim(c),dim(d),dim(e),dim(f)] ---> A.leg_dims = [dim(a),dim(d),dim(e),dim(f)]
        # B.leg_dims = [dim(g),dim(b),dim(h),dim(c),dim(i),dim(j)] ---> B.leg_dims = [dim(g),dim(h),dim(i),dim(j)]
        # AB.leg_dims = [dim(a),dim(d),dim(e),dim(f)]+[dim(g),dim(h),dim(i),dim(j)] np.tensordot convention!
        node1.leg_dims = [node1.leg_dims[i] for i in range(node1.rank) if i not in leg_1]
        node2.leg_dims = [node2.leg_dims[i] for i in range(node2.rank) if i not in leg_2]
        cnode.leg_dims = node1.leg_dims+node2.leg_dims
        
        # remaining edged legs of node1 after removing its legs edged with node2 due to cntrc
        # remaining edged legs of node2 after removing its legs edged with node1 due to cntrc
        # A.edged_legs=[[1,2],[0],[3,4]] -> A.edged_legs=[[0],[3,4]] leg a is edged and (leg d and leg e) edged
        # B.edged_legs=[[1,3],[0],[2,4]] -> B.edged_legs=[[0],[2,4]] leg g is edged and (leg h and leg i) edged
        node1.edged_legs.remove(leg_1) 
        node2.edged_legs.remove(leg_2) 


        # reamining legs of node1 after removing its legs edged with node2 due to cntrc
        # remaining legs of node2 after removing its legs edged with node1 due to cntrc
        # A.leg_indcs = [0,1,2,3,4,5] 0->a, 1->b, 2->c, 3->d, 4->e, 5->f
        # B.leg_indcs = [0,1,2,3,4,5] 0->g, 1->b, 2->h, 3->c 4->i, 5->j
        # A.leg_indcs = [0,1,2,3,4,5] -> A.leg_indcs = [0,3,4,5] remaining after contracting over b and c (1 and 2)
        # B.leg_indcs = [0,1,2,3,4,5] -> B.leg_indcs = [0,2,4,5] remaining after contracting over b and c (1 and 3)
        # AB.leg_indcs = [0,1,2,3,4,5,6,7] = [0,3,4,5]_A + [0,2,4,5]_B i.e.,
        # 0_AB = 0_A, 1_AB = 3_A, 2_AB = 4_A, 3_AB = 5_A and 
        # 4_AB = 0_B, 5_AB = 2_B, 6_AB = 4_B, 7_AB = 5_B
        node1.leg_indcs = [i for i in node1.leg_indcs if i not in leg_1]
        node2.leg_indcs = [i for i in node2.leg_indcs if i not in leg_2]

        # remaining edged nodes of node1 after removing its edge with node2 due to cntrc
        # remaining edged nodes of node2 after removing its edge with node1 due to cntrc
        node1.edge_nodes = [edge_node for edge_node in node1.edge_nodes if edge_node is not node2]
        node2.edge_nodes = [edge_node for edge_node in node2.edge_nodes if edge_node is not node1]


        # we need to update the leg adresses of the edged nodes of node1 written in node1 leg ordering.
        # we need to write the leg adresses of the edged nodes of node1 in conracted_node leg ordering.
        # Reamining edged legs of the nodes according to their precontraction internal ordering
        # A.edged_legs=[[0],[3,4]] leg a is edged and (leg d and leg e) edged
        # B.edged_legs=[[0],[2,4]] leg g is edged and (leg h and leg i) edged
        # We need to change the above leg adresses with respect to the AB leg ordering.
        # AB.leg_indcs = [0,1,2,3,4,5,6,7] = [0,3,4,5]_A + [0,2,4,5]_B i.e.,
        # 0_AB = 0_A, 1_AB = 3_A, 2_AB = 4_A, 3_AB = 5_A and 
        # 4_AB = 0_B therefore index(B[]) in B.leg_indcs = [0,3,4,5] is wrong, it would yield 0. 
        # it is true that the value zero in B has the index 0 in remaining B legs but not in AB legs. Therefore
        # we need to keep track of how many A legs attached to the AB and start the numbering from the last leg 
        # number. Then 0_B -> len(A.leg_indcs)+index(B[]) in B.leg_indcs = [0,3,4,5]
        # Then it becomes 4+0 for the address of 0_B in AB. 
        # [2,4]_B --> [len(A.leg_indcs)+index(B[2]) in B.leg_indcs = [0,3,4,5]], 
            #          len(A.leg_indcs)+index(B[4]) in B.leg_indcs = [0,3,4,5]]
        # [2,4]_B --> [4+1, 4+2]_AB = [5,6]_AB which is true --> 5_AB = 2_B, 6_AB = 4_B
        # Open legs are legs other than the edged ones!  
         
        for leg in node1.edged_legs:
            l= [node1.leg_indcs.index(i) for i in leg]
            cnode.open_legs= [i for i in cnode.open_legs if i not in l]
            cnode.edged_legs.append(l)
            
        
        for leg in node2.edged_legs:
            l= [len(node1.leg_indcs)+node2.leg_indcs.index(i) for i in leg]
            cnode.open_legs= [i for i in cnode.open_legs if i not in l]
            cnode.edged_legs.append(l)
    

        

        # the ordering in edge_leg list should match the ordering
        # in the edge_node list in the network dictionary. Ex. A.edged_legs[[c leg],[a leg],[b leg]] {'A':[c,a,b]}
        # assert index(c in [c,a,b])== index([c leg] in [[c leg],[a leg],[b leg]]) 
        self.edge_net[node1.node_id].remove(node2.node_id)
        self.edge_net[node2.node_id].remove(node1.node_id)
        self.edge_net.update({cnode.node_id:self.edge_net[node1.node_id]+self.edge_net[node2.node_id]})

        
            
        # the ordering in edge_leg list should match the ordering
        # in the edge_node list in the network dictionary. Ex. A.edged_legs=[[c leg],[a leg],[b leg]] {'A':[c,a,b]}
        # assert index(c in [c,a,b])== index([c leg] in [[c leg],[a leg],[b leg]])         
        # Updating the network and the remaining edge nodes of A and B such that now they are put in edge with AB
        # properly and not A and B separetely. The following convention shoukld be respected:
        # Ordering in the tictionary that holds node edge relationship is the same as 
       # the ordering of the edged_legs.
        #. network before contrac {'A':[a1,a2,B], 'B':[b1,b2,A], 'a1':[A], 'a2':[a3,a4,A],'b1':[B], 'b2':[b3,b4,B] the rest}
        #network after contrac {'AB':[a1,a2,b1,b2],a1':[A], 'a2':[a3,a4,A],'b1':[B], 'b2':[b3,b4,B] rest}
        #since a2 has edge relationship in the order of [[a3],[a4],[A]], we have to put AB to as the
        #last item in the list so that the edged node of a2 can be related to AB when a2 is needed to contr with AB
        #same goes for AB as well edge_leg ordering and edge_node ordering in the network dictionary should match!

        for node in node1.edge_nodes:
            for idx,edge in enumerate(self.edge_net[node.node_id]):
                if edge==node1.node_id:
                    self.edge_net[node.node_id][idx]=cnode.node_id
                    
        for node in node2.edge_nodes:
            for idx,edge in enumerate(self.edge_net[node.node_id]):
                if edge==node2.node_id:
                    self.edge_net[node.node_id][idx]=cnode.node_id


        # the ordering in edge_leg list should match the ordering
        # in the edge_node list in the network dictionary. Ex. A.edged_legs[[c leg],[a leg],[b leg]] {'A':[c,a,b]}
        # this is the ordering of the  edge_nodes as well. Example below

        # A.edged_legs=[[c leg],[a leg],[b leg]] {'A':[c,a,b]} A.edge_nodes = [c_node, a_node, b_node]
        # For AB then AB.edged_legs=[[a1 leg],[a2 leg],[b1 leg],[b2 leg]]  {'AB':[a1,a2,b1,b2] ...} AB.edge_nodes = [a1_node,a2_node,b1_node,b2_node]
        # Therefore all ordering follows the np.tensordot convention ----> attributes of the contracted tensor(node) = remaining attributes of first tensor(node) + remaining attributes of second tensor(node)
        # remaining edged nodes of node1 are added to the contracted node list 
        # remaining edged nodes of node2 are added to the contracted node list
        cnode.edge_nodes.extend(node1.edge_nodes)
        cnode.edge_nodes.extend(node2.edge_nodes)
        
    
        # Updating the edged nodes of A (node1), basically removing node1 (A) and adding AB instead. Since now edged nodes of A must be edged with AB
        for node in node1.edge_nodes:
            node.edge_nodes.remove(node1)
            node.edge_nodes.append(cnode)
            
        # Updating the edged nodes of B (node2), basically removing node2 (B) and adding AB instead. Since now edged nodes of B must be edged with AB
        for node in node2.edge_nodes:
            node.edge_nodes.remove(node2)
            node.edge_nodes.append(cnode)

        # Removing A and B from the network dictionary since they no longer exists, AB has been already added
        del self.edge_net[node1.node_id]
        del self.edge_net[node2.node_id]
        node1.edge_nodes=[]
        node2.edge_nodes=[]
       
        return cnode
    
    def con_TN(self,node):
        if self.edge_net[node.node_id] == []:
            return node
        else:
            node_edges = node.edge_nodes 
            con_node = self.contract_node(node,node_edges[0])
            print(self.edge_net)
          
            return self.con_TN(con_node)
    
    def compute_all_shortest_paths(self):
        """
        Computes the shortest path from every node to every other reachable node in the graph.
        """
        graph = copy.deepcopy(self.edge_net)
        all_paths = {}
        
        for start in graph:
            paths = {start: [start]}
            queue = deque([start])
            
            while queue:
                current = queue.popleft()
                
                for neighbor in graph.get(current, []):
                    if neighbor not in paths:
                        paths[neighbor] = paths[current] + [neighbor]
                        queue.append(neighbor)
                        
            all_paths[start] = paths
            
        return all_paths

    def merge_duplicate_edges(self):
        """
        Update the edge network to merge duplicate edges and update edged_legs accordingly.
        """
        updated_edge_net = {}
        for key, edges in self.edge_net.items():
            unique_edges = {}
            for idx, edge in enumerate(edges):
                leg = self.nodes_dic[key].edged_legs[idx]
                if edge in unique_edges:
                    unique_edges[edge].extend(leg)
                else:
                    unique_edges[edge] = list(leg)
            updated_edge_net[key] = list(unique_edges.keys())
            self.nodes_dic[key].edged_legs = list(unique_edges.values())
        self.edge_net = updated_edge_net

    def nullify_edge_relats(self,layer_idx):
        for layer in self.nodes_ls[layer_idx]:
            for node in layer:
                node.edged_legs = []
                node.open_legs = [i for i in range(node.rank)]
                node.leg_indcs = [i for i in range(node.rank)]
                
        return self.nodes_ls[layer_idx]


if __name__ == "__main__":
    
    
    M1=TNode(4, 'M1', [4,4,4,4], False)
    M2=TNode(3, 'M2', [4,4,4], False)
    M3=TNode(2, 'M3', [4,4], False)
    TN = TNetwork() 
    TN.add_nodes([M1,M2,M3])
    TN.add_to_nodes_dic()
    TN.form_edge(p_node=M1,p_leg=[1],c_node=M2,c_leg=[0])
    TN.form_edge(p_node=M2,p_leg=[1],c_node=M3,c_leg=[0])
    print(TN.edge_net)
    all_paths = TN.compute_all_shortest_paths()
    # print(all_paths)
    print("Shortest path from 'M1' to 'M3':", all_paths.get('M1', {}).get('M3'))


    # print(TN.gen_ctr_str_static())


    # # two_gate = TNode(4, '2qb', [2,2,2,2,2], False)
    # # L1 = []
    # # for i in range(6):
    # #     tmp_node_id="2qb_{}".format(i)
    # #     tmp = TNode(4,tmp_node_id,[2,2,2,2,False])
    # #     L1.append(tmp)


    # # TN = TNetwork()
    # # TN.nodes_ls.append([M1,M2,M3])
    # # TN.nodes_ls.append(L1)
    # # print(len(TN.nodes_ls))

    # # for i in range(len(TN.nodes_ls[0])-1):
    # #     tmp_node1=TN.nodes_ls[0][i]
    # #     tmp_node2=TN.nodes_ls[0][i+1]
    # #     tmp_node1_leg = tmp_node1.leg_indcs[-1]
    # #     tmp_node2_leg = tmp_node2.leg_indcs[0]
    # #     TN.form_edge(p_node=tmp_node1,p_leg=[tmp_node1_leg],c_node=tmp_node2,c_leg=[tmp_node2_leg])

    
    # # print(TN.edge_net)
   

    # l1 = TNode(8, 'l1', [8,9,1,10,2,25,27,26], False)
    # l2 = TNode(5, 'l2', [12,11,9,13,10], False)
    # l3 = TNode(1, 'l3', [11], False)
    # l4 = TNode(1, 'l4', [13], False)
    # l5 = TNode(4, 'l5', [25,33,26,27], False)

    # m1 = TNode(9, 'm1', [3,1,5,2,4,21,24,22,23], False)
    # m2 = TNode(5, 'm2', [4,7,6,5,20], False)
    # m3 = TNode(1, 'm3', [6], False)
    # m4 = TNode(1, 'm4', [7], False)
    # m5 = TNode(6, 'm5', [21,23,31,22,24,32], False)


    # TN = TNetwork() 
    # print(TN.edge_net)

    


 
    # TN.add_nodes([l1,l2,l3,l4,l5,m1,m2,m3,m4,m5])
    # TN.add_to_nodes_dic()
    # # print(TN.nodes_dic)


    
    

    # TN.form_edge(p_node=l1,p_leg=[1,3],c_node=l2,c_leg=[2,4])
    # TN.form_edge(p_node=l2,p_leg=[1],c_node=l3,c_leg=[0])
    # TN.form_edge(p_node=l2,p_leg=[3],c_node=l4,c_leg=[0])
    # TN.form_edge(p_node=l1,p_leg=[5,7,6],c_node=l5,c_leg=[0,2,3])

    # TN.form_edge(p_node=l1,p_leg=[2,4],c_node=m1,c_leg=[1,3])
    # TN.form_edge(p_node=m1,p_leg=[2,4],c_node=m2,c_leg=[3,0])
    # TN.form_edge(p_node=m2,p_leg=[2],c_node=m3,c_leg=[0])
    # TN.form_edge(p_node=m2,p_leg=[1],c_node=m4,c_leg=[0])
    # TN.form_edge(p_node=m1,p_leg=[5,7,8,6],c_node=m5,c_leg=[0,3,1,4])

    # print(TN.edge_net)
    # print(TN.gen_ctr_str_static())

    # print(TN.edge_net)
    # TN.remove_node_from_edge_net(l1)
    # print(TN.edge_net)

    # print(TN.edge_net)
    # TN.contract_node(node1=l1,node2=l2)
    # print(TN.edge_net)

   
    # all = TN.con_TN(l1)
    # print(l1.open_legs,l1.edged_legs,l1.leg_dims)
    # print(TN.edge_net)
    # print(TN.nodes_ls)
   
    # a = TN.con_TN(TN.nodes_ls[0])
    # print(l1.open_legs,l1.edged_legs,l1.leg_dims)
    # print("")


    # l1_m1= TN.contract_node(node1=l1,node2=m1)
    # print(l1_m1.open_legs,l1_m1.edged_legs,l1_m1.leg_dims,l1_m1.edge_nodes)
    # print(TN.edge_net)

    # print("")
    # print(TN.edge_net)
    # l1m1l5=TN.contract_node(node1=l1_m1,node2=l5)
    # print("")
    # print(l1m1l5.open_legs,l1m1l5.edged_legs,l1m1l5.leg_dims)
    # print(TN.edge_net)
    # print("")

    # l1m1l5m5=TN.contract_node(node1=l1m1l5,node2=m5)
    # print("")
    # print(l1m1l5m5.open_legs,l1m1l5m5.edged_legs,l1m1l5m5.leg_dims)
    # print(TN.edge_net)
    # print("")

    # l1m1l5m5m2=TN.contract_node(node1=l1m1l5m5,node2=m2)
    # print("")
    # print(l1m1l5m5m2.open_legs,l1m1l5m5m2.edged_legs,l1m1l5m5m2.leg_dims)
    # print(TN.edge_net)
    # print("")

    # l1m1l5m5m2l2=TN.contract_node(node1=l1m1l5m5m2,node2=l2)
    # print("")
    # print(l1m1l5m5m2l2.open_legs,l1m1l5m5m2l2.edged_legs,l1m1l5m5m2l2.leg_dims)
    # print(TN.edge_net)
    # print("")

    # l1m1l5m5m2l2m3=TN.contract_node(node1=l1m1l5m5m2l2,node2=m3)
    # l1m1l5m5m2l2m3m4=TN.contract_node(node1=l1m1l5m5m2l2m3,node2=m4)
    # l1m1l5m5m2l2m3m4l3=TN.contract_node(node1=l1m1l5m5m2l2m3m4,node2=l3)
    # l1m1l5m5m2l2m3m4l3l4=TN.contract_node(node1= l1m1l5m5m2l2m3m4l3,node2=l4)
    # print(TN.edge_net)
    # print(l1m1l5m5m2l2m3m4l3l4.open_legs,l1m1l5m5m2l2m3m4l3l4.edged_legs,l1m1l5m5m2l2m3m4l3l4.leg_dims)
    # print(l1m1l5m5m2l2m3m4l3l4.edge_nodes)



#     l1m1l5l2=TN.contract_node(node1=l1m1l5,node2=m5)
