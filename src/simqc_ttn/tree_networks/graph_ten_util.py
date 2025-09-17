def remove_key_from_nest_dic(node_dic,node_to_remove):
        # Removes node_to_remove from the edge network dictionary and also removes it from other node's edge lists. 
        node_to_remove_edges=node_dic[node_to_remove.node_id]
        del node_dic[node_to_remove.node_id]
        for node in node_to_remove_edges:
            node_dic[node].remove(node_to_remove.node_id)
            if not node_dic[node] : del node_dic[node]

def swap_list_vals(ls1,idx1,ls2,idx2):
      for i, j in zip(idx1, idx2):
            ls2[j] = ls1[i]

def rand_circ_netw_gen():
      circ = 0
      TN = 0

      for layer_id in range(len(circ)-1):
            if layer_id%2 == 0:
                  if layer_id+2 < len(circ):
                        TN.form_edge(circ[layer_id][0],[2],circ[layer_id+2][0],[0])
                        TN.form_edge(circ[layer_id][-1],[3],circ[layer_id+2][-1],[1])
                  for gate_id,gate in enumerate(circ[layer_id+1]):
                        TN.form_edge(gate,[0],circ[layer_id][gate_id],[3])
                        TN.form_edge(gate,[1],circ[layer_id][gate_id+1],[2])
                  else:
                        for gate_id,gate in enumerate(circ[layer_id+1]):
                              TN.form_edge(gate,[0],circ[layer_id][gate_id],[3])
                              TN.form_edge(gate,[1],circ[layer_id][gate_id+1],[2])            
            else:
                  for gate_id,gate in enumerate(circ[layer_id]):
                        TN.form_edge(gate,[2],circ[layer_id+1][gate_id],[1])
                        TN.form_edge(gate,[3],circ[layer_id+1][gate_id+1],[0])

      for layer in circ:
            for node in layer:
                  print(node.node_id,node.edged_legs,TN.edge_net[node.node_id])    

def flatten(nested_list):
    for item in nested_list:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item