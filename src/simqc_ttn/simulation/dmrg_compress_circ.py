import numpy as np
import copy
import string
import opt_einsum as oe
import simqc_ttn.tree_networks.graph_ten_util as g_util
import simqc_ttn.simulation.linalg_utils as ut
import cotengra as ctg

class DMRGCompressor:
    def __init__(self, netw, bra_paths):
        """
        netw: The tensor network object to compress.
        bra_paths: Paths dictionary for bra nodes.
        """
        self.netw = netw
        self.bra_paths = bra_paths

    def sweep(self, num_sweeps=4):
        bra_layers = list(g_util.flatten(self.netw.nodes_ls[-1]))
        all_paths = self.bra_paths
        compress_fid = []

        for sweep_idx in range(num_sweeps):
            print(f"DMRG sweep {sweep_idx+1}/{num_sweeps}")
            if sweep_idx % 2 == 0:
                for i in range(len(bra_layers)):
                    curr_node = bra_layers[i]
                    if i < len(bra_layers) - 1:
                        nxt_node = bra_layers[i+1]
                        curr_node.tensor, fid = self.opt_node(curr_node)
                        compress_fid.append(fid)
                        self.iso_towards(curr_node, nxt_node)
                    else:
                        curr_node.tensor, fid = self.opt_node(curr_node)
                        compress_fid.append(fid)
            else:
                for i in reversed(range(len(bra_layers))):
                    curr_node = bra_layers[i]
                    if i > 0:
                        nxt_node = bra_layers[i-1]
                        curr_node.tensor, fid = self.opt_node(curr_node)
                        compress_fid.append(fid)
                        self.iso_towards(curr_node, nxt_node)
                    else:
                        curr_node.tensor, fid = self.opt_node(curr_node)
                        compress_fid.append(fid)
        return compress_fid[-1]

   
    def opt_node(self, node):
        letters = string.ascii_lowercase
        tmp_dic = copy.deepcopy(self.netw.einsum_dic)
        node_leg_lett = tmp_dic[node.node_id]
        del tmp_dic[node.node_id]

        einsum_parts = []
        tens_ls = []
        for key, indices in tmp_dic.items():
            einsum_parts.append(indices)
            tens_ls.append(self.netw.nodes_dic[key].tensor)
        
        einsum_string = ','.join(einsum_parts) + '->' + node_leg_lett

        opt = ctg.ReusableHyperOptimizer(
            methods=["greedy", "random-greedy"],
            max_repeats=16,
            progbar=False,
            parallel=True        
        )

        F = oe.contract(einsum_string,*tens_ls,optimize= opt,memory_limit=12e9)
        # F = oe.contract(einsum_string, *tens_ls, optimize='auto')

        indices = letters[:len(F.shape)]
        einsum_par_fid = f"{indices},{indices}->"
        par_fid = oe.contract(einsum_par_fid, F, F.conj())
        A = F.conj() / np.sqrt(par_fid)

        return A, par_fid

    @staticmethod
    def absorb_R(R, M, M_leg):
        lett = string.ascii_letters
        M_lett = list(lett[:len(M.shape)])
        R_lett = list(lett[len(M.shape):len(M.shape)+len(R.shape)])
        R_lett[1] = M_lett[M_leg]
        res_lett = copy.deepcopy(M_lett)
        res_lett[M_leg] = R_lett[0]
        ein_str = f"{''.join(R_lett)},{''.join(M_lett)}->{''.join(res_lett)}"
        return oe.contract(ein_str, R, M)

    def iso_towards(self, curr_orth_cen, nxt_orth_cen):
        path = self.bra_paths.get(curr_orth_cen.node_id, {}).get(nxt_orth_cen.node_id)
        path_tensors = [self.netw.nodes_dic[node].tensor for node in path]

        for i in range(len(path)-1):
            leg_curr = self.netw.show_edged_legs(self.netw.nodes_dic[path[i]], self.netw.nodes_dic[path[i+1]])
            leg_nxt = self.netw.show_edged_legs(self.netw.nodes_dic[path[i+1]], self.netw.nodes_dic[path[i]])
            qr_mode = leg_curr[0]
            M_leg = leg_nxt[0]

            Q, R = ut.qr_dec(path_tensors[i], qr_mode)
            path_tensors[i] = Q
            self.netw.nodes_dic[path[i]].tensor = path_tensors[i]

            path_tensors[i+1] = self.absorb_R(R, path_tensors[i+1], M_leg)
            self.netw.nodes_dic[path[i+1]].tensor = path_tensors[i+1]
