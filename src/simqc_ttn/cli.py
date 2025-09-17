import argparse
import json
import numpy as np
import os
from datetime import datetime
import pickle
from simqc_ttn.simulation.sim_qaoa_circ import (
    sim_qaoa_flexible,
    sim_qaoa_native,
    generate_3_regular_graph,
    generate_linear_graph,
    chi_bra_layers ,
)
from simqc_ttn.simulation.sim_rand_circ import sim_rand 

def parse_edges(s: str):
    edges = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        a, b = item.split("-")
        edges.append((int(a), int(b)))
    return edges

def build_parser():
    p = argparse.ArgumentParser(prog="simqc-ttn", description="SimQC-TTN")
    p.add_argument("--out", type=str, default="", help="Output directory name specification for the simlation")

    sub = p.add_subparsers(dest="circuit", required=True)
    q = sub.add_parser("qaoa", help="Run QAOA simulation ")
    q.add_argument("--mode", choices=["flex", "native"], required=True, help="QAOA mode")
    q.add_argument("--num_qbts", type=int, required=True, help="Number of qubits")
    q.add_argument("--layers", type=int, default=1, help="Number of QAOA layers")
    grp = q.add_mutually_exclusive_group(required=True)
    grp.add_argument("--edges", type=str, help='User defined edge list for the two-qubit graph connectivity. Comma-separated edges like "0-1,1-2,2-3"')
    grp.add_argument("--default_graph", choices=["linear", "reg3"], help="Generate a built-in graph")
    q.add_argument("--graph_seed", type=int, default=1, help="Seed for graph generation (required using --default_graph) for reproducibility")
    q.add_argument("--chi", type=str, default=None, help="Bond dimension list for the variational state, seperated like 32,16,4,2")
    q.add_argument("--N2g-per-comp", type=int, default=8, dest="N2g_per_comp", help="(required only in flex mode) number of 2-qubit gates per compressive step")
    q.add_argument("--sweeps", type=int, default=4, dest="sweep_per_comp", help="DMRG sweeps per compression step in a single QAOA layer")

    rd = sub.add_parser("rand", help="Run random circuit simulation")
    rd.add_argument("--layers", type=int, default=2, required=True, help="Number of even-odd layers to be applied. This determines the circuit depth as twice the number of layers since each layer consists of either even or odd two-qubit gates")
    rd.add_argument("--num_qbts", type=int, default=8, required=True, help="Number of qubits")
    rd.add_argument("--chi", type=str, default=None, help="Bond dimension list for the variational state, seperated like 32,16,4,2")
    rd.add_argument("--sweeps", type=int, default=4, dest="num_sweeps", required=True, help="DMRG sweeps per random circuit layer")

    return p

def resolve_graph(args):
    if getattr(args, "edges", None):
        return parse_edges(args.edges)
    if getattr(args, "default_graph", None):
        if args.default_graph == "linear":
            G, edge_list = generate_linear_graph(args.num_qbts)
        else:
            G, edge_list = generate_3_regular_graph(args.num_qbts, seed=args.graph_seed)
        return edge_list
    return None

def parse_chi(chi_str):
    if chi_str is None:
        return None
    try:
        layers = [int(x.strip()) for x in chi_str.split(",") if x.strip()]
        return chi_bra_layers(layers)
    except Exception as e:
        raise SystemExit(f"Invalid --chi value, see help: {chi_str!r} ({e})")

def save_json_result(data, prefix, out="", network_obj=None):
    now = datetime.now().strftime("%Y%m%d_%H%M")
    if out:
        dir_name = f"{prefix}_{out}_{now}"
    else:
        dir_name = f"{prefix}_{now}"
    out_dir = os.path.join("results", dir_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "result.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {out_path}")
    if network_obj is not None:
        save_pickle_object(network_obj, out_dir)

def save_pickle_object(obj, out_dir, filename="final_state_network.pkl"):
    out_path = os.path.join(out_dir, filename)
    with open(out_path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Network object saved to {out_path}")

def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)

    def make_jsonable(x):
        if isinstance(x, (np.generic,)):
            x = x.item()
        if isinstance(x, complex):
            return {"real": x.real, "imag": x.imag}
        return x

    if args.circuit == "qaoa":
        edges = resolve_graph(args)
        chi_bra =parse_chi(args.chi)

        if args.mode == "flex":
            dmrg_circ, layer_fids = sim_qaoa_flexible(
                edges=edges,
                num_QAOA_layer=args.layers,
                num_qbts=args.num_qbts,
                chi_bra=chi_bra,
                N2g_per_comp=args.N2g_per_comp,
                sweep_per_comp=args.sweep_per_comp
            )
        else:  
            dmrg_circ, layer_fids = sim_qaoa_native(
                edges=edges,
                circ_depth=args.layers,  
                num_qbts=args.num_qbts,
                chi_bra=chi_bra,
                sweep_per_comp=args.sweep_per_comp
            )

        layer_fids_json = [make_jsonable(fid) for fid in layer_fids]
        sim_fids = np.prod(layer_fids)
        result = {
            "config": vars(args),
            "layer_fidelities": layer_fids_json,
            "total_fidelity": make_jsonable(sim_fids)
        }
        save_json_result(result, "qaoa", out=args.out, network_obj=dmrg_circ)

    elif args.circuit == "rand":
        chi_bra = parse_chi(args.chi)
        dmrg_circ, layer_fids = sim_rand(
            circ_depth=args.layers,
            num_of_qbts=args.num_qbts,
            chi_bra=chi_bra,
            num_sweeps=args.num_sweeps
        )
        layer_fids_json = [make_jsonable(fid) for fid in layer_fids]
        sim_fids = np.prod(layer_fids)
        result = {
            "config": vars(args),
            "layer_fidelities": layer_fids_json,
            "total_fidelity": make_jsonable(sim_fids)
        }
        save_json_result(result, "rand", out=args.out, network_obj=dmrg_circ)
if __name__ == "__main__":
    main()
