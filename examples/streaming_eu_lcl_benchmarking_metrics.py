#!/usr/bin/env python 
import os, sys, csv
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sei_lora.dataloaders import VariantDataset, SeqDataLoader, SeqDataset, VariantDataLoader
from sei_lora.score import get_celltype_asssy_specific, get_sequence_class_scores_and_max 
from tqdm import tqdm
import scipy
from sklearn.metrics import average_precision_score, matthews_corrcoef

import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.special import expit
from sklearn.metrics import average_precision_score, matthews_corrcoef, f1_score, roc_auc_score
import seimodel as sm
import seillra as sl
import torch.nn as nn
import torch


import os, sys
from typing import Optional, Literal

class SeiWrapper(nn.Module):
    def __init__(self, k: int, ft: Optional[str] = None, projection: bool = True, mode: Literal["sequence", "variant"] = "sequence", device: str = "cpu"):
        super().__init__()
        self.device = device
        self.mode = mode
        self.projection = projection
        self.head = sm.get_sei_head().load_weights()
        self.trunk = sm.get_sei_trunk().load_weights()
        
        if self.projection:
            self.proj = sm.get_sei_projection().load_weights()
            self.proj.set_mode(mode)
        self.device = device

    def set_mode(self, mode):
        if mode != "sequence" and mode != "variant":
            print(f"Mode options are: \'sequence\' or \'variant\'. Keeping current mode as {mode}")
        else:
            if self.projection:
                self.proj.set_mode(mode)
            self.mode = mode
    def forward(self, x):
        """
        Forward pass: computes output for both original and reversed input
        and averages the results. This is fed into the projector.
        """
        if self.projection:
            if self.proj.mode == "variant":
                x_r, x_a = x
                for_x_r = self.trunk(x_r)
                for_x_r = self.head(for_x_r)

                rev_x_r = torch.flip(x_r, dims=[1, 2])
                rev_x_r = self.trunk(rev_x_r)
                rev_x_r = self.head(rev_x_r)

                out_r = (for_x_r + rev_x_r) / 2


                for_x_a = self.trunk(x_a)
                for_x_a = self.head(for_x_a)

                rev_x_a = torch.flip(x_a, dims=[1, 2])
                rev_x_a = self.trunk(rev_x_a)
                rev_x_a = self.head(rev_x_a)

                out_a = (for_x_a + rev_x_a) / 2

                out = (out_r, out_a)
                out = self.proj(out)
            else: ## default to sequence
                for_x = self.trunk(x)
                for_x = self.head(for_x)

                rev_x = torch.flip(x, dims=[1, 2])
                rev_x = self.trunk(rev_x)
                rev_x = self.head(rev_x)

                out = (for_x + rev_x) / 2
                out = self.proj(out)
        else:
            if self.mode == "variant":
                x_r, x_a = x
                for_x_r = self.trunk(x_r)
                for_x_r = self.head(for_x_r)

                rev_x_r = torch.flip(x_r, dims=[1, 2])
                rev_x_r = self.trunk(rev_x_r)
                rev_x_r = self.head(rev_x_r)

                out_r = (for_x_r + rev_x_r) / 2


                for_x_a = self.trunk(x_a)
                for_x_a = self.head(for_x_a)

                rev_x_a = torch.flip(x_a, dims=[1, 2])
                rev_x_a = self.trunk(rev_x_a)
                rev_x_a = self.head(rev_x_a)

                out_a = (for_x_a + rev_x_a) / 2

                out = (out_r, out_a)
            else:
                for_x = self.trunk(x)
                for_x = self.head(for_x)

                rev_x = torch.flip(x, dims=[1, 2])
                rev_x = self.trunk(rev_x)
                rev_x = self.head(rev_x)

                out = (for_x + rev_x) / 2

        return out


def initialize_models(rank: int, trained_version: str, quant: bool, full = False):
    if quant == True:
        dev = "cpu"
    else:
        dev = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    if dev == "cpu":
        q = "CPU"
    else:
        q = None
    if not full:
        cp_model_seq = sl.Sei_LLRA(k=rank, projection = False, mode = "sequence")
        cp_model_var = sl.Sei_LLRA(k=rank, projection = False, mode = "variant")
        sc_model_seq = sl.Sei_LLRA(k=rank, projection = True, mode = "sequence")
        sc_model_var = sl.Sei_LLRA(k=rank, projection = True, mode = "variant")

        if quant != True:
            cp_model_seq.trunk.load_weights()
            cp_model_var.trunk.load_weights()
            sc_model_seq.trunk.load_weights()
            sc_model_var.trunk.load_weights()
    else:
        cp_model_seq = SeiWrapper(k=rank, ft = trained_version, projection = False, mode = "sequence", device = dev)
        cp_model_var = SeiWrapper(k=rank, ft = trained_version, projection = False, mode = "variant", device = dev)
        sc_model_seq = SeiWrapper(k=rank, ft = trained_version, projection = True, mode = "sequence", device = dev)
        sc_model_var = SeiWrapper(k=rank, ft = trained_version, projection = True, mode = "variant", device = dev)


    return cp_model_seq, cp_model_var, sc_model_seq, sc_model_var


def get_variants_streaming(model, vcf, out_path, benchmark_name="", debug=False):
    dataset = VariantDataset(file_path=vcf)
    dataloader = VariantDataLoader(dataset=dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=False)
    device = model.device
    model = model.to(device)
    model.eval()

    first_write = True

    for i, batch in enumerate(tqdm(dataloader, desc=f"Running {benchmark_name} benchmark")):
        if debug and i>=12: # Debug - first two batches only
            break
        ref, alt, vcf_batch = batch
        ref, alt = ref.to(device), alt.to(device)

        with torch.no_grad():
            out_ref, out_alt = model((ref, alt))
            out_ref = out_ref.detach().cpu()
            out_alt = out_alt.detach().cpu()
            diff = (out_alt - out_ref).numpy()

        df_pred = pd.DataFrame(vcf_batch, columns=["CHROM", "POS", "NAME", "REF", "ALT"])
        df_pred["POS"] = df_pred["POS"].astype(int)
        df_pred["GM12878_DNase_cp_mean"] = get_celltype_asssy_specific(diff, celltypes=["GM12878_B_Lymphocyte_Blood"], assays=["ATAC-seq", "DNase"], strict=True)
        df_pred["Cardiomyocyte_DNase_cp_mean"] = get_celltype_asssy_specific(diff, celltypes=["Cardiomyocyte"], assays=["ATAC-seq", "DNase"], strict=True)

        df_pred.to_csv(
            out_path,
            sep="\t",
            index=False,
            mode="w" if first_write else "a",
            header=first_write
        )
        first_write = False

        del ref, alt, out_ref, out_alt, diff, df_pred
        if device != "cpu":
            torch.cuda.empty_cache()

    model = model.to("cpu")
    torch.cuda.empty_cache()


def get_gtex_eqtls_promoter(model, rank, trained_version = ""):
    benchmark_name = "gtex_eqtls_near_promoter"
    vcf_name ="../data/tableS1D_gtex_eqtls.vcf"
    sc_ref, sc_alt, vcf = get_variants(model, vcf_name, rank = rank, benchmark_name=benchmark_name, trained_version = trained_version, sc = True)
    df = pd.read_csv("../data/tableS1D_gtex_eqtls.tsv", header = 0, sep = "\t")
   
    outs = get_over_under_null(sc_ref, sc_alt, vcf, df)
    return outs 

def get_mpra_eqtls_promoter(model, rank, trained_version = ""):
    benchmark_name = "mpra_eqtls_near_promoter"
    vcf_name ="../data/tableS1E_mpra_eqtls.vcf"
    sc_ref, sc_alt, vcf = get_variants(model, vcf_name, rank = rank, benchmark_name=benchmark_name, trained_version = trained_version, sc = True)
    df = pd.read_csv("../data/tableS1E_mpra_eqtls.tsv", header = 0, sep = "\t")
   
    outs = get_over_under_null(sc_ref, sc_alt, vcf, df)
    return outs 

def get_gtex_outliers_promoter(model, rank, trained_version = ""):
    benchmark_name = "gtex_outliers_near_promoter"
    vcf_name ="../data/tableS1A_gtex_outliers.vcf"
    sc_ref, sc_alt, vcf = get_variants(model, vcf_name, rank = rank, benchmark_name=benchmark_name, trained_version = trained_version, sc = True)
    df = pd.read_csv("../data/tableS1A_gtex_outliers.tsv", header = 0, sep = "\t")
   
    outs = get_over_under_null(sc_ref, sc_alt, vcf, df)
    return outs 

def get_cagi5_sat_promoter(model, rank, trained_version = ""):
    benchmark_name = "cagi5_sat_near_promoter"
    vcf_name ="../data/tableS1B_cagi5_saturation.vcf"
    sc_ref, sc_alt, vcf = get_variants(model, vcf_name, rank = rank, benchmark_name=benchmark_name, trained_version = trained_version, sc = True)
    df = pd.read_csv("../data/tableS1B_cagi5_saturation.tsv", header = 0, sep = "\t")
   
    outs = get_over_under_null(sc_ref, sc_alt, vcf, df)
    return outs 

def get_mpra_sat_promoter(model, rank, trained_version = ""):
    benchmark_name = "mpra_sat_near_promoter"
    vcf_name ="../data/tableS1C_mpra_saturation.vcf"
    sc_ref, sc_alt, vcf = get_variants(model, vcf_name, rank = rank, benchmark_name=benchmark_name, trained_version = trained_version, sc = True)
    df = pd.read_csv("../data/tableS1C_mpra_saturation.tsv", header = 0, sep = "\t")
   
    outs = get_over_under_null(sc_ref, sc_alt, vcf, df)
    return outs 

def get_ukbb_proteome_promoter(model, rank, trained_version = ""):
    benchmark_name = "ukbb_proteome_near_promoter"
    vcf_name ="../data/tableS1F_ukbb_proteome.vcf"
    sc_ref, sc_alt, vcf = get_variants(model, vcf_name, rank = rank, benchmark_name=benchmark_name, trained_version = trained_version, sc = True)
    df = pd.read_csv("../data/tableS1F_ukbb_proteome.tsv", header = 0, sep = "\t")
   
    outs = get_over_under_null(sc_ref, sc_alt, vcf, df)
    return outs 

def get_gel_rna_promoter(model, rank, trained_version = ""):
    benchmark_name = "gel_rna_near_promoter"
    vcf_name ="../data/tableS1G_gel_rna.vcf"
    sc_ref, sc_alt, vcf = get_variants(model, vcf_name, rank = rank, benchmark_name=benchmark_name, trained_version = trained_version, sc = True)
    df = pd.read_csv("../data/tableS1G_gel_rna.tsv", header = 0, sep = "\t")
   
    outs = get_over_under_null(sc_ref, sc_alt, vcf, df)
    return outs 

def get_over_under_null(sc_ref, sc_alt, vcf, df):
    sc_diff = sc_alt - sc_ref
    df_pred = pd.DataFrame(vcf, columns=["CHROM", "POS", "NAME", "REF", "ALT"])
    df_pred["POS"] = df_pred["POS"].astype(int)
    df_sc = get_sequence_class_scores_and_max(sc_diff)
    df_pred =  pd.concat([df_pred, df_sc], axis=1)

    df_ou = df[df['consequence'].isin(['over', 'under'])].copy()
    df_combine_ou = df_ou.merge(df_pred, left_on = ["chrom", "pos", "ref", "alt"], right_on=["CHROM", "POS", "REF", "ALT"], how = "inner")
    df_combine_ou = df_combine_ou.drop_duplicates()
    binary_labels_ou = (df_combine_ou['consequence'] == 'over')
    roc_promoter_ou = roc_auc_score(binary_labels_ou, df_combine_ou["Promoter"])

    df_un = df[df['consequence'].isin(['under', 'none'])].copy()
    df_combine_un = df_un.merge(df_pred, left_on = ["chrom", "pos", "ref", "alt"], right_on=["CHROM", "POS", "REF", "ALT"], how = "inner")
    df_combine_un = df_combine_un.drop_duplicates()
    binary_labels_un = (df_combine_un['consequence'] == 'under')
    roc_promoter_un = roc_auc_score(binary_labels_un, -df_combine_un["Promoter"])

    df_on = df[df['consequence'].isin(['over', 'none'])].copy()
    df_combine_on = df_on.merge(df_pred, left_on = ["chrom", "pos", "ref", "alt"], right_on=["CHROM", "POS", "REF", "ALT"], how = "inner")
    df_combine_on = df_combine_on.drop_duplicates()
    binary_labels_on = (df_combine_on['consequence'] == 'over')
    roc_promoter_on = roc_auc_score(binary_labels_on, df_combine_on["Promoter"])


    return roc_promoter_ou, roc_promoter_un, roc_promoter_on 


def get_eu_lcl_caqtls(model, rank, trained_version = "", debug=False):
    benchmark_name = "caqtls_eu_GM12878"
    vcf_name = "../data/caqtls.eu.lcls.benchmarking.all.vcf"
    temp_out = f"scores/caqtls_eu_lcl_seilora_rank{rank}_stream.tsv"
    final_out = f"scores/caqtls_eu_lcl_seilora_rank{rank}_quant.tsv.gz"

    # Step 1: stream inference, write scores to disk batch-by-batch
    get_variants_streaming(model, vcf_name, temp_out, benchmark_name=benchmark_name, debug=debug)

    # Step 2: load predictions into a lookup dict
    pred_dict = {}
    for chunk in pd.read_csv(temp_out, sep="\t", chunksize=100_000):
        for row in chunk.itertuples(index=False):
            key = (row.CHROM, int(row.POS))
            pred_dict[key] = (row.GM12878_DNase_cp_mean, row.Cardiomyocyte_DNase_cp_mean)

    # Step 3: stream ground truth, compute obs.label, merge with predictions
    df = pd.read_csv("../data/caqtls.eu.lcls.benchmarking.all.tsv", header=0, sep="\t")
    if debug: # Debug - only first twelve batches
        df = df.head(96).reset_index(drop=True)
    # df = df[df["var.isused"]]
    # print("var.isused details")
    # print(df["var.isused"].dtype, df["var.isused"].unique())
    # print(df.index)
    df = df[df["var.isused"]].copy().reset_index(drop=True)
    df["log10p"] = np.log10(df["obs.pval"]) * -1

    dataf1 = df[df["log10p"] > 6].copy().reset_index(drop=True)
    dataf2 = df[df["log10p"] < 3].copy().reset_index(drop=True)
    # dataf1.loc[:, "obs.label"] = 1
    # dataf2.loc[:, "obs.label"] = 0
    dataf1["obs.label"] = 1
    dataf2["obs.label"] = 0

    # In debug mode, filtering might lead to one of the dataframes being empty
    if dataf1.empty or dataf2.empty:
        print("Warning: one label group is empty in debug mode, skipping metrics.")
        return None, None

    df = pd.concat([dataf1, dataf2])

    keys = list(zip(df["var.chr"], df["var.pos_hg38"].astype(int)))
    df["GM12878_DNase_cp_mean"] = [pred_dict.get(k, (np.nan, np.nan))[0] for k in keys]
    df["Cardiomyocyte_DNase_cp_mean"] = [pred_dict.get(k, (np.nan, np.nan))[1] for k in keys]
    df = df.dropna(subset=["GM12878_DNase_cp_mean"])

    df.to_csv(final_out, sep="\t", index=False, compression="gzip")
    print("EU LCL CAQTLs: Saved scores")

    ap_unsigned = average_precision_score(df["obs.label"], abs(df["GM12878_DNase_cp_mean"]))

    df_sig = df[df["log10p"] > 6]
    pearson_signed = scipy.stats.pearsonr(df_sig["GM12878_DNase_cp_mean"], df_sig["obs.beta"])

    return pearson_signed, ap_unsigned


def get_variants(model, vcf, rank, benchmark_name="", trained_version = "", sc = False):
    dataset = VariantDataset(file_path=vcf)
    # dataloader = VariantDataLoader(dataset=dataset, batch_size=32, shuffle=False, num_workers=15)
    dataloader = VariantDataLoader(dataset=dataset, batch_size=8, shuffle=False, num_workers=8) # Reduce batch size due to CUDA OOM
    device = model.device
    model = model.to(device)
    model.eval()

    all_cp_ref = []
    all_cp_alt = []
    all_vcf = []

    progress_bar = tqdm(dataloader, desc=f"Running {rank} {trained_version} {benchmark_name} benchmark")

    for batch in progress_bar:
        ref, alt, vcf = batch
        ref, alt = ref.to(device), alt.to(device)

        cp_outputs = model((ref, alt))  # both are tuples: (refproj, altproj)

    
        all_cp_ref.append(cp_outputs[0].detach().cpu())
        all_cp_alt.append(cp_outputs[1].detach().cpu())
        all_vcf.append(vcf)

        # Accumulate by appending to list
    
    all_cp_ref = torch.cat([t.detach().cpu() for t in all_cp_ref], dim=0).numpy()
    all_cp_alt = torch.cat([t.detach().cpu() for t in all_cp_alt], dim=0).numpy()

    all_vcf = np.concatenate(all_vcf, axis=0)
    model = model.to("cpu")
    torch.cuda.empty_cache()

    return all_cp_ref, all_cp_alt, all_vcf

def get_scores(model, bed, rank, benchmark_name="", trained_version = "", scores = None, sc = False):
    dataset = SeqDataset(file_path=bed, scores_path = scores, fasta_path="../../sei-framework-main/resources/hg38_UCSC.fa",
            mode = "test", val_chrom = "chr10", test_chrom = ["chr8", "chr9"])
    dataloader = SeqDataLoader(dataset=dataset, batch_size=32, shuffle=False, num_workers=8, n_samples=10_000)
    device = model.device

    model = model.to(device)
    model.eval()

    all_cp = []
    all_scores = []

    progress_bar = tqdm(dataloader, desc=f"Running {rank} {trained_version} {benchmark_name} benchmark")

    for batch in progress_bar:
        data, score = batch
        data = data.to(device)

        out = model(data)  # both are tuples: (refproj, altproj)

    
        all_cp.append(out.detach().cpu())
        all_scores.append(score)

        # Accumulate by appending to list
    model = model.to("cpu")
    
    all_cp = torch.cat([t.detach().cpu() for t in all_cp], dim=0).numpy()
    all_scores = np.concatenate(
        [t.detach().cpu().numpy() if torch.is_tensor(t) else t for t in all_scores],
        axis=0
    )
    torch.cuda.empty_cache()
    return all_cp, all_scores

def save_output(rank = 256, trained_version = None, quant = False, full = False, debug=False):
    if quant:
        q = "quant"
    else:
        q = "no_quant"

    model_name = f"seilora_{rank}_{trained_version}_{q}"
    print(f"Model name: {model_name}")

    cp_seq_mod, cp_var_mod, sc_seq_mod, sc_var_mod = initialize_models(rank = rank, trained_version = trained_version, quant = quant, full = full)

    try:
        eu_pearson, eu_ap = get_eu_lcl_caqtls(model = cp_var_mod, rank = rank, trained_version = trained_version, debug=debug)
    except Exception as e:
        print(f"get_eu_lcl_caqtls failed: {e}")

    bpn_path = "benchmark_chrombpnet_all_quant.tsv"
    bpn_row_dict = {
        "model": model_name,
        "EU_LCL_pearson_signed": round(eu_pearson.statistic, 4),
        "EU_LCL_AP_unsigned": round(eu_ap, 4),  
    }

    os.makedirs("scores", exist_ok=True) # Check output directory exists
    file_exists = os.path.isfile(bpn_path)
    with open(bpn_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=bpn_row_dict.keys(), delimiter="\t")
            if not file_exists:
                writer.writeheader()
            writer.writerow(bpn_row_dict)

            print(f"Results saved to {bpn_path}")
    

    torch.cuda.empty_cache()

def main():
    os.makedirs("scores", exist_ok=True)
    save_output(rank=1, quant=True, debug=True)
    # save_output(rank=1, quant=True)

if __name__ == '__main__':
    main()