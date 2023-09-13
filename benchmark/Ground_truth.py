import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, \
                            homogeneity_completeness_v_measure
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
from pathlib import Path
import stlearn as st
import matplotlib.pyplot as plt
import scanpy
import os
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    cm = contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm)
# specify PATH to data
BASE_PATH = Path("/home/zzq/zzq/DeepST-main/data/data/DLPFC")
OUTPUT_PATH= Path('/home/zzq/zzq/DeepST-main/Benchmark/groundTruth/coloreNo')
# here we include all 12 samples
sample_list = ["151507", "151508", "151509",
               "151510", "151669", "151670",
               "151671", "151672", "151673",
               "151674", "151675", "151676"]
for i in range(len(sample_list)):
    sample = sample_list[i]
    data = st.Read10X(os.path.join(BASE_PATH, sample))
    ground_truth_df = pd.read_csv(BASE_PATH / sample / 'metadata.tsv', sep='\t')
    ground_truth_df['ground_truth'] = ground_truth_df['layer_guess']
    le = LabelEncoder()
    ground_truth_le = le.fit_transform(list(ground_truth_df["ground_truth"].values))
    n_cluster = len((set(ground_truth_df["ground_truth"]))) - 1
    data.obs['ground_truth'] = ground_truth_df["ground_truth"]
    ground_truth_df["ground_truth_le"] = ground_truth_le
    #plot_color=["#F56867","#FEB915","#C798EE","#59BE86","#7495D3","#D1D1D1","#6D1A9C","#15821E","#3A84E6","#997273","#787878","#DB4C6C","#9E7A7A","#554236","#AF5F3C","#93796C","#F9BD3F","#DAB370","#877F6C","#268785"]
    #plot_color = ["#a68069","#91b758","#5891b7","#7e58b7","#c688a3","#86c89f","#3cc6cb"]
    plot_color = ["#F56867","#FEB915","#C798EE","#59BE86","#7495D3","#D1D1D1","#6D1A9C"]
    #a68069,#91b758,#5891b7,#7e58b7,#c688a3,#86c89f,#3cc6cb
    #load data
    data = st.Read10X(BASE_PATH / sample)
    ground_truth_df = ground_truth_df.reindex(data.obs_names)
    data.obs["ground_truth"] = pd.Categorical(ground_truth_df["ground_truth"])
    #st.pl.cluster_plot(data, use_label="ground_truth",dpi=300)

    st.pl.cluster_plot(data,use_label="ground_truth",show_image=None,dpi=300,crop=True,cmap=plot_color)

    plt.savefig(OUTPUT_PATH / f'{sample}.png')

