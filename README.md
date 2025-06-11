# Unveiling Latent Structural Collapse: A Discrete Morse Perspective on SAE Representations for Gene Expression

## Overview
<pre>
  <code>
    📁 Project Root
      ├── README.md                
      ├── SAE.py                  # Sparse Autoencoder model 
      ├── cellbert_runner.py      # BERT-based classifier runner
      ├── data_lookup.ipynb       # Data lookup and inspection notebook (Could neglect)
      ├── dm_full.py              # Discrete Morse skeleton code
      ├── pipeline.ipynb          # Full pipeline demo notebook 
      ├── presentation_url.txt    # Link to presentation video (*)
      ├── requirements.txt        # Python dependencies
      └── test.ipynb              # SAE / visualization checks
  </code>
</pre>

## Dataset
Before running this project, we should download the data from [Single Cell 3' Gene Expression](https://www.10xgenomics.com/datasets/pbm-cs-from-a-healthy-donor-whole-transcriptome-analysis-3-1-standard-4-0-0).
At "Output and supplemental files", I suggest to download "Feature / cell matrix HDF5 (filtered)" and "Clustering analysis". Put them into a folder called "data" in the root (if not exist, plz create one).
<p align="center">
  <table>
    <tr>
      <td align="center"><img src="images/newplot.png" alt="t-SNE Projection of Cells Colored by Clustering" width="500"/></td>
      <td align="center"><img src="images/newplotden.png" alt="t-SNE Projection of Cells Colored by UMI Counts" width="500"/></td>
    </tr>
  </table>
</p>

## Libraries
Plz refer to the requirements.txt. That would provide necessary depedencies (but might not satisfy all, as I might add more extensions to the project). If that happens, plz import needed libs by yourself.

## Latent Space by SAE and BERT
<p align="center">
  <img src="images/struct.png" alt="Pipeline Overview: SAE + BERT" width="800"/>
</p>
According to the pipeline, we will run a SAE multiple times for different latent space and apply them to BERT models for classification tasks. 
Use test.ipynb to train SAE for latent spaces. Refer to the code (take dim=128 as an example):
<pre>
  <code>
    X_dense = adata.X.toarray()
    X_scaled = StandardScaler().fit_transform(X_dense)
    data_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    model = SparseAutoencoder(input_dim=X_scaled.shape[1], latent_dim=128)
    trained_model = train_sae(model, data_tensor, epochs=50, l1_weight=1e-4)
  </code>
</pre>

Use pipeline.ipynb to train BERT model for classfication tasks (take dim=8 as an example).
<pre>
  <code>
    cfg = {
        "adata_h5": "data/Parent_NGSC3_DI_PBMC_filtered_feature_bc_matrix.h5",
        "cluster_csv": "data/Parent_NGSC3_DI_PBMC_analysis/analysis/clustering/graphclust/clusters.csv",
        "use_latent": True,
        "latent_path": "latent_db/latent_8.npy",
        "bins": 100,
        "embed_dim": 128,
        "layers": 4,
        "heads": 8,
        "dropout": 0.1,
        "epochs": 15,
        "batch_size": 64,
        "lr": 1e-4,
    }
    model, test_acc = cr.run_experiment(cfg)
    print("Final test acc:", test_acc)
  </code>
</pre>

## Discrete Morse Skeleton
Use test.ipynb to create DMS for latent spaces. Refer to the code for step 1:
<pre>
  <code>
    X_lat = np.load("latent_db/latent_8.npy") 
    pca = PCA(n_components=2, whiten=False, random_state=0)
    pts2d = pca.fit_transform(X_lat)             
    pts_min = pts2d.min(0)
    pts_span = pts2d.max(0) - pts_min
    pts_norm = (pts2d - pts_min) / pts_span   
  </code>
</pre>
Refer to the code for step 2:
<pre>
  <code>
    G = discrete_morse_graph(
            pts_norm,
            grid_res = 512,
            sigma    = 1,
            pers_len = 0.001, 
            pers_rho = 0.0001,
            visualize=False)
    print(f"Skeleton: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    visualize_graph2d(G)   
  </code>
</pre>
## Other Operations
About other operations such as persistent homology analysis. Plz check the specific file (e.g. test.ipynb) for detailed deployment.

Thank you for viewing this project.
