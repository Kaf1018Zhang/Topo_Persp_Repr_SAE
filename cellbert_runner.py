import os, numpy as np, pandas as pd, scanpy as sc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm

class ExpressionTokenizer:
    def __init__(self, bins: int = 7):
        self.bins = bins
        self.pad_id, self.cls_id = 0, bins + 1            # 0:[PAD]  bins+1:[CLS]

    def encode(self, vec: np.ndarray) -> np.ndarray:
        """vector -> token sequence (len = genes+1); first token is [CLS]"""
        thresholds = np.percentile(vec, np.linspace(0, 100, self.bins + 1)[1:-1])
        tokens = (np.digitize(vec, thresholds) + 1).astype(np.int64)  # 1..bins
        return np.concatenate([[self.cls_id], tokens])

class CellBERT(nn.Module):
    def __init__(self, num_genes, num_bins, embed_dim, layers, heads,
                 num_classes, dropout):
        super().__init__()
        vocab = num_bins + 2                             # bins + [PAD]+[CLS]
        self.token_emb = nn.Embedding(vocab, embed_dim)
        self.pos_emb   = nn.Embedding(num_genes + 1, embed_dim)  #+1 for CLS
        enc_layer = nn.TransformerEncoderLayer(embed_dim, heads,
                                               dim_feedforward=embed_dim*4,
                                               dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.cls_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):                                # x:(B,T)
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        h = self.token_emb(x) + self.pos_emb(pos)
        h = self.encoder(h)
        cls = self.norm(h[:, 0, :])                      # CLS token
        return self.cls_head(cls)

def _normalize_log1p(mat: np.ndarray):
    tot = mat.sum(1, keepdims=True)
    mat = mat / (tot + 1e-9) * 1e4
    return np.log1p(mat)

def _make_loader(x, y, bs, shuffle):
    ds = TensorDataset(torch.tensor(x, dtype=torch.long),
                       torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)

def run_experiment(cfg: dict):
    adata_h5     = cfg["adata_h5"]
    cluster_csv  = cfg["cluster_csv"]
    use_latent   = cfg.get("use_latent", False)
    latent_path  = cfg.get("latent_path", "")

    bins       = cfg.get("bins", 7)
    embed_dim  = cfg.get("embed_dim", 128)
    layers     = cfg.get("layers", 4)
    heads      = cfg.get("heads", 8)
    dropout    = cfg.get("dropout", 0.1)
    epochs     = cfg.get("epochs", 15)
    batch_size = cfg.get("batch_size", 64)
    lr         = cfg.get("lr", 1e-4)

    test_size    = cfg.get("test_size", 0.30)
    random_state = cfg.get("random_state", 42)
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adata = sc.read_10x_h5(adata_h5)
    barcodes = adata.obs_names.tolist()

    if use_latent:
        X = np.load(latent_path).astype(np.float32)
    else:
        sc.pp.normalize_total(adata); sc.pp.log1p(adata)
        X = adata.X.toarray().astype(np.float32)
    num_genes = X.shape[1]

    df = pd.read_csv(cluster_csv)
    y_map = dict(zip(df.Barcode, df.Cluster))
    y = np.array([y_map.get(bc, -1) for bc in barcodes])
    keep = y != -1
    X, y = X[keep], y[keep]

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=test_size,
                                                stratify=y, random_state=random_state)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5,
                                                stratify=y_tmp, random_state=random_state)

    tok = ExpressionTokenizer(bins)
    def v2seq(mat):
        if not use_latent:
            mat = _normalize_log1p(mat)
        return np.stack([tok.encode(v) for v in mat])
    seq_tr, seq_val, seq_te = map(v2seq, (X_tr, X_val, X_te))

    classes, y_tr_enc = np.unique(y_tr, return_inverse=True)
    id_map = {c:i for i,c in enumerate(classes)}
    y_val_enc = np.array([id_map[c] for c in y_val])
    y_te_enc  = np.array([id_map[c] for c in y_te])

    train_loader = _make_loader(seq_tr, y_tr_enc, batch_size, True)
    val_loader   = _make_loader(seq_val, y_val_enc, batch_size, False)
    test_loader  = _make_loader(seq_te, y_te_enc, batch_size, False)

    model = CellBERT(num_genes, bins, embed_dim, layers, heads,
                     len(classes), dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs}", leave=False)
        run_loss = 0.0
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward(); optimizer.step()
            run_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss = run_loss / len(train_loader)

        model.eval(); preds=[]
        with torch.no_grad():
            for xb,_ in val_loader:
                preds += torch.argmax(model(xb.to(device)),1).cpu().tolist()
        val_acc = accuracy_score(y_val_enc, preds)

        tqdm.write(f"[Epoch {ep:02d}] train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")

    model.eval(); preds=[]
    with torch.no_grad():
        for xb,_ in test_loader:
            preds += torch.argmax(model(xb.to(device)),1).cpu().tolist()
    test_acc = accuracy_score(y_te_enc, preds)
    tqdm.write(f"\n>>> Test Accuracy: {test_acc:.4f}\n")

    return model, test_acc
