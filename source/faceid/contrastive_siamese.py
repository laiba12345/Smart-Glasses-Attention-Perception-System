# contrastive_siamese.py
# Simple Siamese contrastive learner on top of FaceNet embeddings (CSV: folder_name, file_name, e0..e511)

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# -------------------------
# small helpers
# -------------------------

def l2(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps)

def embedding_cols(df: pd.DataFrame, prefix: str = "e") -> list[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"No embedding columns found with prefix '{prefix}'.")
    return sorted(cols, key=lambda c: int(c[len(prefix):]))


# -------------------------
# model + loss
# -------------------------

class Projector(nn.Module):
    """MLP projector with L2-normalized output."""
    def __init__(self, sizes: Sequence[int] = (512, 256, 128), dropout: float = 0.1):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers += [nn.ReLU(inplace=True)]
                if dropout > 0:
                    layers += [nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return l2(self.net(x), dim=1)


class ContrastiveLoss(nn.Module):
    """
    Classic Siamese contrastive loss (pairs):
      y=0 => same person (positive)
      y=1 => different people (negative)
      L = (1-y)*d^2 + y*max(0, margin-d)^2
    """
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = float(margin)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        d = (z1 - z2).norm(p=2, dim=1)  # with L2-normalized vectors, d in ~[0,2]
        y = y.float()
        pos = (1.0 - y) * (d ** 2)
        neg = y * (torch.clamp(self.margin - d, min=0.0) ** 2)
        return (pos + neg).mean()


# -------------------------
# config
# -------------------------

@dataclass
class Cfg:
    sizes: Tuple[int, ...] = (512, 256, 128)
    dropout: float = 0.1
    margin: float = 1.0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    batch_size: int = 128
    num_pairs_per_epoch: int = 15000     # you called it "num_triplets" earlier, but siamese uses pairs
    pos_fraction: float = 1 / 3          # ~1/3 positives, ~2/3 negatives
    prefix: str = "e"
    folder_col: str = "folder_name"
    file_col: str = "file_name"
    seed: int = 42
    sim_threshold: float = 70.0        # similarity% gate for "known"



# -------------------------
# main trainer
# -------------------------

class Siamese:
    def __init__(self, cfg: Cfg = Cfg(), device: Optional[str] = None):
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.rng = np.random.default_rng(cfg.seed)

        self.model = Projector(cfg.sizes, cfg.dropout).to(self.device)
        self.loss_fn = ContrastiveLoss(cfg.margin).to(self.device)
        self.prototypes: Optional[Dict[str, torch.Tensor]] = None  # name -> (D,)

    # ---- data loading ----

    def _load_csv(self, path: Union[str, Path]):
        df = pd.read_csv(path)
        cols = embedding_cols(df, self.cfg.prefix)

        X = torch.tensor(df[cols].values, dtype=torch.float32)  # (N,512)
        X = l2(X, dim=1)                                        # safe normalization

        y = df[self.cfg.folder_col].astype(str).to_numpy()
        return df, X, y

    # ---- sampling pairs (balanced anchors) ----

    def _sample_pairs(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        people = sorted(set(y.tolist()))
        idx = {p: np.where(y == p)[0] for p in people}

        per_person = max(1, self.cfg.num_pairs_per_epoch // len(people))
        n_pos = int(round(per_person * self.cfg.pos_fraction))
        n_neg = per_person - n_pos

        I, J, Y = [], [], []
        for p in people:
            ids = idx[p]
            if len(ids) == 0:
                continue

            # positives (need >=2)
            if len(ids) >= 2:
                for _ in range(n_pos):
                    i = int(self.rng.choice(ids))
                    j = int(self.rng.choice(ids))
                    while j == i:
                        j = int(self.rng.choice(ids))
                    I.append(i); J.append(j); Y.append(0)

            # negatives
            others = [q for q in people if q != p]
            for _ in range(n_neg):
                i = int(self.rng.choice(ids))
                q = str(self.rng.choice(others))
                j = int(self.rng.choice(idx[q]))
                I.append(i); J.append(j); Y.append(1)

        # trim/pad to exact num_pairs_per_epoch
        n = self.cfg.num_pairs_per_epoch
        if len(I) >= n:
            sel = self.rng.choice(len(I), size=n, replace=False)
        else:
            sel = self.rng.choice(len(I), size=n, replace=True)

        I = np.asarray(I)[sel]; J = np.asarray(J)[sel]; Y = np.asarray(Y)[sel]
        perm = self.rng.permutation(len(I))
        return I[perm], J[perm], Y[perm]

    # ---- prototypes + val loss ----

    @torch.no_grad()
    def _compute_prototypes(self, X: torch.Tensor, y: np.ndarray) -> Dict[str, torch.Tensor]:
        self.model.eval()
        Z = self.model(X.to(self.device)).cpu()  # (N,D)

        prot = {}
        for p in sorted(set(y.tolist())):
            rows = np.where(y == p)[0]
            mu = Z[rows].mean(dim=0)
            prot[p] = l2(mu.unsqueeze(0), dim=1).squeeze(0)  # (D,)
        return prot

    @torch.no_grad()
    def _proto_mse_loss(self, X: torch.Tensor, y: np.ndarray, protos: Dict[str, torch.Tensor]) -> float:
        """Mean squared distance to the true class prototype."""
        self.model.eval()
        Z = self.model(X.to(self.device)).cpu()
        d2 = []
        for i, label in enumerate(y):
            if label in protos:
                d = (Z[i] - protos[label]).norm(2).item()
                d2.append(d * d)
        return float(np.mean(d2)) if d2 else float("nan")

    @torch.no_grad()
    def _proto_accuracy(self, X: torch.Tensor, y: np.ndarray, protos: Dict[str, torch.Tensor]) -> float:
        """Nearest-prototype classification accuracy."""
        self.model.eval()
        Z = self.model(X.to(self.device)).cpu()  # (N,D)
        names = list(protos.keys())
        P = torch.stack([protos[n] for n in names], dim=0)  # (M,D)
        D = torch.cdist(Z, P)                               # (N,M)
        pred_idx = torch.argmin(D, dim=1).numpy()
        pred = np.array(names)[pred_idx]
        return float(np.mean(pred == y))


    @torch.no_grad()
    def _val_loss(self, X_train: torch.Tensor, y_train: np.ndarray, X_val: torch.Tensor, y_val: np.ndarray) -> float:
        """
        Proto/val loss (simple + stable):
          - prototypes from TRAIN using current model
          - val loss = mean squared distance between each val embedding and its class prototype
        """
        prot = self._compute_prototypes(X_train, y_train)
        self.prototypes = prot

        self.model.eval()
        Z = self.model(X_val.to(self.device)).cpu()  # (Nv,D)

        d2 = []
        for i, label in enumerate(y_val):
            if label not in prot:
                continue
            d = (Z[i] - prot[label]).norm(p=2).item()
            d2.append(d * d)
        return float(np.mean(d2)) if d2 else float("nan")

    # ---- training ----

    def fit(
        self,
        train_csv: Union[str, Path],
        val_csv: Optional[Union[str, Path]] = None,
        out_dir: Union[str, Path] = "contrastive_out",
        # optional overrides
        num_triplets: Optional[int] = None,   # alias: treated as num_pairs_per_epoch
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
        weight_decay: Optional[float] = None,
        margin: Optional[float] = None,
        batch_size: Optional[int] = None,
        sim_threshold: Optional[float] = None,
    ):
        if num_triplets is not None:
            self.cfg.num_pairs_per_epoch = int(num_triplets)
        if epochs is not None:
            self.cfg.epochs = int(epochs)
        if lr is not None:
            self.cfg.lr = float(lr)
        if weight_decay is not None:
            self.cfg.weight_decay = float(weight_decay)
        if batch_size is not None:
            self.cfg.batch_size = int(batch_size)
        if margin is not None:
            self.cfg.margin = float(margin)
            self.loss_fn.margin = float(margin)
        if sim_threshold is not None:
            self.cfg.sim_threshold = float(sim_threshold)

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        _, Xtr, ytr = self._load_csv(train_csv)
        val_pack = self._load_csv(val_csv) if val_csv else None

        opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        for ep in range(1, self.cfg.epochs + 1):
            self.model.train()

            I, J, Y = self._sample_pairs(ytr)  # resample each epoch
            # iterate in batches
            total = 0.0
            n_batches = 0
            for k in range(0, len(I), self.cfg.batch_size):
                idx = slice(k, k + self.cfg.batch_size)

                x1 = Xtr[I[idx]].to(self.device)
                x2 = Xtr[J[idx]].to(self.device)
                y  = torch.tensor(Y[idx], dtype=torch.int64, device=self.device)

                z1 = self.model(x1)
                z2 = self.model(x2)
                loss = self.loss_fn(z1, z2, y)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                total += float(loss.item())
                n_batches += 1

            train_loss = total / max(1, n_batches)

            # ---- val metrics (no grad): build train prototypes, then eval train/val ----
            with torch.no_grad():
                prot = self._compute_prototypes(Xtr, ytr)
                self.prototypes = prot

                train_acc = self._proto_accuracy(Xtr, ytr, prot)

                val_loss = float("nan")
                val_acc = float("nan")
                if val_pack:
                    _, Xv, yv = val_pack
                    val_loss = self._proto_mse_loss(Xv, yv, prot)
                    val_acc = self._proto_accuracy(Xv, yv, prot)

            hist["train_loss"].append(train_loss)
            hist["val_loss"].append(val_loss)
            hist["train_acc"].append(train_acc)
            hist["val_acc"].append(val_acc)

            print(f"epoch {ep:03d}/{self.cfg.epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                  f"train_acc={train_acc:.3f} | val_acc={val_acc:.3f}")

        # save model + prototypes
        ckpt = out_dir / "siamese_model.pt"
        torch.save({"state": self.model.state_dict(), "cfg": asdict(self.cfg)}, ckpt)

        # final prototypes from train
        prot = self._compute_prototypes(Xtr, ytr)
        self.prototypes = prot
        proto_csv = out_dir / "prototypes.csv"
        self._save_prototypes_csv(proto_csv)

        # ---- plot learning curves ----
        import matplotlib.pyplot as plt

        epochs = np.arange(1, len(hist["train_loss"]) + 1)
        fig, ax1 = plt.subplots()

        ax1.plot(epochs, hist["train_loss"], label="train contrastive loss")
        ax1.plot(epochs, hist["val_loss"], label="val proto-MSE loss")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss")

        ax2 = ax1.twinx()
        ax2.plot(epochs, hist["train_acc"], linestyle="--", label="train acc")
        ax2.plot(epochs, hist["val_acc"], linestyle="--", label="val acc")
        ax2.set_ylabel("accuracy")

        # single combined legend
        lines = ax1.get_lines() + ax2.get_lines()
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="best")

        fig.suptitle(f"Loss & Accuracy (margin={self.cfg.margin})")
        fig.tight_layout()

        plot_path = Path(out_dir) / f"curves_margin{self.cfg.margin}.png"
        fig.savefig(plot_path, dpi=150)
        plt.show()
        print("saved plot:", plot_path)

        print("saved:", ckpt, "and", proto_csv)
        return ckpt

    # ---- save/load ----

    def load(self, ckpt_path: Union[str, Path]):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.cfg = Cfg(**ckpt["cfg"])
        self.model = Projector(self.cfg.sizes, self.cfg.dropout).to(self.device)
        self.loss_fn = ContrastiveLoss(self.cfg.margin).to(self.device)
        self.model.load_state_dict(ckpt["state"])
        self.model.eval()

    def _save_prototypes_csv(self, path: Union[str, Path]):
        if not self.prototypes:
            raise ValueError("No prototypes computed.")
        rows = []
        for name, v in self.prototypes.items():
            row = {self.cfg.folder_col: name}
            vv = v.cpu().numpy().astype(np.float32)
            for i, val in enumerate(vv):
                row[f"p{i}"] = float(val)
            rows.append(row)
        pd.DataFrame(rows).to_csv(path, index=False)

    def load_prototypes(self, proto_csv: Union[str, Path]):
        df = pd.read_csv(proto_csv)
        pcols = [c for c in df.columns if c.startswith("p")]
        prot = {}
        for _, r in df.iterrows():
            name = str(r[self.cfg.folder_col])
            v = torch.tensor(r[pcols].values.astype(np.float32))
            prot[name] = l2(v.unsqueeze(0), dim=1).squeeze(0)
        self.prototypes = prot
        return prot

    # ---- inference ----

    @torch.no_grad()
    def project(self, facenet_emb: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        x = torch.tensor(facenet_emb, dtype=torch.float32) if isinstance(facenet_emb, np.ndarray) else facenet_emb.float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = l2(x, dim=1)
        self.model.eval()
        return self.model(x.to(self.device)).cpu().squeeze(0)  # (D,)

    @staticmethod
    def similarity_percent(distance: float) -> float:
        # with L2-normalized embeddings, Euclidean distance ~ [0,2]
        return float(100.0 * max(0.0, 1.0 - (distance / 2.0)))

    def _is_stranger(self, best_distance: float) -> bool:
        # Use similarity% gate: too LOW similarity => stranger
        return self.similarity_percent(best_distance) < float(self.cfg.sim_threshold)

    @torch.no_grad()
    def predict(self, facenet_emb: Union[np.ndarray, torch.Tensor]) -> Dict[str, Union[str, float]]:
        if not self.prototypes:
            raise ValueError("Call load_prototypes() or train first.")
        z = self.project(facenet_emb)

        names = list(self.prototypes.keys())
        P = torch.stack([self.prototypes[n] for n in names], dim=0)  # (M,D)
        d = torch.cdist(z.unsqueeze(0), P).squeeze(0)                # (M,)
        k = int(torch.argmin(d).item())
        dist = float(d[k].item())

        pred_name = names[k]
        sim = self.similarity_percent(dist)

        if self._is_stranger(dist):
            pred_name = "Stranger"

        return {
            "pred": pred_name,
            "distance": dist,
            "similarity_percent": sim,
        }


    @torch.no_grad()
    def predict_csv(self, in_csv: Union[str, Path], out_csv: Union[str, Path] = None, batch_size: int = 1024) -> pd.DataFrame:
        if not self.prototypes:
            raise ValueError("Call load_prototypes() or train first.")

        df = pd.read_csv(in_csv)
        cols = embedding_cols(df, self.cfg.prefix)

        X = torch.tensor(df[cols].values, dtype=torch.float32)
        X = l2(X, dim=1)

        names = list(self.prototypes.keys())
        P = torch.stack([self.prototypes[n] for n in names], dim=0)  # (M,D)

        preds, dists, sims = [], [], []
        self.model.eval()

        for k in range(0, len(X), batch_size):
            xb = X[k:k+batch_size].to(self.device)
            zb = self.model(xb).cpu()                 # (B,D)
            D = torch.cdist(zb, P)                    # (B,M)
            idx = torch.argmin(D, dim=1)              # (B,)
            best = D[torch.arange(D.size(0)), idx]    # (B,)

            for i in range(D.size(0)):
                di = float(best[i].item())
                si = self.similarity_percent(di)

                name = names[int(idx[i].item())]
                if self._is_stranger(di):
                    name = "Stranger"

                preds.append(name)
                dists.append(di)
                sims.append(si)


        out = pd.DataFrame({
            self.cfg.folder_col: df[self.cfg.folder_col] if self.cfg.folder_col in df.columns else [""] * len(df),
            self.cfg.file_col: df[self.cfg.file_col] if self.cfg.file_col in df.columns else [""] * len(df),
            "pred_folder_name": preds,
            "distance": dists,
            "similarity_percent": sims,
        })
        if out_csv is not None:
            out_csv = Path(out_csv)
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(out_csv, index=False)
        return out
