import os
import pickle
import uuid
import shutil
import cv2
import numpy as np
from datetime import datetime


class ObjectDatabase:
    """
    Stores detected objects (cropped images + metadata) in three collections:
    - all       → every detection
    - unique    → deduplicated detections
    - clustered → cluster representatives
    """

    def __init__(self):
        from config import Config
        self.all_db_path     = 'all_objects_database.pkl'
        self.unique_db_path  = 'unique_objects_database.pkl'
        self.cluster_db_path = 'cluster_objects_database.pkl'
        self.config          = Config

        self.all_objects:       list = self._load_db(self.all_db_path)
        self.unique_objects:    list = self._load_db(self.unique_db_path)
        self.clustered_objects: list = self._load_db(self.cluster_db_path)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load_db(self, path: str) -> list:
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, list):
                    return data
                print(f'Warning: {path} contained a {type(data).__name__}, not a list. Resetting.')
            except Exception as exc:
                print(f'Warning: could not load {path}: {exc}')
        return []

    def _save_db(self, data: list, path: str) -> None:
        try:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as exc:
            print(f'Error saving database {path}: {exc}')

    # ------------------------------------------------------------------
    # Adding objects
    # ------------------------------------------------------------------

    def add_to_all_objects(self, label: str, image: np.ndarray,
                           confidence: float, frame_info: dict) -> dict:
        """Save a cropped object image to disk and record metadata."""
        label_clean = label.replace(' ', '_').lower()
        label_dir   = os.path.join(self.config.ALL_OBJECTS_FOLDER, label_clean)
        os.makedirs(label_dir, exist_ok=True)

        filename = f'{label_clean}_{uuid.uuid4().hex[:8]}.jpg'
        filepath = os.path.join(label_dir, filename)
        cv2.imwrite(filepath, image)

        record = {
            'id':         uuid.uuid4().hex,
            'label':      label,
            'filename':   filename,
            'filepath':   filepath,
            'web_path':   f'/all_objects/{label_clean}/{filename}',
            'confidence': confidence,
            'timestamp':  datetime.now().isoformat(),
            'frame_info': frame_info,
        }
        self.all_objects.append(record)
        self._save_db(self.all_objects, self.all_db_path)
        return record

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_all_objects(self, db_type: str = 'all') -> list:
        if db_type == 'all':
            return list(reversed(self.all_objects))
        elif db_type == 'unique':
            return list(reversed(self.unique_objects))
        elif db_type == 'clustered':
            return list(reversed(self.clustered_objects))
        return []

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict:
        label_counts: dict = {}
        for obj in self.all_objects:
            label_counts[obj['label']] = label_counts.get(obj['label'], 0) + 1
        return {
            'all_objects':    len(self.all_objects),
            'unique_objects': len(self.unique_objects),
            'clusters':       len(self.clustered_objects),
            'label_counts':   label_counts,
        }

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _image_hash(self, image: np.ndarray, size: int = 8) -> np.ndarray:
        """Average-hash for fast similarity comparison."""
        gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
        return resized > resized.mean()

    def _similarity(self, img_a: np.ndarray, img_b: np.ndarray) -> float:
        """Return [0, 1] similarity score between two images."""
        try:
            ha = self._image_hash(img_a)
            hb = self._image_hash(img_b)
            return float(np.sum(ha == hb)) / ha.size
        except Exception:
            return 0.0

    def process_discard_duplicates(self, delete_original: bool = False) -> dict:
        """Find unique objects among all_objects and populate unique_objects."""
        try:
            unique_records: list = []
            unique_images:  list = []
            threshold = self.config.UNIQUE_SIMILARITY_THRESHOLD

            for record in self.all_objects:
                filepath = record.get('filepath', '')
                if not os.path.exists(filepath):
                    continue
                img = cv2.imread(filepath)
                if img is None:
                    continue

                is_dup = any(self._similarity(img, ex) >= threshold
                             for ex in unique_images)
                if not is_dup:
                    unique_images.append(img)
                    label_clean = record['label'].replace(' ', '_').lower()
                    dest_dir    = os.path.join(self.config.UNIQUE_OBJECTS_FOLDER, label_clean)
                    os.makedirs(dest_dir, exist_ok=True)
                    dest_path = os.path.join(dest_dir, record['filename'])
                    shutil.copy2(filepath, dest_path)

                    ur              = dict(record)
                    ur['filepath']  = dest_path
                    ur['web_path']  = f'/unique_objects/{label_clean}/{record["filename"]}'
                    unique_records.append(ur)

            self.unique_objects = unique_records
            self._save_db(self.unique_objects, self.unique_db_path)

            return {
                'success': True,
                'stats': {
                    'total_processed':   len(self.all_objects),
                    'unique_found':      len(unique_records),
                    'duplicates_removed': len(self.all_objects) - len(unique_records),
                },
            }
        except Exception as exc:
            print(f'Error in deduplication: {exc}')
            import traceback; traceback.print_exc()
            return {'success': False, 'error': str(exc)}

    # ==================================================================
    # Clustering
    # ==================================================================

    # ------------------------------------------------------------------
    # 1. Feature extraction
    # ------------------------------------------------------------------

    def _init_extractor(self) -> None:
        """
        Lazy-load a pretrained ResNet-18 as a fixed-weight feature extractor.

        The final FC layer is replaced with Identity so the network outputs
        raw 512-dim semantic embeddings instead of class probabilities.
        These embeddings encode shape, texture, and appearance in a way
        that hand-crafted features (pixel histograms, gradients) simply
        cannot match.

        Falls back to HOG + spatial HSV colour pyramid if torchvision
        is unavailable.
        """
        if getattr(self, '_extractor_ready', False):
            return
        try:
            import torch
            import torchvision.transforms as T
            from torchvision.models import resnet18, ResNet18_Weights

            net    = resnet18(weights=ResNet18_Weights.DEFAULT)
            net.fc = torch.nn.Identity()   # 512-dim output
            net.eval()

            self._net       = net
            self._torch     = torch
            self._transform = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std =[0.229, 0.224, 0.225]),
            ])
            self._use_deep = True
            print('[Clustering] Feature extractor: ResNet-18 (512-dim deep embeddings)')
        except Exception as exc:
            self._use_deep = False
            print(f'[Clustering] ResNet-18 unavailable ({exc}); using HOG+colour fallback.')
        self._extractor_ready = True

    def _embed_batch(self, images: list) -> np.ndarray:
        """
        Return (N, D) float32 feature matrix for a list of BGR images.
        Deep path  : ResNet-18 → 512-dim, batch-processed on CPU.
        Fallback   : HOG + spatial HSV pyramid per image.
        """
        self._init_extractor()
        if self._use_deep:
            return self._deep_embed(images)
        return np.array([self._cv_embed(img) for img in images], dtype=np.float32)

    def _deep_embed(self, images: list) -> np.ndarray:
        """Batch-forward images through ResNet-18 backbone; L2-normalise."""
        torch   = self._torch
        tensors = [self._transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                   for img in images]
        batch   = torch.stack(tensors)                    # (N, 3, 224, 224)
        with torch.no_grad():
            feats = self._net(batch).numpy()              # (N, 512)
        norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
        return (feats / norms).astype(np.float32)

    def _cv_embed(self, img: np.ndarray) -> np.ndarray:
        """
        CV fallback feature vector (~1924-dim before PCA):
          • Proper HOG descriptor on 64×64 greyscale  (≈1764-dim)
          • Spatial HSV colour pyramid: global + 4 quadrants (160-dim)
        Much stronger than raw-pixel or simple gradient features.
        """
        img64 = cv2.resize(img, (64, 64))
        gray  = cv2.cvtColor(img64, cv2.COLOR_BGR2GRAY)

        # ── HOG ───────────────────────────────────────────────────────
        hog      = cv2.HOGDescriptor(
            _winSize   =(64, 64), _blockSize=(16, 16),
            _blockStride=(8, 8),  _cellSize =(8,  8),
            _nbins     =9,
        )
        hog_feat = hog.compute(gray).flatten().astype(np.float32)
        hog_feat /= (np.linalg.norm(hog_feat) + 1e-8)

        # ── Spatial HSV pyramid ────────────────────────────────────────
        hsv = cv2.cvtColor(img64, cv2.COLOR_BGR2HSV)

        def _hsv_hist(patch: np.ndarray) -> np.ndarray:
            h = cv2.calcHist([patch], [0], None, [16], [0, 180]).flatten()
            s = cv2.calcHist([patch], [1], None, [8],  [0, 256]).flatten()
            v = cv2.calcHist([patch], [2], None, [8],  [0, 256]).flatten()
            f = np.concatenate([h, s, v]).astype(np.float32)
            return f / (f.sum() + 1e-8)

        colour_parts = [_hsv_hist(hsv)]
        for r in range(2):
            for c in range(2):
                colour_parts.append(
                    _hsv_hist(hsv[r * 32:(r + 1) * 32, c * 32:(c + 1) * 32])
                )
        colour_feat = np.concatenate(colour_parts)        # 5 × 32 = 160-dim

        return np.concatenate([hog_feat, colour_feat])

    # ------------------------------------------------------------------
    # 2. Optimal-k search
    # ------------------------------------------------------------------

    def _find_optimal_k(self, X: np.ndarray, max_k: int) -> int:
        """
        Scan k = 2 … max_k and score each with a weighted combination of:
          • Silhouette score      (60 %) — cluster separation & cohesion
          • Calinski-Harabász     (40 %) — between/within cluster variance ratio

        Both scores are min-max normalised before combining so their
        different scales don't bias the decision.
        silhouette_score is capped at 500 samples for speed.
        """
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.metrics import silhouette_score, calinski_harabasz_score

        n     = len(X)
        max_k = min(max_k, n - 1)
        if max_k < 2:
            return 1

        sil_list, ch_list, k_list = [], [], []
        sample = min(500, n)

        for k in range(2, max_k + 1):
            try:
                km   = MiniBatchKMeans(n_clusters=k, random_state=42,
                                       n_init='auto', batch_size=max(256, k * 10))
                lbls = km.fit_predict(X)
                if len(set(lbls)) < 2:
                    continue
                sil_list.append(
                    silhouette_score(X, lbls, sample_size=sample, random_state=42))
                ch_list.append(calinski_harabasz_score(X, lbls))
                k_list.append(k)
            except Exception:
                continue

        if not k_list:
            return 2

        def _norm(arr: np.ndarray) -> np.ndarray:
            rng = arr.max() - arr.min()   # arr.ptp() removed in NumPy 2.0
            return (arr - arr.min()) / (rng + 1e-9)

        sil = np.array(sil_list, dtype=np.float64)
        ch  = np.array(ch_list,  dtype=np.float64)
        combined = 0.6 * _norm(sil) + 0.4 * _norm(ch)
        return k_list[int(np.argmax(combined))]

    # ------------------------------------------------------------------
    # 3. Copy helper
    # ------------------------------------------------------------------

    def _copy_to_clustered(self, record: dict,
                            cluster_id: int, cluster_size: int) -> 'dict | None':
        """Copy the representative image to the clustered folder."""
        try:
            label_clean = record['label'].replace(' ', '_').lower()
            dest_dir    = os.path.join(self.config.CLUSTERED_OBJECTS_FOLDER, label_clean)
            os.makedirs(dest_dir, exist_ok=True)
            dest_path   = os.path.join(dest_dir, record['filename'])
            src         = record.get('filepath', '')
            if os.path.exists(src):
                shutil.copy2(src, dest_path)
            rep = dict(record)
            rep['cluster_id']   = cluster_id
            rep['cluster_size'] = cluster_size
            rep['filepath']     = dest_path
            rep['web_path']     = (f"/clustered_objects/{label_clean}"
                                   f"/{record['filename']}")
            return rep
        except Exception as exc:
            print(f'_copy_to_clustered error: {exc}')
            return None

    # ------------------------------------------------------------------
    # 4. Main entry point
    # ------------------------------------------------------------------

    def cluster_similar_objects(self, n_clusters: 'int | None' = None,
                                use_optimal: bool = True) -> dict:
        """
        High-quality visual clustering pipeline
        ────────────────────────────────────────
        Step 1 — Deep feature extraction (ResNet-18 backbone, batched)
                 Objects with similar appearance → nearby embeddings,
                 regardless of colour variation or minor pose change.
                 Falls back to HOG + spatial HSV pyramid if torchvision
                 is not installed.

        Step 2 — PCA (≤64 components)
                 Removes redundant/noisy dimensions; speeds up distance
                 computations significantly.

        Step 3 — L2 normalisation
                 Converts Euclidean distance to cosine similarity, which
                 works better in high-dimensional embedding space.

        Step 4 — Label-aware clustering
                 Each YOLO class (person, car, cat …) is clustered
                 independently so semantically different objects are NEVER
                 placed in the same cluster.

        Step 5 — Automatic optimal-k (silhouette + Calinski-Harabász)
                 Finds the natural number of visual sub-groups per class
                 when use_optimal=True.

        Step 6 — MiniBatchKMeans
                 Scales efficiently to thousands of objects.
        """
        try:
            from sklearn.cluster       import MiniBatchKMeans
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import normalize

            # ── Source objects ─────────────────────────────────────────
            source = self.unique_objects if self.unique_objects else self.all_objects
            if not source:
                return {
                    'success': False,
                    'error': ('No objects to cluster. '
                              'Run detection (and optionally deduplication) first.'),
                }

            # ── Load images & group by YOLO class label ────────────────
            label_groups: dict = {}
            for rec in source:
                fp = rec.get('filepath', '')
                if not os.path.exists(fp):
                    continue
                img = cv2.imread(fp)
                if img is None:
                    continue
                label_groups.setdefault(rec['label'], []).append((rec, img))

            if not label_groups:
                return {'success': False, 'error': 'No readable images found on disk.'}

            total_src = sum(len(v) for v in label_groups.values())
            print(f'[Clustering] {total_src} objects across '
                  f'{len(label_groups)} class(es): {list(label_groups.keys())}')

            # ── Cluster each class independently ───────────────────────
            clustered: list = []
            global_id: int  = 0
            total:     int  = 0
            summary:   list = []

            for lbl, items in sorted(label_groups.items()):
                n_items = len(items)
                total  += n_items
                recs    = [r   for r, _   in items]
                imgs    = [img for _,  img in items]

                print(f'[Clustering]  → "{lbl}": {n_items} object(s)')

                # Single object — trivially its own cluster
                if n_items == 1:
                    rep = self._copy_to_clustered(recs[0], global_id, 1)
                    if rep:
                        clustered.append(rep)
                        summary.append({'label': lbl, 'k': 1, 'objects': 1})
                        global_id += 1
                    continue

                # ── Step 1: Extract deep (or CV fallback) features ─────
                X = self._embed_batch(imgs)            # (N, D)

                # ── Step 2: PCA dimensionality reduction ───────────────
                n_comp = min(64, n_items - 1, X.shape[1])
                if n_comp >= 2:
                    X = PCA(n_components=n_comp,
                            random_state=42).fit_transform(X)

                # ── Step 3: L2 normalise ───────────────────────────────
                X = normalize(X, norm='l2')

                # ── Step 4: Determine k for this class ─────────────────
                max_k = min(
                    n_items,
                    n_clusters if n_clusters is not None
                    else self.config.DEFAULT_CLUSTER_N_CLUSTERS,
                )
                max_k = max(2, max_k)

                if use_optimal and n_items >= 4:
                    k = self._find_optimal_k(X, max_k)
                else:
                    k = min(max_k, n_items)
                k = max(1, min(k, n_items))

                print(f'[Clustering]     k={k} (max allowed={max_k})')

                # k == 1: collapse entire class to one representative
                if k == 1:
                    rep = self._copy_to_clustered(recs[0], global_id, n_items)
                    if rep:
                        clustered.append(rep)
                        summary.append({'label': lbl, 'k': 1, 'objects': n_items})
                        global_id += 1
                    continue

                # ── Step 5: MiniBatchKMeans ────────────────────────────
                km = MiniBatchKMeans(
                    n_clusters  = k,
                    random_state= 42,
                    n_init      = 'auto',
                    batch_size  = max(256, k * 10),
                )
                labels_arr = km.fit_predict(X)

                # Collect cluster representatives and their feature vectors
                cluster_reps = {}
                for cid in range(k):
                    idx = [i for i, lb in enumerate(labels_arr) if lb == cid]
                    if not idx:
                        continue
                    centroid = km.cluster_centers_[cid]
                    dists    = [np.linalg.norm(X[i] - centroid) for i in idx]
                    best_i   = idx[int(np.argmin(dists))]
                    cluster_reps[cid] = {
                        'best_i': best_i,
                        'indices': idx,
                        'feat': X[best_i]
                    }

                # ── Step 6: Post-clustering merge of near-identical clusters ──
                # If representatives are too similar (cosine similarity/dot product >= threshold), merge them
                merged = True
                threshold = self.config.UNIQUE_SIMILARITY_THRESHOLD
                while merged:
                    merged = False
                    cids = list(cluster_reps.keys())
                    to_merge = None
                    for i in range(len(cids)):
                        for j in range(i + 1, len(cids)):
                            cid1 = cids[i]
                            cid2 = cids[j]
                            # Cosine similarity is simple dot product since X features are L2 normalized
                            sim = float(np.dot(cluster_reps[cid1]['feat'], cluster_reps[cid2]['feat']))
                            if sim >= threshold:
                                to_merge = (cid1, cid2)
                                break
                        if to_merge:
                            break
                    
                    if to_merge:
                        cid1, cid2 = to_merge
                        # Merge the one with fewer items into the one with more items
                        if len(cluster_reps[cid1]['indices']) >= len(cluster_reps[cid2]['indices']):
                            keep, discard = cid1, cid2
                        else:
                            keep, discard = cid2, cid1

                        # Merge indices
                        cluster_reps[keep]['indices'].extend(cluster_reps[discard]['indices'])
                        # Recalculate representative (closest to centroid of combined features)
                        combined_idx = cluster_reps[keep]['indices']
                        combined_feats = X[combined_idx]
                        mean_feat = combined_feats.mean(axis=0)
                        mean_feat /= (np.linalg.norm(mean_feat) + 1e-8)
                        
                        dists = [np.linalg.norm(X[idx_val] - mean_feat) for idx_val in combined_idx]
                        new_best_i = combined_idx[int(np.argmin(dists))]
                        
                        cluster_reps[keep]['best_i'] = new_best_i
                        cluster_reps[keep]['feat'] = X[new_best_i]
                        
                        # Remove discarded cluster
                        del cluster_reps[discard]
                        merged = True

                # Save finalized clusters
                for new_cid, rep_info in enumerate(cluster_reps.values()):
                    best_idx = rep_info['best_i']
                    rep = self._copy_to_clustered(recs[best_idx], global_id, len(rep_info['indices']))
                    if rep:
                        clustered.append(rep)
                        global_id += 1

                summary.append({'label': lbl, 'k': len(cluster_reps), 'objects': n_items})

            self.clustered_objects = clustered
            self._save_db(self.clustered_objects, self.cluster_db_path)

            feat_type = ('ResNet-18 deep embeddings'
                         if getattr(self, '_use_deep', False)
                         else 'HOG + spatial HSV pyramid')
            print(f'[Clustering] Done — {global_id} cluster(s) from {total} '
                  f'object(s) using {feat_type}.')

            return {
                'success':               True,
                'n_clusters':            global_id,
                'total_objects':         total,
                'cluster_representatives': len(clustered),
                'feature_type':          feat_type,
                'per_class':             summary,
            }

        except ImportError:
            return {'success': False,
                    'error': 'scikit-learn is required. Run: pip install scikit-learn'}
        except Exception as exc:
            print(f'[Clustering] Error: {exc}')
            import traceback; traceback.print_exc()
            return {'success': False, 'error': str(exc)}

    # ------------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------------

    def clear_database(self, folder_type: str = 'all') -> dict:
        """Clear one or all object collections."""
        try:
            folder_map = {
                'all':       (self.config.ALL_OBJECTS_FOLDER,       self.all_db_path),
                'unique':    (self.config.UNIQUE_OBJECTS_FOLDER,    self.unique_db_path),
                'clustered': (self.config.CLUSTERED_OBJECTS_FOLDER, self.cluster_db_path),
            }
            targets = (list(folder_map.items()) if folder_type == 'everything'
                       else [(folder_type, folder_map[folder_type])]
                       if folder_type in folder_map else [])

            for key, (folder, db_path) in targets:
                if os.path.exists(folder):
                    shutil.rmtree(folder)
                    os.makedirs(folder, exist_ok=True)
                if key == 'all':
                    self.all_objects = []
                    self._save_db(self.all_objects, db_path)
                elif key == 'unique':
                    self.unique_objects = []
                    self._save_db(self.unique_objects, db_path)
                elif key == 'clustered':
                    self.clustered_objects = []
                    self._save_db(self.clustered_objects, db_path)

            return {'success': True, 'message': f'Cleared {folder_type} database'}
        except Exception as exc:
            return {'success': False, 'error': str(exc)}
