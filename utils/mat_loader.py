import torch
import scipy.io as sio
from torch.utils.data import Dataset
import numpy as np
import scipy.sparse as sp

class MatDataset(Dataset):
    def __init__(self, path, data_key='X', label_key='Y', device='cpu'):
        try:
            mat_data = sio.loadmat(path)
        except Exception as e:
            raise ValueError(f"can't reload .mat file: {path}. Error: {e}")

        if label_key not in mat_data:
            raise KeyError(f"can't find Y '{label_key}'")
        self.labels = mat_data[label_key].flatten()
        if self.labels.min() == 1: self.labels = self.labels - 1
        self.labels = self.labels.astype(np.int64)
        self.labels_tensor = torch.from_numpy(self.labels).to(device)

        if data_key not in mat_data: raise KeyError(f"can't find X '{data_key}'")
        raw_X = mat_data[data_key]
        self.views = []
        if raw_X.dtype == 'object': 
            num_views = raw_X.shape[0]
            
            for i in range(num_views):
                v_data = raw_X[i, 0] 
                v_data = self._preprocess(v_data)
                v_tensor = torch.from_numpy(v_data).to(device)
                self.views.append(v_tensor)
                print(f"  - View {i+1}: {v_tensor.shape}")
        else:
            raise ValueError("data is not a Cell Array")

    def _preprocess(self, data):
        import scipy.sparse as sp
    
        if sp.issparse(data):
            try:
                data = data.toarray()
            except Exception:
                data = np.array(data.todense(), dtype=object)
                
        data = np.asarray(data, dtype=np.float32)
    
        if data.shape[0] != self.labels.shape[0] and data.shape[1] == self.labels.shape[0]:
            data = data.T
    
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        data = data / norms
        
        return data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return [v[idx] for v in self.views], self.labels_tensor[idx], idx