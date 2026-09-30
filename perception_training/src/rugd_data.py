import os, cv2, yaml, numpy as np, torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from preprocessing import resize_image_and_mask, normalize_image

PT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cls = yaml.safe_load(open(PT + "/configs/classes.yaml")); tr = yaml.safe_load(open(PT + "/configs/training.yaml"))
NAMES = [c["name"] for c in cls["classes"]]; IGNORE = cls["ignore_index"]
assert NAMES[IGNORE] == "void" and len(NAMES) == cls["num_classes"] == 20

RUGD_BASE = "/kaggle/input/datasets/moysuau/rugd-dataset/RUGD"
HOLD = {"park-8", "trail-15", "trail-7"}          # 970 frames, never trained on
RUGD = {(0,0,0):"void",(108,64,20):"dirt",(255,229,204):"sand",(0,102,0):"grass",(0,255,0):"tree",
 (0,153,153):"pole",(0,128,255):"water",(0,0,255):"sky",(255,255,0):"vehicle",(255,0,127):"container",
 (64,64,64):"asphalt",(255,128,0):"gravel",(255,0,0):"building",(153,76,0):"mulch",(102,102,0):"rock-bed",
 (102,0,0):"log",(0,255,128):"bicycle",(204,153,255):"person",(102,0,204):"fence",(255,153,204):"bush",
 (0,102,102):"sign",(153,204,255):"rock",(102,255,255):"bridge",(101,101,11):"concrete",(114,85,47):"picnic-table"}
# None = ignored. sand, gravel, mulch and rock-bed have no clean class in our taxonomy, so they are ignored.
MAP = {"void":None,"dirt":"dirt","grass":"grass","tree":"tree","pole":"pole","water":"water","sky":"sky",
       "vehicle":"vehicle","container":"object","asphalt":"asphalt","building":"building","log":"log",
       "person":"person","fence":"fence","bush":"bush","concrete":"concrete","bicycle":"object",
       "sign":"object","picnic-table":"object","rock":None,"bridge":None,
       "sand":None,"gravel":None,"mulch":None,"rock-bed":None}
assert set(MAP) == set(RUGD.values()) and all(v is None or v in NAMES for v in MAP.values())

lut_arr = np.full(1 << 24, IGNORE, np.uint8)
for (r, g, b), name in RUGD.items():
    if MAP[name]: lut_arr[(r << 16) | (g << 8) | b] = NAMES.index(MAP[name])

def decode(path):
    a = np.array(Image.open(path).convert("RGB")).astype(np.int64)
    return lut_arr[(a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]]

class RUGDSeg(Dataset):
    def __init__(self, base=RUGD_BASE, include=None, exclude=None, hflip=False):
        img_dir = f"{base}/3.after join creek/image"
        col = {}
        for d in (f"{base}/1.indexLabel-color", f"{base}/2.indexLabel- color-creek"):
            col.update({f: f"{d}/{f}" for f in os.listdir(d)})
        self.items = []
        for f in sorted(os.listdir(img_dir)):
            seq = f.rsplit("_", 1)[0]
            if (include and seq not in include) or (exclude and seq in exclude): continue
            assert f in col, f"no color mask for {f}"
            self.items.append((f"{img_dir}/{f}", col[f]))
        self.size, self.hflip = tuple(tr["input_size"]), hflip

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        ip, mp = self.items[i]
        image = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
        image, mask = resize_image_and_mask(image, decode(mp), size=self.size)
        if self.hflip and torch.rand(1).item() < 0.5:
            image, mask = image[:, ::-1], mask[:, ::-1]
        image = normalize_image(np.ascontiguousarray(image), mean=tuple(tr["normalize_mean"]), std=tuple(tr["normalize_std"]))
        return torch.from_numpy(image).permute(2, 0, 1).float(), torch.from_numpy(np.ascontiguousarray(mask)).long()

def build(base=RUGD_BASE):
    # no flip: matches the RELLIS side on main, which has no augmentation
    train_full = RUGDSeg(base, exclude=HOLD)
    return train_full, Subset(train_full, range(0, len(train_full), 3)), RUGDSeg(base, include=HOLD)
