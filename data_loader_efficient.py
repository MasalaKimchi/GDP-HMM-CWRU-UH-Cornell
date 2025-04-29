'''
    The script is modified from the original one to make it more efficient. https://github.com/RiqiangGao/GDP-HMM_AAPMChallenge/blob/main/data_loader.py
    It removes redundant keys and variables saved in the output (or 'data_dict' in the original script).
'''

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import numpy as np
import json
import pdb

from scipy import ndimage 
from toolkit import *

HaN_OAR_LIST = [ 'Cochlea_L', 'Cochlea_R','Eyes', 'Lens_L', 'Lens_R', 'OpticNerve_L', 'OpticNerve_R', 'Chiasim', 'LacrimalGlands', 'BrachialPlexus', 'Brain',  'BrainStem_03',  'Esophagus', 'Lips', 'Lungs', 'Trachea', 'Posterior_Neck', 'Shoulders', 'Larynx-PTV', 'Mandible-PTV', 'OCavity-PTV', 'ParotidCon-PTV', 'Parotidlps-PTV', 'Parotids-PTV', 'PharConst-PTV', 'Submand-PTV', 'SubmandL-PTV', 'SubmandR-PTV', 'Thyroid-PTV', 'SpinalCord_05']

HaN_OAR_DICT = {HaN_OAR_LIST[i]: (i+1) for i in range(len(HaN_OAR_LIST))}

Lung_OAR_LIST = ["PTV_Ring.3-2", "Total Lung-GTV", "SpinalCord",  "Heart",  "LAD", "Esophagus",  "BrachialPlexus",  "GreatVessels", "Trachea", "Body_Ring0-3"]

Lung_OAR_DICT = {Lung_OAR_LIST[i]: (i+10) for i in range(len(Lung_OAR_LIST))}

# ---------------------------  data_loader.py  ---------------------------------
class MyDataset(Dataset):
    """
    Lean version:   returns only the tensors that training / inference / evaluation
                    really need.
    Keys in each item:
        ├─ data           (C, D, H, W)   ← still identical construction
        ├─ label          (1, D, H, W)   ← only when GT dose exists
        ├─ id             (str)
        ├─ ori_img_size   (3 tuple tensor)
        └─ ori_isocenter  (3 tuple tensor)
    """

    def __init__(self, cfig, phase):
        self.cfig  = cfig
        self.phase = phase

        df   = pd.read_csv(cfig['csv_root'])
        df   = df.loc[df['dev_split'] == phase]
        self.data_list  = df['npz_path'].tolist()
        self.site_list  = df['site'].tolist()
        self.cohort_list= df['cohort'].tolist()

        self.scale_dose_Dict = json.load(open(cfig['scale_dose_dict'], 'r'))
        self.pat_obj_dict    = json.load(open(cfig['pat_obj_dict'], 'r'))

    def __len__(self): return len(self.data_list)

    # ------------------------------------------------------------------ #
    def __getitem__(self, idx):
        # ---------- 1. read case & basic pre‑processing -----------------
        npz_path = self.data_list[idx]
        ID       = npz_path.split('/')[-1].replace('.npz', '')
        Patient  = ID.split('+')[0]

        if len(str(Patient)) < 3:
            Patient = f"{Patient:0>3}"

        data_npz = np.load(npz_path, allow_pickle=True)

        In_dict  = dict(data_npz)['arr_0'].item()

        # clip & normalise CT
        In_dict['img'] = np.clip(In_dict['img'],
                                 self.cfig['down_HU'],
                                 self.cfig['up_HU']) / self.cfig['denom_norm_HU']

        ori_img_size = In_dict['img'].shape
        isocenter    = In_dict['isocenter']

        # ---------- 2. (optional) rescale GT dose -----------------------
        if 'dose' in In_dict:
            pdose = self.scale_dose_Dict[Patient]['PTV_High']['PDose']
            opt   = self.scale_dose_Dict[Patient]['PTV_High']['OPTName']
            In_dict['dose'] = (In_dict['dose'] * In_dict['dose_scale'])
            norm = pdose / (np.percentile(In_dict['dose'][In_dict[opt]>0], 3)+1e-5)
            In_dict['dose'] = np.clip(
                In_dict['dose'] * norm / self.cfig['dose_div_factor'],
                0, pdose*1.2)

        # ---------- 3. beam & angle plates (needed for data channels) ---
        if 'angle_plate' not in In_dict:         # fallback
            In_dict['angle_plate'] = np.ones(In_dict['img'][0].shape)

        ang3d = np.zeros_like(In_dict['img'])
        z0, z1 = int(isocenter[0])-5, int(isocenter[0])+5
        z0, z1 = max(0,z0), min(ang3d.shape[0], z1)
        plate  = np.repeat(In_dict['angle_plate'][np.newaxis, :, :], z1-z0, axis=0)
        if plate.shape[1:] != ang3d.shape[1:]:
            plate = ndimage.zoom(plate,
                    (1, ang3d.shape[1]/plate.shape[1], ang3d.shape[2]/plate.shape[2]),
                    order=0)
        ang3d[z0:z1] = plate
        In_dict['angle_plate'] = ang3d

        KEYS = list(In_dict.keys())
        for key in In_dict.keys(): 
            if isinstance(In_dict[key], np.ndarray) and len(In_dict[key].shape) == 3:
                In_dict[key] = torch.from_numpy(In_dict[key].astype('float'))[None]
            else:
                KEYS.remove(key)

        # ---------- 4. spatial augmentation --------------------------------
        aug = (tr_augmentation if self.phase=='train' and
               self.cfig.get('with_aug', True) else tt_augmentation)
        In_dict = aug(KEYS, self.cfig['in_size'], self.cfig['out_size'], isocenter)(In_dict)

        # ---------- 5. build INPUT CHANNEL STACK ---------------------------
        # CatStructures is always False → use *combined* masks only
        site_flag = self.site_list[idx] < 1.5
        OAR_LIST  = HaN_OAR_LIST if site_flag else Lung_OAR_LIST
        OAR_DICT  = HaN_OAR_DICT if site_flag else Lung_OAR_DICT
        need_oars = self.pat_obj_dict.get(Patient, OAR_LIST)

        # If 'use_dict' is True, we compute the distance map for the OARs
        # If 'use_dict' is False, we one-hot encode the OARs
        if self.cfig['use_dist']:
            comb_oar, _         = combine_oar_dist(In_dict, need_oars,
                                            self.cfig['norm_oar'], OAR_DICT)
        else:
            comb_oar, _         = combine_oar(In_dict, need_oars,
                                            self.cfig['norm_oar'], OAR_DICT)

        opt_dict, dose_dict = {}, {}
        for k,v in self.scale_dose_Dict[Patient].items():
            if k not in ['PTV_High','PTV_Mid','PTV_Low']: continue
            pd = v['PDose']/self.cfig['dose_div_factor']
            opt_dict[v['OPTName']]   = pd
            tgt = v['OPTName'] if k=='PTV_High' else v['StructName']
            dose_dict[tgt] = pd

        comb_optptv,_,_ = combine_ptv(In_dict,opt_dict)
        comb_ptv,_,_    = combine_ptv(In_dict,dose_dict)

        # -------- prompt → keep same numeric layout but don't store it ----
        prompt = torch.tensor([In_dict['isVMAT'],
                               len(opt_dict),
                               self.site_list[idx],
                               self.cohort_list[idx]]).float()
        prompt_ext = prompt[None,:,None,None].repeat(
            1, self.cfig['out_size'][0]//4,
            self.cfig['out_size'][1],
            self.cfig['out_size'][2])

        # final stack (same order & channel‑count as original pipeline)
        data_tensor = torch.cat((
            comb_optptv, comb_ptv, comb_oar,
            In_dict['Body'], In_dict['img'],
            In_dict['beam_plate'], In_dict['angle_plate'],
            prompt_ext), dim=0)

        # ---------- 6. assemble the *minimal* dictionary -------------------
        out = {
            'data'          : data_tensor,
            'id'            : ID,
            'ori_img_size'  : torch.tensor(ori_img_size),
            'ori_isocenter' : torch.tensor(isocenter)
        }
        if 'dose' in In_dict:
            out['label'] = In_dict['dose'] * In_dict['Body']

        return out
# ---------------------------------------------------------------------------

class GetLoader:
    def __init__(self, cfig): self.cfig = cfig
    def train_dataloader(self):
        return DataLoader(MyDataset(self.cfig,'train'),
                          batch_size=self.cfig['train_bs'],
                          shuffle=True , num_workers=self.cfig['num_workers'])
    def val_dataloader(self):
        return DataLoader(MyDataset(self.cfig,'valid'),
                          batch_size=self.cfig['val_bs'],
                          shuffle=False, num_workers=self.cfig['num_workers'])
    def test_dataloader(self):
        return DataLoader(MyDataset(self.cfig,'test'),
                          batch_size=self.cfig['val_bs'],
                          shuffle=False, num_workers=self.cfig['num_workers'])
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    cfig = {
            'train_bs': 2,
             'val_bs': 2, 
             'num_workers': 0, 
             'csv_root': '/data/meta_data_test.csv',
             'scale_dose_dict': '/data/PTV_DICT.json',
             'pat_obj_dict': '/data/Pat_Obj_DICT.json',
             'down_HU': -1000,
             'up_HU': 1000,
             'denom_norm_HU': 500,
             'in_size': (96, 128, 144), 
             'out_size': (96, 128, 144), 
             'norm_oar': True,
             'CatStructures': False,
             'dose_div_factor': 10,
             'use_dist': False
             }
    
    loaders = GetLoader(cfig)
    train_loader = loaders.train_dataloader()

    for i, data in enumerate(train_loader):

        pdb.set_trace()
        print(data['id'])
        print (data['data'].shape, data['label'].shape, data['ori_isocenter'], data['ori_img_size'])
