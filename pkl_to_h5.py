'''
Errors stored as gzip pickle files --> convert to h5.
'''

import gzip
import pickle
import h5py
import numpy as np
from argparse import ArgumentParser as AP
from pathlib import Path
from glob import glob

def process(rdir):
  files = Path(rdir).glob("error_data_*.pkl")

  for file in files:
    with gzip.open(file, 'rb') as f:
      print(file)
      data = pickle.load(f)
      #print(data)
      hpref = file.name.replace(".pkl", "")      
      
      for key in data:
        hfile = file.with_name(f"{hpref}_{key}.h5")
        with h5py.File(hfile, "w") as hf:
          hf.create_dataset("error", 
                            data=data[key], 
                            dtype=np.float16,
                            chunks = (1, data[key].shape[1], data[key].shape[2]),
                            compression="gzip", 
                            compression_opts=4,                             
                            shuffle = True)
      
    







if __name__ == "__main__":
  #rdir = "I:\My Drive\Pubdata\Results\Scene7_illusions\EOS6D_A_Left\fl_70mm\inference\F16.0\err_GT"
  parser = AP()
  parser.add_argument("--rdir", required=True)
  args = parser.parse_args()
  process(args.rdir)

