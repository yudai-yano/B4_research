import sys
sys.path.append(rf'C:\Users\yota0\Desktop\yano\program\python\definition')

import make_trcfile

motion_csv = rf'C:\Users\yota0\Desktop\yano\trial_folder\csv_data\test.csv'
motion_trc = rf'C:\Users\yota0\Desktop\yano\trial_folder\trc_data\test.csv'
    

    
make_trcfile.csv_to_trc(motion_csv, motion_trc)