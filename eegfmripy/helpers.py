import nibabel as nib 



"""
read .vmrk file, get all unique events, and return the following lists

list1: event ids
list2: event latencies
"""

def read_vmrk(path):

    with open(path) as f:
        content = f.readlines()
    
    event_ids = []
    event_lats = []
    for line in content:
        if line[0:2] == 'Mk':
            a = line.split(',')
            event_ids.append(a[1])
            event_lats.append(int(a[2]))
            
    return event_ids, event_lats


"""
fmri_info: get information relating to repetition time (TR), number of slices
per image (n_slices), and number of volumes per scan (n_volumes)

"""
def fmri_info(path):
    fmri = nib.load(path)
    hdr = fmri.get_header()
    
    TR = hdr.get_zooms()[3] 
    n_slices = hdr.get_n_slices()
    n_volumes = hdr.get_data_shape()[3]
    
    return TR, n_slices, n_volumes


