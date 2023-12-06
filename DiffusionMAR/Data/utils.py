import os

def unpair_img_paths(af_paths, gt_paths):
    size = len(af_paths)
    print('size of dataset', size)
    
    af_out = []
    gt_out = []
    
    af_patients = {}
    gt_patients = {}
    
    for i, af_path in enumerate(af_paths):
        dir_name = os.path.dirname(af_path)
        patient_number = os.path.basename(dir_name).split('_')[0]
        if patient_number in af_patients.keys():
            #print('add af', patient_number)
            af_out.append(af_path)
            af_patients[patient_number] += 1
        elif patient_number in gt_patients.keys():
            #print('add gt', patient_number)
            gt_out.append(gt_paths[i])
            gt_patients[patient_number] += 1
        elif len(af_out) < (size/2 - 2000):
            #print('NEW af', patient_number)
            af_patients.update({patient_number: 1})
            af_out.append(af_path)
        else:
            #print('NEW gt', patient_number)
            gt_patients.update({patient_number: 1})
            gt_out.append(gt_paths[i])
    
    
    dif = len(af_out) - len(gt_out)  
    print('initial dif (af-gt)', dif)
    best = abs(dif)
    remove_af_data = False
    
    if dif == 0:
        print('it worked in one go, yeaahhh')
        
    # af_out < gt_out
    elif dif < 0: 
        for patient in gt_patients.keys():
            data_n = gt_patients[patient]
            if 2 * data_n == abs(dif):
                patient_to_move = patient
                best = 0
                break
            # new_dif = len(af) + data_n - (len(gt) - data_n) = abs(2*n + dif) 
            elif abs(2 * data_n + dif) < best:
                best = abs(2 * data_n + dif)
                patient_to_move = patient
                if 2 * data_n < abs(dif):
                    remove_af_data = False
                else:
                    remove_af_data = True  
        
        templist = []
        for i, gt_path in enumerate(gt_out):
            dir_name = os.path.dirname(gt_path)
            patient_number = os.path.basename(dir_name).split('_')[0]
            if patient_number == patient_to_move:
                af_out.append(gt_path)
                templist.append(i)
        for i in templist:
            gt_out.pop(i)
        
    # af_out > gt_out
    elif dif > 0:
        for patient in af_patients.keys():
            data_n = af_patients[patient]
            if 2 * data_n == dif:
                patient_to_move = patient
                best = 0
                break
            # new_dif = len(af) - data_n - (len(gt) + data_n) = abs(dif - 2*n)
            elif abs(dif - 2 * data_n) < best:
                best = abs(dif - 2 * data_n)
                patient_to_move = patient
                if 2 * data_n < dif:
                    remove_af_data = True
                else:
                    remove_af_data = False
        
        templist = []
        for i, af_path in enumerate(af_out):
            dir_name = os.path.dirname(af_path)
            patient_number = os.path.basename(dir_name).split('_')[0]
            if patient_number == patient_to_move:
                gt_out.append(af_path)
                templist.append(i)
        for i in templist:
            af_out.pop(i)
            
    print('best dif when moving one patient', best)
    
    for _ in range(best):
        if remove_af_data:
            af_out.pop(-1)
        else:
            gt_out.pop(-1)
    
    return af_out, gt_out

def get_img_paths(path):    
    types = ['Affected', 'Ground truth']
    af_path = os.path.abspath(os.path.join(path, types[0]))
    gt_path = os.path.abspath(os.path.join(path, types[1]))
    
    af_out = []
    gt_out = []
    folders = os.listdir(af_path)
    for folder in folders:
        af_path_2 = os.path.join(af_path, folder)
        gt_path_2 = os.path.join(gt_path, folder)
        imgs = os.listdir(af_path_2)
        for img in imgs:
            af_out.append(os.path.join(af_path_2, img))
            gt_out.append(os.path.join(gt_path_2, img))
    return af_out, gt_out

def get_test_data_paths(path):   
    types = ['Affected', 'Ground truth']
    af_path = os.path.abspath(os.path.join(path, types[0]))
    gt_path = os.path.abspath(os.path.join(path, types[1]))
    
    af_out = []
    gt_out = []
    items = os.listdir(af_path)
    for img in items:
        af_out.append(os.path.join(af_path, img))
        gt_out.append(os.path.join(gt_path, img))
    return af_out, gt_out