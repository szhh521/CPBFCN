import os

#seq_len = ['101','201','501']
seq_len = ['101']
loss_threshold = [0.3,0.5,0.7]
loss_function = ['BCE','HNML']
model_select = ['FCN','FCNA']

data_path = '/home/szhen/Downloads/CPBPFCN/all_data'

for sl in seq_len:
    for lf in loss_function:
        if lf == 'BCE':
            for ms in model_select:
                cmd = 'python run_motif_parm_1.py -d ' + data_path + ' -ld ' + sl + ' -ls ' + lf + ' -ms ' + ms
                print(cmd)
                os.system(cmd)
        elif lf == 'HNML':
            for lt in loss_threshold:
                for ms in model_select:
                    cmd = 'python run_motif_parm_1.py -d ' + data_path + ' -ld ' + sl + ' -ls ' + lf + ' -ms ' + ms + ' -lt ' + str(lt)
                    print(cmd)
                    os.system(cmd)
