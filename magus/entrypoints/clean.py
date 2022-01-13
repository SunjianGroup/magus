import os

remain = ['results', 'inputFold', 'input.yaml', 'Seeds']
def clean(*args, force=False, **kwargs):
    os.system('mv allparameters.yaml log.txt *err* *out* formula_pool results/')
    os.system('cp -r input* results/')
    os.system('rm -rf calcFold')
    if force:
        a = input('use force will only remain files: {}\n' 
                  'Are you sure you want to continue?[y/n]'
                  ''.format(', '.join(remain[:-1]) + 'and ' + remain[-1]))
        if a.lower() == 'y' or a.lower() == 'yes':
            print('zhen you ni de')
            os.system('rm -rf {}'.format(' '.join([f for f in os.listdir() if f not in remain ])))
    print('clean Done!')
