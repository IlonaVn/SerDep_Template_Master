from pathlib import Path

computer = 'laptop' #laptop, setup, lab, office, MPI; define computer on which to run experiment

### Analysis ###
exclude_outliers = True
split = [90,45]

### Path definitions ###
if computer == 'laptop':
    path_str = 'C:/Users/illil/Documents/01 Uni Göttingen/01 Masterarbeit/Master_Data/Experiment Memory Templates/Templates_Exp_Code_Cleaned'
    path_main = Path(path_str)
elif computer == 'office':
    path_main = Path('W:/ilona.vieten/Experiment Memory Templates/Templates_Exp_Code')
elif computer == 'setup':
    path_main = Path('C:/Users/Caspar/Desktop/ilona/Templates_Exp_Code')
elif computer == 'MPI':
    path_main = Path('P:/2021-0306-MemoryDistortions/MemTemplColor')

path_expScripts = path_main / 'ExpCode/' #experimental code
path_anScripts = path_main / 'AnCode/' #analysis code
path_rawDat = path_str + '/RawData/' #raw data
path_edfDat = path_main / 'EyetrackingData/' #raw eyetracking data
path_results = path_main / 'Results'

### File names ###

filename_list = ["MemItem_Rep_performance.csv", "Template_Rep_performance.csv", "Template_Judg_performance.csv", "Modelfree_dependence_measure.csv"]