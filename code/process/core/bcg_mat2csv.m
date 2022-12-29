
function [] = bcg_mat2csv(path_raw)
    % This script formats the data in mat files to csv files.
    % Thus, the data can be processed with python later on.
    if nargin > 0
        path = path_raw;
    else
        path = '.';
    end
    
    disp('Formating subjects info...')
    mat = load(strcat(path,'/Bed_System_Database.mat'));
    
    writetable(mat.Bed_System_Database(:,[1:5,7:8]), strcat(path,'/bcg_pat_info.csv'));
    
    disp('Formating subjects signal...')
    mat2 = load(strcat(path,'/Preprocessed_Database.mat'));
    
    mkdir(strcat(path,'pats/'))
    for c = 1:40
        disp(strcat('Sub',num2str(c)))
        writetable(mat.Bed_System_Database.RawData{c,1}, strcat(path,'pats/pat',num2str(c),'.csv'));
        writetable(mat2.Preprocessed_Database.FilteredData{c,1}, strcat(path,'pats/pat',num2str(c),'pred.csv'));
    end
end