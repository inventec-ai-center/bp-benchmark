% This script formats the data in mat files to csv files.
% Thus, the data can be processed with python later on.

mat = load('Bed_System_Database.mat');

writetable(mat.Bed_System_Database(:,[1:5,7:8]), 'bcg_pat_info.csv');

mat2 = load('Preprocessed_Database.mat');

for c = 1:40
    writetable(mat.Bed_System_Database.RawData{c,1}, strcat('pats/pat',num2str(c),'.csv'));
    writetable(mat2.Preprocessed_Database.FilteredData{c,1}, strcat('pats/pat',num2str(c),'pred.csv'));
end