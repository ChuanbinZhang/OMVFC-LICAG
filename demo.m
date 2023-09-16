clear;
clc;
close all;

data_dir = './datasets';

dataset = 'WebKB';

fprintf("%12s\t%6s\t%6s\t%6s\t%s\n", "Methods", "Purity", "ARI", "NMI", "Runtime");

% Latent information extraction based on cross-view anchor graph (LICAG)
result = LICAGC(data_dir, dataset, 0.3, 5);
fprintf("%12s\t%6.2f\t%6.2f\t%6.2f\t%7.2f\n", "LICAG", result(1)*100, result(2)*100, result(3)*100, result(4));

% One-step multi-view fuzzy clustering (OMVFC)
result = OMVFC_LICAG(data_dir, dataset, 10000, 0, 0.3, 5);
fprintf("%12s\t%6.2f\t%6.2f\t%6.2f\t%7.2f\n", "OMVFC", result(1)*100, result(2)*100, result(3)*100, result(4));

% Latent information-guided one-step multi-view fuzzy clustering based on cross-view anchor graph (OMVFC-LICAG)
result = OMVFC_LICAG(data_dir, dataset, 10000, 1, 0.3, 5);
fprintf("%12s\t%6.2f\t%6.2f\t%6.2f\t%7.2f\n", "OMVFC-LICAG", result(1)*100, result(2)*100, result(3)*100, result(4));

