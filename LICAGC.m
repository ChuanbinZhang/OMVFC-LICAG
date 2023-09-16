function [result] = LICAGC(data_dir, dataset, rate, dim, n_clusters, max_iter, verbose)
default_parameters = [
    "data_dir",     "'datasets'";
    "dataset",      "";
    "rate",         0.1;
    "dim"           10;
    "n_clusters",	0;
    "max_iter",     100;
    "verbose",      0;
];

for i=4:size(default_parameters, 1)
    if nargin < i
        s = strcat(default_parameters(i, 1), "=", default_parameters(i, 2), ";");
        eval(s);
    end
end

load(fullfile(data_dir, dataset), "X", "y");

if n_clusters < 2
    n_clusters = length(unique(y));
end

for i = 1:length(X)
    N = normalization(X{i},"range",1);
    X{i} = N';
end

n_anchors = floor(rate * length(y));
n_neighbors = n_anchors - 1;

LE_path = './SpectralEmbedding';
if ~exist(LE_path, 'dir')
    mkdir(LE_path)
end
LE_path = fullfile(LE_path, sprintf('%s', dataset));
if ~exist(LE_path, 'dir')
    mkdir(LE_path)
end

tic;
LE_path = fullfile(LE_path, sprintf('%d_%0.3f.mat', dim, rate));
if exist(LE_path, 'file')
    load(LE_path, "H");
else
    H = LICAG(X, dim, n_anchors, n_neighbors);
    save(LE_path, "H");
end

if verbose
    for i=1:size(default_parameters, 1)
        eval(default_parameters(i,1));
    end
end

[prediction] = litekmeans(H', n_clusters, 'MaxIter', max_iter);
rt = toc;

result = ClusteringMeasure(y, prediction);
result(length(result)+1)=rt;
end