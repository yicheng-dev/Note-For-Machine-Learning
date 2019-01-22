load fisheriris
data = load('data2.txt');
[dataCnt, d] = size(data);
X = data(:, 1 : (d - 1));
y = data(:, d);
ctree = fitctree(X, y);
view(ctree)