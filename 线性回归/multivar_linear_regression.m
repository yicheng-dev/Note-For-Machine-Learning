data = load('multivar_data.txt');
[dataCnt, d] = size(data);
y = data(:, d);
X = [data(:, 1 : (d - 1)), ones(length(y), 1)];
w = regress(y, X)