data = load('data.txt');
[dataCnt, d] = size(data);
X = data(:, 1 : (d - 1));
y = data(:, d);
coeff = glmfit(X, [y ones(length(y), 1)], 'binomial', 'link', 'logit');
y1 = 1 / (1 + exp(-([X ones(length(y), 1)] * coeff)))