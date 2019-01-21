data = load('data.txt');
xData = data(:, 1);
yData = data(:, 2);
coeff = polyfit(xData, yData, 1);
w = coeff(1, 1);
b = coeff(1, 2);
y = w * xData + b;
plot(xData, yData, '.', xData, y, 'r')