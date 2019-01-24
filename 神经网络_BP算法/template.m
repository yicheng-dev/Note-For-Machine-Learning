x = xlsread('data.xlsx', 'B2:I18');
y = xlsread('data.xlsx', 'J2:J18');
[sample_num, attr_num] = size(x);
dest_num = size(y(1, :));
x = x';
y = y';

%归一化
[x_normal, inputStr] = mapminmax(x);
[y_normal, outputStr] = mapminmax(y);

%建议BP神经网络
net = newff(x_normal, y_normal, [attr_num, attr_num + 1, dest_num], {'purelin', 'logsig', 'logsig'});

%最大训练次数
net.trainParam.epochs = 5000;
%学习率
net.trainParam.lr = 0.1;
%目标误差
net.trainParam.goal = 1 * 10^(-4);

net = train(net, x_normal, y_normal);