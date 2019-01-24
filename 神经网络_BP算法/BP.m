%目前只针对西瓜书数据集3.0的模型，并将文本属性值根据语义调整为数据。在此基础上尽可能保证通用性。
%参考了https://blog.csdn.net/icefire_tyh/article/details/52106069提供的实现
clear

%输入数据集
x = xlsread('data.xlsx', 'B2:I18');
y_sample = xlsread('data.xlsx', 'J2:J18');
[sample_num, attr_num] = size(x);

%输入层的结点个数
input_num = attr_num;

%输出层的结点个数
output_num = 1;

%隐含层的结点个数
hidden_num = attr_num + 1;

%BP算法的各个参数和变量（命名依照书中的符号）
v = rand(input_num, hidden_num);
w = rand(hidden_num, output_num);
gamma = rand(hidden_num);
theta = rand(output_num);
eta = 0.5;
b = zeros(hidden_num);
y = zeros(sample_num, output_num);

%判断终止条件
last_E = 0;
count = 0;
threshold = 0.0001;
max_count = 1000;

while (1)
    E = 0;
    for s = 1 : sample_num
        %隐层输出
        for h = 1 : hidden_num
            tmp = 0;
            for i = 1 : input_num
                tmp = tmp + v(i, h) * x(s, i);
            end
            b(h) = 1 / (1 + exp(-(tmp - gamma(h))));
        end
        %输出层输出
        for j = 1 : output_num
            tmp = 0;
            for h = 1 : hidden_num
                tmp = tmp + w(h, j) * b(h);
            end
            y(s, j) = 1 / (1 + exp(-(tmp - theta(j))));
        end
        %计算当前迭代累计误差
        for j = 1 : output_num
            E = E + ((y_sample(s) - y(s, j))^2) / 2;
        end
        %计算gj
        gj = zeros(output_num);
        for j = 1 : output_num
            gj(j) = y(s, j) * (1 - y(s, j)) * (y_sample(s) - y(s, j));
        end
        %计算eh
        eh = zeros(hidden_num);
        for h = 1 : hidden_num
            tmp = 0;
            for j = 1 : output_num
                tmp = tmp + w(h, j) * gj(j);
            end
            eh(h) = tmp * b(h) * (1 - b(h));
        end
        %更新w, theta
        for j = 1 : output_num
            theta(j) = theta(j) + (-eta) * gj(j);
            for h = 1 : hidden_num
                w(h, j) = w(h, j) + eta * gj(j) * b(h);
            end
        end
        %更新v, gamma
        for h = 1 : hidden_num
            gamma(h) = gamma(h) + (-eta) * eh(h);
            for i = 1 : input_num
                v(i, h) = v(i, h) + eta * eh(h) * x(s, i);
            end
        end
    end
    %迭代终止判断
    if(abs(last_E - E) < threshold)
        count = count + 1;
        if(count >= max_count)
            break;
        end
    else
        last_E = E;
        count = 0;
    end
end
y