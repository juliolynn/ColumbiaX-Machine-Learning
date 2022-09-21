x = [1:20]';
y = x.^1.5 -1;
X = [ones(size(x),1),x];
sd = std(x);
lambda = 5;
XTX =  X' * X;
SIGMA = inv(lambda * eye(size(XTX)) + sd^-2 *XTX)
mu = inv(lambda * sd^2 * eye(size(XTX)) + XTX)*X'*y

# incomplete


plot(x,y)
hold on;
plot(xn,yn)