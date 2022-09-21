x = [1:20]';
y = x.^1.5 -1;
X = [ones(size(x),1),x];
theta = (inv(X'*X))*X'*y
xn = [1:.4:25]';
Xn = [ones(size(xn),1),xn];
yn = Xn * theta;

plot(x,y)
hold on;
plot(xn,yn)