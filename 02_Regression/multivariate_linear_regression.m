clear all
close all

iterations = 3000000;
learning_rate = 0.01;
weights = [-1000, 1000, 1000];

% Import data, 3 columns: RS, OBP, SLG
A = readtable('baseball.csv');
X1 = A.OBP;
X2 = A.SLG;
Y = A.RS;

X = [X1, X2]; % 입력 데이터

% 비용 함수 정의
function mse = cost_function(w, X, Y)
    w0 = w(1);
    w1 = w(2);
    w2 = w(3);
    predictions = w0 + X(:, 1) * w1 + X(:, 2) * w2;
    errors = predictions - Y;
    mse = mean(errors.^2);
end

% 그래디언트 계산 함수 정의
function gradients = compute_gradient(w, X, Y)
    w0 = w(1);
    w1 = w(2);
    w2 = w(3);
    predictions = w0 + X(:, 1) * w1 + X(:, 2) * w2;
    errors = predictions - Y;

    grad_w0 = mean(2 * errors);
    grad_w1 = mean(2 * errors .* X(:, 1));
    grad_w2 = mean(2 * errors .* X(:, 2));

    gradients = [grad_w0, grad_w1, grad_w2];
end

% 경사 하강법 반복 수행
for i = 1:iterations
    gradients = compute_gradient(weights, X, Y);
    weights = weights - learning_rate * gradients;

    if mod(i, 500000) == 0
        cost = cost_function(weights, X, Y);
        fprintf('Iteration %d: Cost %.6f\n', i, cost);
    end
end

% 예측값 계산
predictions = weights(1) + X(:, 1) * weights(2) + X(:, 2) * weights(3);

Y_mean = mean(Y);
SS_res = sum((Y - predictions).^2);
SS_tot = sum((Y - Y_mean).^2);
R_squared = 1 - (SS_res / SS_tot);

fprintf('결정계수 R²: %.6f\n', R_squared);
fprintf('OBP 기울기: %.6f\n', weights(2));
fprintf('SLG 기울기: %.6f\n', weights(3));
fprintf('절편: %.6f\n', weights(1));