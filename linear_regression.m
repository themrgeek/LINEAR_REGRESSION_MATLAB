% Load the data from Excel file
data = xlsread('./A1_Data_Set.xlsx');
x = data(:, 1); % Assuming the independent variable is in the first column
y = data(:, 2); % Assuming the dependent variable is in the second column

% Number of data points
N = length(y);

% Closed-form solution for linear regression
sum_x = sum(x);
sum_y = sum(y);
sum_x_squared = sum(x .^ 2);
sum_xy = sum(x .* y);

w1_closed = (N * sum_xy - sum_x * sum_y) / (N * sum_x_squared - sum_x^2);
w0_closed = (sum_y - w1_closed * sum_x) / N;

fprintf('Closed-form solution: w0 = %.4f, w1 = %.4f\n', w0_closed, w1_closed);

% Plotting the cost function Jw vs w1 with w0 = 0
w1_range = 0:0.1:20; 
Jw = zeros(size(w1_range));

for i = 1:length(w1_range)
    w1 = w1_range(i);
    h = w1 * x;
    Jw(i) = (1/(2*N)) * sum((h - y).^2);
end

figure;
plot(w1_range, Jw);
xlabel('w1');
ylabel('Jw');
title('Cost function Jw vs w1 with w0 = 0');

% Gradient Descent
alpha = 0.01; % Initial learning rate
num_iters = 1000;
theta = zeros(2, 1); % Initial parameters [w0; w1]
X = [ones(N, 1) x]; % Design matrix
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    h = X * theta;
    errors = h - y;
    theta = theta - (alpha/N) * (X' * errors);
    J_history(iter) = (1/(2*N)) * sum(errors.^2);
end

w0_gradient = theta(1);
w1_gradient = theta(2);

fprintf('Gradient Descent solution: w0 = %.4f, w1 = %.4f\n', w0_gradient, w1_gradient);

% Plotting Jw vs number of iterations
figure;
plot(1:num_iters, J_history);
xlabel('Number of iterations');
ylabel('Jw');
title('Cost function Jw vs number of iterations');

% Increasing learning rate until fluctuation
alphas = [0.01, 0.05, 0.1, 0.5, 1];
figure;
hold on;
for alpha = alphas
    theta = zeros(2, 1); % Reset parameters
    J_history = zeros(num_iters, 1);
    for iter = 1:num_iters
        h = X * theta;
        errors = h - y;
        theta = theta - (alpha/N) * (X' * errors);
        J_history(iter) = (1/(2*N)) * sum(errors.^2);
    end
    plot(1:num_iters, J_history, 'DisplayName', ['\alpha = ' num2str(alpha)]);
end
hold off;
legend show;
xlabel('Number of iterations');
ylabel('Jw');
title('Cost function Jw vs number of iterations for different learning rates');

% Save the figures and results
saveas(figure(1), 'CostFunction_vs_w1.png');
saveas(figure(2), 'CostFunction_vs_Iterations.png');
saveas(figure(3), 'CostFunction_vs_Iterations_LearningRates.png');
