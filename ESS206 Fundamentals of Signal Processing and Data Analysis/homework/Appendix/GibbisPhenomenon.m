%% Variable allocation
n = 200; % Fitting order
maxgibbis = zeros(1, n); % Maximum value of the fitting function in each iteration
a = -1.5; b = 1.5; % Define the interval
N = 3e3; delta = (b - a) / N; x_lin = transpose(a:delta:b); % Sample points
N = N + 1;

% Basis functions: cosine
createfun = @(ii, x) cos((ii - 1) * 2 * pi / (b - a) * (x - (b + a) / 2));

vec_fun = zeros(N, n);
for ii = 1:n
    vec_fun(:, ii) = createfun(ii, x_lin);
end

% Square wave target function
orifun = @(x) 0.5 + 0.5 * square(pi * (x + 0.5));
ori_fun = orifun(x_lin);

% Gram-Schmidt orthogonalization
normal_fun = vec_fun;
coeff = sum(delta .* normal_fun(:, 1).^2);
normal_fun(:, 1) = normal_fun(:, 1) / sqrt(coeff);

for ii = 2:n
    for jj = 1:ii-1
        coeff = sum(normal_fun(:, jj) .* normal_fun(:, ii)) * delta;
        normal_fun(:, ii) = normal_fun(:, ii) - coeff * normal_fun(:, jj);
    end
    coeff = sum(delta .* normal_fun(:, ii).^2);
    normal_fun(:, ii) = normal_fun(:, ii) / sqrt(coeff);
end

% Fitting process
coeff = zeros(n, 1);
fit_fun = zeros(N, 1);
figure; rows = ceil(n / 40); cols = 4;

for ii = 1:n
    coeff(ii) = sum(ori_fun .* normal_fun(:, ii)) * delta;
    fit_fun = fit_fun + coeff(ii) * normal_fun(:, ii);
    maxgibbis(ii) = max(fit_fun);
    if mod(ii, 10) == 0
        subplot(rows, cols, ii / 10);
        plot(x_lin, ori_fun, 'r', 'LineWidth', 2);
        hold on;
        plot(x_lin, fit_fun, 'b', 'LineWidth', 2);
        title(['Fitted Function of Order ' num2str(ii)]);
        grid on;
    end
end

% Gibbs phenomenon plot
figure;
plot(1:n, maxgibbis - 1, 'r'); hold on;
y_constant = 0.09;
plot([1, n], [y_constant, y_constant], 'b--');
title('The Gibbs Phenomenon');


