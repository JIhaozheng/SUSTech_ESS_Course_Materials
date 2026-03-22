tic
clear;clc;close all;
n = 12;             % Fitting order
a = 2*pi; b = 6*pi;  % Interval
orifun = @(x) sin(x);% Function to be fitted
createfun = @(ii,x) log(x+ii-1);
% Options for basis:
% Exponential: exp((ii-1)*x)
% Polynomial: x.^(ii-1)
% Sine: sin(ii*2*pi/(b-a)*(x-(a+b)/2))
% Cosine: cos((ii-1)*2*pi/(b-a)*(x-(a+b)/2))
% Logarithm: log(ii+x)

x = linspace(a,b,100);

fun = cell(1,n);       % Basis functions (not orthonormalized)
normalfun = cell(1,n); % Orthonormalized basis functions
 
% Generate the required bases
for ii = 1:n
    fun{ii} = @(x) createfun(ii,x);
    normalfun{ii} = fun{ii};
end

% Normalize the first vector
coeff = integral(@(x) normalfun{1}(x).^2, a, b);
normalfun{1} = @(x) normalfun{1}(x) / sqrt(coeff);
% Orthonormalize the n vectors
for ii = 2:n
    for jj = 1:ii-1
        coeff = integral(@(x) normalfun{ii}(x) .* normalfun{jj}(x), a, b);
        normalfun{ii} = @(x) normalfun{ii}(x) - coeff .* normalfun{jj}(x);
    end
    coeff = integral(@(x) normalfun{ii}(x).^2, a, b);
    normalfun{ii} = @(x) normalfun{ii}(x) / sqrt(coeff);
end

% Fitting process
coeff = zeros(1, n);
fitfun = @(x) 0;

% Create figure layout
figure;
rows = ceil(n/6);
cols = 6;
% Iterative fitting
for ii = 1:n
    coeff(ii) = integral(@(x) orifun(x) .* normalfun{ii}(x), a, b);
    fitfun = @(x) fitfun(x) + coeff(ii) .* normalfun{ii}(x);
    % Plot each fitting curve
    subplot(rows, cols, ii);
    plot(x, orifun(x), 'r', 'LineWidth', 2, 'DisplayName', 'Original function');
    hold on;
    plot(x, fitfun(x), 'b', 'LineWidth', 2, ...
         'DisplayName', ['Fitted function (Order ' num2str(ii) ')']);
    contribution = @(x) coeff(ii) .* normalfun{ii}(x);
    plot(x, contribution(x), '--b', 'DisplayName', ...
         ['Additional fitted function (Order ' num2str(ii) ')']);
    % Set title and legend for each subplot
    title(['Fitted Function of Order ' num2str(ii)]);
    grid on;
end

saveas(gcf, 'fit_result.png');

toc