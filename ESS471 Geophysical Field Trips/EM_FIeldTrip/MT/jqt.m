clc;clear;close all;
% 设置路径和常数
filename = '2025MTdataStudents.xlsx';
mu0 = 4 * pi * 1e-7;
% 保留原始列名
opts = detectImportOptions(filename, 'Sheet', 'Taul1', 'VariableNamingRule', 'preserve');
data = readtable(filename, opts);
data.Properties.VariableNames = strtrim(data.Properties.VariableNames);
components = {'Zxx', 'Zxy', 'Zyx', 'Zyy'};
colors = lines(4);
% ========== 合并图：视电阻率 ==========
figure; hold on;
for i = 1:1
comp = components{i};
idx = strcmp(data.Comp, comp);
freq = data.Freq(idx);
amp = data.Amp(idx);
omega = 2 * pi * freq;
rho_a = (1 ./ (omega * mu0)) .* (amp .^ 2);
plot(log10(freq), log10(rho_a), 'o-', ...
'Color', colors(i,:), ...
'MarkerFaceColor', colors(i,:), ...
'MarkerSize', 4, ...
'LineWidth', 1.5, ...
'DisplayName', comp);
end
set(gca, 'XDir', 'reverse');
xlabel('log_{10}(Frequency / Hz)');
ylabel('log_{10}(Apparent Resistivity / \Omega·m)');
title('Combined: Apparent Resistivity');
legend show;
grid on;
% ========== 合并图：相位角 ==========
figure; hold on;
for i = 1:1
comp = components{i};
idx = strcmp(data.Comp, comp);
freq = data.Freq(idx);
phase = data.('Phase(Degree)')(idx);
plot(log10(freq), phase, 'o-', ...
'Color', colors(i,:), ...
'MarkerFaceColor', colors(i,:), ...
'MarkerSize', 4, ...
'LineWidth', 1.5, ...
'DisplayName', comp);
end
set(gca, 'XDir', 'reverse');
xlabel('log_{10}(Frequency / Hz)');
ylabel('Phase (Degree)');
title('Combined: Phase Angle');
legend show;
grid on;
% ========== 单独图：视电阻率 ==========
for i = 1:1
comp = components{i};
idx = strcmp(data.Comp, comp);
freq = data.Freq(idx);
amp = data.Amp(idx);
omega = 2 * pi * freq;
rho_a = (1 ./ (omega * mu0)) .* (amp .^ 2);
figure;
plot(log10(freq), log10(rho_a), 'o-', ...
'Color', colors(i,:), ...
'MarkerFaceColor', colors(i,:), ...
'MarkerSize', 4, ...
'LineWidth', 1.5);
set(gca, 'XDir', 'reverse');
xlabel('log_{10}(Frequency / Hz)');
ylabel('log_{10}(Apparent Resistivity / \Omega·m)');
title(['Apparent Resistivity - ' comp]);
grid on;
end