clc; clear; close all;

%% 参数设置（三层模型）
rho = [10, 1000, 50];    % 每层电阻率 (Ω·m)
thk = [50, 500];         % 前两层厚度 (m)，第3层为无限厚
mu = 4 * pi * 1e-7;

% 频率设置
f = logspace(-2, 2, 200);   % 0.01 Hz 到 100 Hz
omega = 2 * pi * f;

Z_app = zeros(size(f));
phase = zeros(size(f));

for i = 1:length(f)
    w = omega(i);

    % 计算每层的传播常数和阻抗
    k = sqrt(1i * w * mu ./ rho);
    Z = sqrt(1i * w * mu * rho(3));  % 最底层阻抗

    % 从底层往上传递阻抗
    for j = 2:-1:1
        Zj = sqrt(1i * w * mu * rho(j));
        dj = thk(j);
        rj = Zj;

        Z = rj * (Z + rj * tanh(k(j) * dj)) / (rj + Z * tanh(k(j) * dj));
    end

    % 视电阻率与相位
    Z_app(i) = abs(Z)^2 / (mu * w);
    phase(i) = atan2(imag(Z), real(Z)) * 180 / pi;
end

%% 绘图
figure;

subplot(2,1,1);
plot(log10(f), log10(Z_app), 'b', 'LineWidth', 1.5);
xlabel('Frequency (Hz)');
ylabel('Apparent Resistivity (\Omega·m)');
title('3-Layer MT Forward: Apparent Resistivity');
set(gca, 'XDir', 'reverse');
grid on;

subplot(2,1,2);
plot(log10(f), phase, 'r', 'LineWidth', 1.5);
xlabel('Frequency (Hz)');
ylabel('Phase (°)');
title('3-Layer MT Forward: Phase');
set(gca, 'XDir', 'reverse');
grid on;
