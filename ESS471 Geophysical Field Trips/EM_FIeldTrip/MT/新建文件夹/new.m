 %% Read the original data
Zxy = readtable('2025MTdataStudents.xlsx', 'Sheet', 1, 'Range', 'B42:D81');
Zxy = table2array(Zxy);
Zyx = readtable('2025MTdataStudents.xlsx', 'Sheet', 1, 'Range', 'B82:D121');
Zyx = table2array(Zyx);
mu=4*pi*1E-7;

% Calculate apparent resistivity
for ii=1:length(Zxy)
    Zxy(ii,4)=Zxy(ii,2)^2/(2*pi*Zxy(ii,1)*mu);
    Zyx(ii,4)=Zyx(ii,2)^2/(2*pi*Zyx(ii,1)*mu);
end



%% Plot in log scale
figure;

% Apparent resistivity scatter plot
subplot(1,2,1);
scatter(log10(Zxy(:,1)), log10(Zxy(:,4)), 30, 'r', 'filled');
hold on;
scatter(log10(Zyx(:,1)), log10(Zyx(:,4)), 30, 'b', 'filled');
title('Apparent Resistivity from Zxy and Zyx', 'FontSize', 12);
legend('Zxy', 'Zyx');
xlabel('log_{10}(Frequency) (Hz)');
ylabel('log_{10}(Apparent Resistivity) (\Omega\cdot m)');
set(gca, 'XDir', 'reverse');
grid on;

% Phase scatter plot
subplot(1,2,2);
scatter(log10(Zxy(:,1)), Zxy(:,3), 30, 'r', 'filled');
hold on;
scatter(log10(Zyx(:,1)), Zyx(:,3), 30, 'b', 'filled');
title('Phase from Zxy and Zyx', 'FontSize', 12);
legend('Zxy', 'Zyx');
xlabel('log_{10}(Frequency) (Hz)');
ylabel('Phase (Degree)');
set(gca, 'XDir', 'reverse');
grid on;

%% Quality control using moving variance
newm = [log10(Zxy(:,4)), Zxy(:,3), log10(Zyx(:,4)), Zyx(:,3)];
labels = {'Zxy-\rho', 'Zxy-phase', 'Zyx-\rho', 'Zyx-phase'};
windows = 5;
cal_var = zeros(length(newm) - windows + 1, 4);
index=[];

% Calculate variance using a sliding window
threshold = zeros(1,4);
for ii = 1:4
    temp_var = movvar(newm(:,ii), windows);
    threshold(ii) = mean(temp_var) + 2*std(temp_var);  % Dynamic threshold
end

% Record anomalous data
figure;
for ii = 1:4
    subplot(2,2,ii);
    hold on;
    for jj = 1:length(newm) - windows + 1
        cal_var(jj,ii) = var(newm(jj:jj+windows-1, ii));
        if cal_var(jj,ii) >= threshold(ii)
            fprintf('Anomaly detected in %s at window starting index %d (variance = %.3f)\n', ...
                labels{ii}, jj, cal_var(jj,ii));
            index(end+1)=jj;
        end
    end
    plot(cal_var(:,ii), 'LineWidth', 1.2);
    yline(threshold(ii), 'r--', 'Threshold');
    title(['Moving Var of ', labels{ii}]);
    xlabel('Window Start Index');
    ylabel('Variance');
    grid on;
end

% Remove abnormal data
index=unique(index','rows');
for ii=1:length(index)
    Zxy(ii+windows-1,:)=[];
    Zyx(ii+windows-1,:)=[];
end

%% Plot in log scale after quality control
figure;

% Apparent resistivity scatter plot
subplot(1,2,1);
scatter(log10(Zxy(:,1)), log10(Zxy(:,4)), 30, 'r', 'filled');
hold on;
scatter(log10(Zyx(:,1)), log10(Zyx(:,4)), 30, 'b', 'filled');
title('Apparent Resistivity from Zxy and Zyx after QC', 'FontSize', 12);
legend('Zxy', 'Zyx');
xlabel('log_{10}(Frequency) (Hz)');
ylabel('log_{10}(Apparent Resistivity) (\Omega\cdot m)');
set(gca, 'XDir', 'reverse');
grid on;

% Phase scatter plot
subplot(1,2,2);
scatter(log10(Zxy(:,1)), Zxy(:,3), 30, 'r', 'filled');
hold on;
scatter(log10(Zyx(:,1)), Zyx(:,3), 30, 'b', 'filled');
title('Phase from Zxy and Zyx after QC', 'FontSize', 12);
legend('Zxy', 'Zyx');
xlabel('log_{10}(Frequency) (Hz)');
ylabel('Phase (Degree)');
set(gca, 'XDir', 'reverse');
grid on;

%% Draw apparent resistivity profile
freq=Zxy(:,1);
ave_res=(Zxy(end,4)+Zyx(end,4))./2;
app_res=(Zxy(:,4)+Zyx(:,4))./2;
skin_depth=503*sqrt(ave_res/freq(end));
depth=503*sqrt(ave_res./freq);
figure;
plot(log10(app_res),depth, 'LineWidth',1.2);
title('Apparent resistivity profile');
xlabel('log_{10}(Apparent Resistivity) (\Omega\cdot m)');
ylabel('Depth (m)');
set(gca,'YDir','reverse');



function [apparentResistivity] = modelMT(resistivities, thicknesses,frequency)
mu = 4*pi*1E-7; %Magnetic Permeability (H/m)  
w = 2 * pi * frequency;  %Angular Frequency (Radians);
n=length(resistivities); %Number of Layers
impedances = zeros(n,1);
Zn = sqrt(sqrt(-1)*w*mu*resistivities(n));
impedances(n) = Zn;

for j = n-1:-1:1
    resistivity = resistivities(j);
    thickness = thicknesses(j);
    dj = sqrt(sqrt(-1)* (w * mu * (1.0/resistivity)));
    wj = dj * resistivity;
    ej = exp(-2*thickness*dj);
    belowImpedance = impedances(j + 1);
    rj = (wj - belowImpedance)/(wj + belowImpedance);
    re = rj*ej;
    Zj = wj * ((1 - re)/(1 + re));
    impedances(j) = Zj;
end

Z = impedances(1);
absZ = abs(Z);
apparentResistivity = (absZ * absZ)/(mu * w);
end

%% Invert Zxy data using 3-layereded model
app_res = (Zxy(:,4));
freq = Zxy(:,1);

% Set initial parameters
guess_res = [200, 2000, 2000];   % Initial resistivity values
guess_thick = [300,1000];       % Thicknesses of upper layered
x0 = [guess_res, guess_thick];

% Set lower bounds
lb = 100*ones(size(x0));

% Define the objective function to calculate error
objectiveFunc = @(x) computeError(x, freq, app_res);

% Use fmincon for constrained optimization
options = optimoptions('fmincon', 'Display', 'iter', 'Algorithm', 'interior-point');
[x_opt, ~] = fmincon(objectiveFunc, x0, [], [], [], [], lb, [], [], options);

% Get optimized model parameters
guess_res_opt = x_opt(1:3);
guess_thick_opt = x_opt(4:5);

% Compute apparent resistivity based on the model
compute_res_opt = arrayfun(@(f) modelMT(guess_res_opt, guess_thick_opt, f), freq);
error_opt = sum((compute_res_opt - app_res).^2);
disp(['error after fitting: ', num2str(error_opt)]);

figure;
subplot(1,2,1);
scatter(log10(freq), log10(app_res), 30, 'r', 'filled');
hold on;
plot(log10(freq), log10(compute_res_opt),'b','LineWidth',1.2);
title('Apparent resistivity from Zxy data and 3-layered model', 'FontSize', 20);
legend('original data', 'simulated data','FontSize',20);
xlabel('log_{10}(Frequency) (Hz)','FontSize',20);
ylabel('log_{10}(Apparent Resistivity) (\Omega\cdot m)','FontSize',20);
set(gca, 'XDir', 'reverse');
grid on;

subplot(1,2,2);
plot3LayerModel(guess_res_opt, guess_thick_opt);

%% Invert Zyx data using 3-layered model
app_res = (Zyx(:,4));
freq = Zxy(:,1);

% Set initial parameters
guess_res = [200, 2000, 2000];   % Initial resistivity values
guess_thick = [10,1000];       % Thicknesses of upper layered
x0 = [guess_res, guess_thick];

% Set lower bounds
lb = 100*ones(size(x0));

% Define the objective function to calculate error
objectiveFunc = @(x) computeError(x, freq, app_res);

% Use fmincon for constrained optimization
options = optimoptions('fmincon', 'Display', 'iter', 'Algorithm', 'interior-point');
[x_opt, ~] = fmincon(objectiveFunc, x0, [], [], [], [], lb, [], [], options);

% Get optimized model parameters
guess_res_opt = x_opt(1:3);
guess_thick_opt = x_opt(4:5);

% Compute apparent resistivity based on the model
compute_res_opt = arrayfun(@(f) modelMT(guess_res_opt, guess_thick_opt, f), freq);
error_opt = sum((compute_res_opt - app_res).^2);
disp(['error after fitting: ', num2str(error_opt)]);

figure;
subplot(1,2,1);
scatter(log10(freq), log10(app_res), 30, 'r', 'filled');
hold on;
plot(log10(freq), log10(compute_res_opt),'b','LineWidth',1.2);
title('Apparent resistivity from Zyx data and 3-layered model', 'FontSize', 20);
legend('original data', 'simulated data','FontSize',20);
xlabel('log_{10}(Frequency) (Hz)','FontSize',20);
ylabel('log_{10}(Apparent Resistivity) (\Omega\cdot m)','FontSize',20);
set(gca, 'XDir', 'reverse');
grid on;

subplot(1,2,2);
plot3LayerModel(guess_res_opt, guess_thick_opt);

function plot3LayerModel(resistivities, thicknesses)
    % Get current axes
    ax = gca;
    hold(ax, 'on');
    grid(ax, 'on');
    box(ax, 'on');
    
    % Set Y-axis direction (depth increases downward)
    set(ax, 'YDir', 'reverse');
    
    % Define layer heights (second layer thicker than the first, third extends to bottom)
    layer_heights = [0.15, 0.25, 0.6];  % Layer 1: 0.15, Layer 2: 0.25, Layer 3: 0.6
    
    % Calculate starting positions of each layer (to ensure continuity)
    layer_starts = [0, cumsum(layer_heights(1:2))];
    
    % Define colors for each layer
    colors = [0.9 0.95 1;      % Layer 1 (light blue)
              0.8 0.9 1;       % Layer 2 (medium blue)
              0.7 0.85 1];     % Layer 3 (dark blue)
    
    % Draw each layer
    for i = 1:3
        % Calculate position and height of current layer
        rect_y = layer_starts(i);
        rect_height = layer_heights(i);
        
        % Draw rectangle representing the layer
        rectangle(ax, 'Position', [0.1, rect_y, 0.8, rect_height], ...
                  'FaceColor', colors(i, :), ...
                  'EdgeColor', 'k', ...
                  'LineWidth', 1.5);
        
        % Add resistivity label
        text(ax, 0.5, rect_y + rect_height/2, ...
             sprintf('\\rho_{%d} = %.1f \\Omega\\cdotm', i, resistivities(i)), ...
             'FontSize', 12, 'FontWeight', 'bold', ...
             'HorizontalAlignment', 'center');
        
        % Add thickness label (only for the first two layers)
        if i < 3
            % Compute label position (just above the bottom of current layer)
            h_label_y = rect_y + rect_height - 0.02;
            
            % Draw thickness indicator line (only for the current layer)
            line(ax, [0.05, 0.05], [rect_y, rect_y + rect_height], ...
                 'Color', 'r', 'LineWidth', 2);
            
            % Add thickness text (left of the indicator line)
            text(ax, 0.02, rect_y + rect_height/2, ...
                 sprintf('h_{%d} = %.1f m', i, thicknesses(i)), ...
                 'FontSize', 10, 'Color', 'r', ...
                 'Rotation', 90, ...
                 'VerticalAlignment', 'middle', ...
                 'HorizontalAlignment', 'center');
        end

        % Add dashed horizontal boundary line (for layers below the first)
        if i > 1
            line(ax, [0, 1], [rect_y, rect_y], ...
                 'Color', 'k', 'LineWidth', 2, 'LineStyle', '--');
        end
    end
    
    % Draw top surface line
    line(ax, [0, 1], [0, 0], 'Color', 'k', 'LineWidth', 3);
    
    % Label the half-space at the top
    text(ax, 0.5, 0.95, 'half-space', ...
         'FontSize', 11, 'Color', [0.5 0.5 0.5], ...
         'HorizontalAlignment', 'center');
    
    % Set plot limits and formatting
    xlim(ax, [0, 1]);
    ylim(ax, [0, 1]);
    set(ax, 'XTick', []);
    set(ax, 'YTick', []);
    title(ax, '3-Layered Model','FontSize',20);
    ylabel(ax, 'Depth','FontSize',20);
    hold(ax, 'off');
end

function error = computeError(x, freq, app_res)
    res_params = x(1:3);
    thick_params = x(4:5);
    if any(res_params <= 0) || any(thick_params <= 0)
        error = inf; 
        return;
    end
    compute_res = zeros(size(freq));
    for i = 1:length(freq)
        compute_res(i) = modelMT(res_params, thick_params, freq(i));
    end
    error = sum((compute_res - app_res).^2);
end