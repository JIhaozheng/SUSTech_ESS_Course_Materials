x = linspace(0.1, 5, 1000);
y = linspace(0.1, 5, 1000); 


[X, Y] = meshgrid(x, y);
Z = Y.^2 - log(Y.^2) - 4*log(X) - 4./X;

figure;
contourf(X, Y, Z, 100); 
title('equipotential-surface');
xlabel('r/rc');
ylabel('v/vc');
colorbar; 


hold on;
contour(X, Y, Z, [-3, -3], 'k', 'LineWidth', 2); 
legend('equipotential-surface', 'C=-3');
grid on;

%%
vc = 128.433; 
rc = 4.014e6; 
omega_sun = 2.85e-6; 


planets = {'Mercury', 'Earth', 'Mars', 'Jupiter', 'Neptune'};
distances = [57.9e6, 149.6e6, 227.9e6, 778.5e6, 4500e6];


solar_wind_velocity = zeros(size(distances));
angles = zeros(size(distances));


for i = 1:length(distances)
    r = distances(i);
    

    v = 2 * vc * sqrt(log(r / rc));
    solar_wind_velocity(i) = v;
    

    theta = atan(v / (r * omega_sun)) * (180 / pi); % Convert to degrees
    angles(i) = theta;
end

disp('Planet  | Distance (km)   | Solar Wind Velocity (km/s) | Angle (degrees)');
disp('------------------------------------------------------------');
for i = 1:length(planets)
    fprintf('%-8s | %.1f            | %.1f                     | %.2f\n', ...
        planets{i}, distances(i), solar_wind_velocity(i), angles(i));
end
