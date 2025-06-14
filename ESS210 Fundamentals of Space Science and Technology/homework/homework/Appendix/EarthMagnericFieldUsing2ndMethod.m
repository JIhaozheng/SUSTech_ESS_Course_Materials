clc;clear;close all;
xm=15; ym=10; 
x=linspace(-xm,2*xm,500);
y=linspace(-2*ym,2*ym,500);
[X,Y]=meshgrid(x,y);

% Avoid division by zero at the origin
X(X==0)=1e-6;
Y(Y==0)=1e-6;

%% Figure 1: Magnetic dipole field
figure;
xlim([-xm xm]); ylim([-ym ym]);
hold on;

% Plot field lines of a single dipole centered at origin
for jj=75:1:105
    theta=jj*pi/180;
    x0=cos(theta);
    y0=sin(theta);
    U0=x0/(x0^2+y0^2)^1.5;  % reference potential for field line
    r=sqrt(X.^2+Y.^2);
    U=X./r.^3;  % magnetic field line equation for dipole
    contour(X,Y,U,[U0 U0],'LineColor','k');
end

% Draw the central dipole as a circle
rectangle('Position', [0-1, 0-1, 2, 2], 'Curvature', [1, 1], ...
    'EdgeColor', 'b', 'FaceColor', 'w');

title('Magnetic Dipole Field Lines of Earth','FontSize',20);
axis equal;
axis([-xm xm -ym ym]);

%% Figure 2: Two symmetric dipoles (mirror image)
figure;
xlim([0 2*xm]); ylim([-2*ym 2*ym]);
hold on;

% Calculate combined potential from dipoles at x = ±10
r1=sqrt((X+10).^2+Y.^2);
r2=sqrt((X-10).^2+Y.^2);
U=(X+10)./r1.^3+(X-10)./r2.^3;

% Contour levels for visualizing field lines
neg_levels=linspace(-1e-2,-1e-6,15);
pos_levels=linspace(1e-6,5e-3,15);
contour(X,Y,U,[neg_levels pos_levels],'LineWidth',0.7,'LineColor','k');

% Draw the Earth dipole as a circle
rectangle('Position', [10-1, 0-1, 2, 2], 'Curvature', [1, 1], ...
          'EdgeColor', 'b', 'FaceColor', 'w');

title('Magnetic Field Lines of Symmetric Mirrored Dipoles', 'FontSize', 20);
axis equal;
axis([0 2*xm -2*ym 2*ym]);
