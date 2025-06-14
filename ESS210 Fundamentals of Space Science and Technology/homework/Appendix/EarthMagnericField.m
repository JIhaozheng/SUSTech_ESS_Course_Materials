clc; clear;
xm=15; ym=10; x=linspace(-xm,xm,500); y=linspace(-ym,ym,500);
[X,Y]=meshgrid(x,y); X(X==0)=1e-6; Y(Y==0)=1e-6;

N=5000; xt=zeros(1,N); yt=zeros(1,N); l=0.01;

figure(1); xlim([-xm xm]); ylim([-ym ym]);
rectangle('Position', [0-1, 0-1, 2, 2], 'Curvature', [1, 1], ...
          'EdgeColor', 'b', 'FaceColor', 'none'); % First dipole (blue circle)
hold on;

for jj=75:1:90
    theta=jj*pi/180; xt(1)=cos(theta); yt(1)=sin(theta);
    for ii=1:N-1
        k=2*yt(ii)./3./xt(ii)-xt(ii)/3./yt(ii);
        xt(ii+1)=xt(ii)+l*(1+k.^2)^(-0.5);
        yt(ii+1)=yt(ii)+l*k*(1+k.^2)^(-0.5);
    end
    plot(xt,yt,'k'); plot(-xt,yt,'k');
    plot(xt,-yt,'k'); plot(-xt,-yt,'k');
end
title('Magnetic Dipole Field Lines of Earth','FontSize',20);
axis equal;

%%
figure(2); xlim([-2*xm 2*xm]); ylim([-ym ym]);
rectangle('Position', [10-1, 0-1, 2, 2], 'Curvature', [1, 1], ...
          'EdgeColor', 'b', 'FaceColor', 'none'); % First dipole (blue circle)
hold on;
rectangle('Position', [-10-1, 0-1, 2, 2], 'Curvature', [1, 1], ...
          'EdgeColor', 'b', 'FaceColor', 'none'); % First dipole (blue circle)
plot([0; 0], [-2*ym; 2*ym],'r'); 
for jj=70:1:90
    theta=jj*pi/180; xt(1)=10+cos(theta); yt(1)=sin(theta);
    for ii=1:N-1
        r1=((xt(ii)-10)^2+yt(ii)^2)^0.5;
        r2=((xt(ii)+10)^2+yt(ii)^2)^0.5;
        k=((2*yt(ii)^2-(xt(ii)-10)^2)/r1^5+(2*yt(ii)^2-(xt(ii)+10)^2)/r2^5) / (3*yt(ii)*(xt(ii)-10)/r1^5+3*yt(ii)*(xt(ii)+10)/r2^5);
        xt(ii+1)=xt(ii)+l*(1+k^2)^(-0.5);
        yt(ii+1)=yt(ii)+l*k*(1+k^2)^(-0.5);
    end
    plot(xt,yt,'k'); plot(xt,-yt,'k');
end

for jj=90:1:110
    theta=jj*pi/180; xt(1)=10+cos(theta); yt(1)=sin(theta);
    for ii=1:N-1
        r1=((xt(ii)-10)^2+yt(ii)^2)^0.5;
        r2=((xt(ii)+10)^2+yt(ii)^2)^0.5;
        k=((2*yt(ii)^2-(xt(ii)-10)^2)/r1^5+(2*yt(ii)^2-(xt(ii)+10)^2)/r2^5) / (3*yt(ii)*(xt(ii)-10)/r1^5+3*yt(ii)*(xt(ii)+10)/r2^5);
        xt(ii+1)=xt(ii)-l*(1+k^2)^(-0.5);
        yt(ii+1)=yt(ii)-l*k*(1+k^2)^(-0.5);
    end
    plot(xt,yt,'k'); plot(xt,-yt,'k');
end
title('Magnetic Field Lines of Symmetric Mirrored Dipoles', 'FontSize', 20);
axis equal;