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
