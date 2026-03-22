
freq=(1:2:10400);
res=[363,1000,200];
thick=[1000,2000];
n=length(freq);

app_res=zeros(1,n);
phase=zeros(1,n);

for ii=1:n
app_res(ii)=modelMT(guess_res_opt,guess_thick_opt,freq(ii));
end

figure;
scatter(log10(freq), log10(app_res), 30, 'b', 'filled');
title('Apparent Resistivity from 3-layer model', 'FontSize', 12);
xlabel('log_{10}(Frequency) (Hz)');
ylabel('log_{10}(Apparent Resistivity) (\Omega\cdot m)');
set(gca, 'XDir', 'reverse');
grid on;
