%% The High-Frequency Explosive Signal 
clc; clear; load kerl030916ud.txt
Fs = 50; dt = 1/Fs;
wp = 1.5*2/Fs; ws = 2.5*2/Fs;
Rp = 1; Rs = 30; Nn = 128;
[N,Wn] = buttord(wp, ws, Rp, Rs);
[b,a] = butter(N, Wn, 'High');
[H,f] = freqz(b,a,Nn,Fs);

figure;
subplot(2,2,3); plot(f, 20*log10(abs(H)));
xlabel('Frequency/Hz'); ylabel('Amplitude/dB');
grid on; set(gca, 'FontSize', 16);

subplot(2,2,4); plot(f, 180/pi*unwrap(angle(H)));
xlabel('Frequency/Hz'); ylabel('Phase/(degree)');
grid on; set(gca, 'FontSize', 16);

x = kerl030916ud';
y = filter(b, a, x);
t = (0:(length(x)-1)) * dt;

subplot(2,2,1); plot(t, x); title('Input Signals');
set(gca, 'FontSize', 16);

subplot(2,2,2); plot(t, y); title('Output Signals');
xlabel('Time/s'); set(gca, 'FontSize', 16);

%% FIR
clc; clear;
wp = [0.032 0.2]; N = 320; dt = 0.02;
b = fir1(N, wp, hanning(N+1));
[H,f] = freqz(b,1,512,1/dt);

figure;
subplot(2,2,3); plot(f, 20*log10(abs(H)));
xlabel('Frequency/Hz'); ylabel('Amplitude/dB');
grid on; set(gca, 'FontSize', 16);

subplot(2,2,4); plot(f, 180/pi*unwrap(angle(H)));
xlabel('Frequency/Hz'); ylabel('Phase/(degree)');
grid on; set(gca, 'FontSize', 16);

load ChangChun.txt
y = filtfilt(b, 1, ChangChun);
t = 0:length(ChangChun)-1;

subplot(2,2,1); plot(t, ChangChun);
title('Input Signals'); set(gca, 'FontSize', 16);

subplot(2,2,2); plot(t, y);
title('Output Signals (FIR)'); xlabel('Time/s');
set(gca, 'FontSize', 16);
%% IIR
clc; clear; load ChangChun.txt
Fs = 50; dt = 1/Fs;
wp = [0.8 5] * 2 / Fs;
ws = [0.5 7] * 2 / Fs;
Rp = 3; Rs = 30;

[N, Wn] = buttord(wp, ws, Rp, Rs);
[b, a] = butter(N, Wn, 'bandpass');
[H, f] = freqz(b, a, 512, Fs);

figure;
subplot(2,2,3); plot(f, 20*log10(abs(H)));
xlabel('Frequency/Hz'); ylabel('Amplitude/dB');
title('IIR Bandpass Magnitude');
grid on; set(gca, 'FontSize', 16);

subplot(2,2,4); plot(f, 180/pi*unwrap(angle(H)));
xlabel('Frequency/Hz'); ylabel('Phase/(degree)');
title('IIR Bandpass Phase');
grid on; set(gca, 'FontSize', 16);

t = (0:(length(ChangChun)-1)) * dt;
subplot(2,2,1); plot(t, ChangChun);
title('Input Signals'); set(gca, 'FontSize', 16);

y = filtfilt(b, a, ChangChun);
subplot(2,2,2); plot(t, y);
title('Output Signals (IIR)'); xlabel('Time/s');
set(gca, 'FontSize', 16);
