clear;clc;close all;
n=50;%Fitting order
a=-4;b=4;%Interval
N=10^4;delta=(b-a)/N;x_lin=transpose(a:delta:b);
N=N+1;

%Options for basis
createfun=@(ii,x) sin((1/4*(-1)^(ii)-1/4+ii/2)*2*pi/(b-a)*(x-(b+a)/2)+pi/4+pi/4*(-1)^(ii-1));

%Exponential(Laplace Transform):   exp((ii-1)*x)
%Polynomial(Z transform):          x.^(ii-1)
% Logarithm:                       log(ii+x)
%Sine:                             sin(ii*2*pi/(b-a)*(x-(b+a)/2))
%Cosine:                           cos((ii-1)*2*pi/(b-a)*(x-(b+a)/2))
%Fourier Transform:                sin((1/4*(-1)^(ii)-1/4+ii/2)*2*pi/(b-a)*(x-(b+a)/2)+pi/4+pi/4*(-1)^(ii-1))

vec_fun=zeros(N,n);%basis function vector space
for ii=1:n
    vec_fun(:,ii)=createfun(ii,x_lin);
end

%% function to be fitted
orifun=@(x) x;
% square(x+pi/2)
% square(x)
% x
% exp(x)
% exp(-0.1.*x)*cos(x)
ori_fun=orifun(x_lin);


%% generate the orthonormal basis
normal_fun=vec_fun;
% Normalize the first vector
for ii=1:N
    coeff=sum(delta.*normal_fun(:,1).*normal_fun(:,1));
end
normal_fun(:,1)=normal_fun(:,1)./sqrt(coeff);
% Orthonormalize the n vectors
for ii=2:n
    % Orthonormalize
    for jj=1:ii-1
        coeff=sum(normal_fun(:,jj).*normal_fun(:,ii)).*delta;
        normal_fun(:,ii)=normal_fun(:,ii)-coeff.*normal_fun(:,jj);
    end
    % Normalize
    coeff=sum(delta.*normal_fun(:,ii).*normal_fun(:,ii));
    normal_fun(:,ii)=normal_fun(:,ii)./sqrt(coeff);
end

%% fit the fuction and draw the plot
coeff=zeros(n,1); %projection coefficient
fit_fun=zeros(N,1);

%control the distribution of the plot
%figure;rows=ceil(n/4);cols=4;

%Iterative fitting
for ii=1:n
    coeff(ii,1)=sum(ori_fun.*normal_fun(:,ii))*delta;
    fit_fun=fit_fun+coeff(ii).*normal_fun(:,ii);

    %subplot(rows, cols, ii);
    %plot(x_lin, ori_fun, 'r', 'LineWidth', 2, 'DisplayName', 'Original function');
    %hold on;
    %plot(x_lin, fit_fun, 'b', 'LineWidth', 2, 'DisplayName', ['Fitted function (Order ' num2str(ii) ')']);
    %contribution =coeff(ii) .* normal_fun(:,ii);
    %plot(x_lin,contribution,':','LineWidth',2, 'DisplayName', ['Addition fitted function (Order ' num2str(ii) ')']);
    %title(['Fitted Function of Order ' num2str(ii)]);grid on;
end

%draw the final plot
figure
plot(x_lin,ori_fun,'r','LineWidth',2);
hold on;
plot(x_lin,fit_fun,'b','LineWidth',2);