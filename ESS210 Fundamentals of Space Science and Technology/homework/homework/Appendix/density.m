f=@(x) (5/3+1).*x.^2./((5/3-1).*x.^2+2);
x=linspace(0,35,1000);
plot(x,f(x));
xlabel('M');
ylabel('rho2/rho1');