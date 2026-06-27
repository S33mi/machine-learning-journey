% You can run this file or copy past code in matlab and run manually
% You can past you own code as well to generate figure results
clear; close all;

% Parameters from Bishop Fig 1.4-1.7
N = 10;  % training points
M = 9;   % high degree to show overfitting

[x_train, t_train] = generate_data(N, 0.3);  % noise std=0.3

x_plot = linspace(0,1,100)';

% ML fits for different M (Figure 1.4 style)
figure;
for m = [0 1 3 9]
    w = fit_polynomial_ML(x_train, t_train, m);
    y = evaluate_polynomial(x_plot, w);
    subplot(2,2,find([0 1 3 9]==m));
    plot(x_train, t_train, 'bo', x_plot, y, 'r-', x_plot, sin(2*pi*x_plot), 'g--');
    title(['M=' num2str(m)]);
end
saveas(gcf, 'Figure_1-4.png');

% Similar for other figures (regularization effect, etc.)
% Figure 1.5: different noise levels / M
% Figure 1.6-1.7: effect of alpha (regularization)

% Final comparison ML vs MAP
alpha = 0.005;  % example value, tune as needed
beta = 1 / (0.3)^2;

w_ml = fit_polynomial_ML(x_train, t_train, M);
w_map = fit_polynomial_MAP(x_train, t_train, M, alpha, beta);

figure;
plot(x_train, t_train, 'bo');
hold on;
plot(x_plot, evaluate_polynomial(x_plot, w_ml), 'r-', 'LineWidth', 2);
plot(x_plot, evaluate_polynomial(x_plot, w_map), 'm-', 'LineWidth', 2);
plot(x_plot, sin(2*pi*x_plot), 'g--');
legend('Data', 'ML', 'MAP', 'True');
saveas(gcf, 'polynomial_fitting_ML_VS_MAP.png');

% ******Take aways*******

%Put all .m files in one folder.
%Run get_all_results.m.
%Tune alpha and beta until plots match Bishop's book (overfitting in ML, smoother in MAP).
%Generate the required PNGs.
%Common issues:
%Column vs row vectors → use ' transpose.
%Use \ for solving linear systems (numerically stable).
%High M (like 9) shows dramatic overfitting in ML.
