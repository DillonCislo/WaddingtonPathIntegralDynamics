%% Test Gaussian KDE (1D) =================================================
clear; close all; clc;

numPoints = 3000;
mu = 0;
sigma = rand(1) + 0.1;
P = normrnd(0, sigma, [numPoints, 1]);
Q = linspace(-5, 5, 500).';

% Optimal bandwidth for univariate normal distribution
pointDensity = gaussianKDE(P, Q, [], (4/(3*numPoints))^(1/5) * sigma);

plot(Q, normpdf(Q, mu, sigma), 'Color', 'k', 'LineWidth', 2);
hold on
plot(Q, pointDensity, '--r', 'LineWidth', 2);
scatter(P, zeros(numPoints, 1), 'x', 'Color', 0.8 * ones(1,3));
hold off
