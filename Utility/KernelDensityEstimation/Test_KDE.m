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

%% Test Gaussian KDE (1D -- Self-Sample) ==================================
clear; close all; clc;

numPoints = 100;
mu = 0;
sigma = rand(1) + 0.1;
P = sort(normrnd(0, sigma, [numPoints, 1]));
Q = linspace(-5, 5, 500).';

% Optimal bandwidth for univariate normal distribution
distMatrix = pdist2(P, P, 'euclidean');
pointDensity = gaussianKDE(P, P, [], (4/(3*numPoints))^(1/5) * sigma);
pointDensityLOO = gaussianKDE(P, P, [], (4/(3*numPoints))^(1/5) * sigma, ...
    false, [], distMatrix, true, true);

% Identical computation using the less flexible 'estimatePotentialAndGrad'
% [~, ~, ~, pointDensityLOO] = estimatePotentialAndGrad(P, P, ...
%     (4/(3*numPoints))^(1/5) * sigma, 'ExcludeSelf', true);
% pointDensityLOO = pointDensityLOO ./ ...
%     sqrt(2 * pi * ((4/(3*numPoints))^(1/5) * sigma).^2);

fprintf('Maximum difference (LOO) = %0.5e\n', ...
    max(abs(pointDensity - pointDensityLOO)));

plot(Q, normpdf(Q, mu, sigma), 'Color', 'k', 'LineWidth', 2);
hold on
plot(P, pointDensity, '--r', 'LineWidth', 2);
plot(P, pointDensityLOO, '--b', 'LineWidth', 2);
scatter(P, zeros(numPoints, 1), 'x', 'Color', 0.8 * ones(1,3));
hold off

%% Test Gaussian KDE (N-D -- Self-Sample) =================================
clear; close all; clc;

%--------------------------------------------------------------------------
% Draw Samples from a Given Probability Distribution
%--------------------------------------------------------------------------

dim = 3;
numPoints = 5000;
% rng(1, twister); % For reproducible random numbers
fprintf('Generating random samples in %d-D... ', dim);

% Uniform Samples on U[0,10]^dim --------------------------------------------

% X = 10 * rand(numPoints, dim);
% truePDF = ones(numPoints, 1) ./ 10.^dim;

% Simple Gaussian ---------------------------------------------------------
% mu = 5 * ones(1, dim); 
% % sigma = [1 0.25; 0.25, 0.5];
% sigma = 0.75 * rand(dim) + 0.25;
% 
% X = mvnrnd(mu, sigma, numPoints);
% truePDF = mvnpdf(X, repmat(mu, [numPoints, 1]), sigma);

% Gaussian Mixture --------------------------------------------------------
numGaussians = randi([3,5]);
mu = 5 * rand(numGaussians, dim) + 2.5;
sigma = reshape(0.25 * rand(dim, numGaussians) + 0.125, [1, dim, numGaussians]);
gm = gmdistribution(mu, sigma);

X = random(gm, numPoints);
truePDF = pdf(gm, X);

fprintf('Done\n\n');

%--------------------------------------------------------------------------
% Perform Kernel Density Estimation
%--------------------------------------------------------------------------

[~, sigmaKDE] = knnsearch(X, X, 'K', 5);
sigmaKDE = mean(sigmaKDE(:, end));
% sigmaKDE = 0.25;

% Scott-Silverman rule (assumes data is normally distributed which of
% course is not true)
% sigmaKDE = sqrt(sum(sum((X - mean(X,1)).^2, 2), 1) ./ numPoints);
% sigmaKDE = sigmaKDE .* (4 ./ (numPoints.* (dim+2))).^(1/(dim+4));

pointDensity = gaussianKDE(X, X, [], sigmaKDE);
pointDensityLOO = gaussianKDE(X, X, [], sigmaKDE, ...
    false, [], [], true, true);

% Identical computation using less flexible 'estimatePotentialAndGrad'
% [~, ~, ~, pointDensityLOO] = estimatePotentialAndGrad(X, X, ...
%     sigmaKDE, 'ExcludeSelf', true);
% pointDensityLOO = pointDensityLOO ./ (2 * pi * sigmaKDE.^2).^(dim/2);

pdfErr = abs(pointDensity - truePDF) ./ abs(truePDF);
pdfErrLOO = abs(pointDensityLOO - truePDF) ./ abs(truePDF);

fprintf('Maximum Error = %0.5e\n', max(pdfErr));
fprintf('Mean Error = %0.5e\n', mean(pdfErr));
fprintf('Median Error = %0.5e\n\n', median(pdfErr));

fprintf('Maximum Error (LOO) = %0.5e\n', max(pdfErrLOO));
fprintf('Mean Error (LOO) = %0.5e\n', mean(pdfErrLOO));
fprintf('Median Error (LOO) = %0.5e\n', median(pdfErrLOO));