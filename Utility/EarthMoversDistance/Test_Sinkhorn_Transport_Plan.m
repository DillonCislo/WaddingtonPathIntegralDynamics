%% TEST SINKHORN TRANSPORT PLAN ===========================================
% This is a script to test the functionality of the 'sinkhornDistance'
% routine in producing a transport plan that aligns two distributions

%% Test Transport Plan 1D =================================================
clear; close all; clc;

numPoints = 5e3;
maxIter = 1e4;
maxDelta = 1e-4;
lambda = 75;

% Generate two random distributions ---------------------------------------
numGaussiansP = randi([1, 5]);
muP = 5 * rand(numGaussiansP, 1) + 2.5;
sigmaP = permute(0.25 * rand(numGaussiansP, 1) + 0.125, [3 2 1]);
gmP = gmdistribution(muP, sigmaP);
P = random(gmP, numPoints);
PW = ones(numPoints, 1) ./ numPoints;

numGaussiansQ = randi([1, 5]);
muQ = 5 * rand(numGaussiansQ, 1) + 2.5;
sigmaQ = permute(0.25 * rand(numGaussiansQ, 1) + 0.125, [3 2 1]);
gmQ = gmdistribution(muQ, sigmaQ);
Q = random(gmQ, numPoints);
QW = ones(numPoints, 1) ./ numPoints;

% Estimate the regularized optimal transport plan -------------------------
M = pdist2(P, Q, 'squaredeuclidean');
M = M ./ median(M(:));

[D, T] = sinkhornDistance( PW, QW, 'CostMatrix', M, ...
    'Lambda', lambda, 'MaxIterations', maxIter, ...
    'useGPU', true, 'Verbose', true, 'Stability', 'stable', ...
    'MaxDelta', maxDelta);

% Try to transport the points using barycentric averaging -----------------

movP = (T * Q) ./ sum(T,2);
movQ = (T.' * P) ./ sum(T,1).';

%--------------------------------------------------------------------------
% View Results
%--------------------------------------------------------------------------

[fp, xp] = kde(P);
[fq, xq] = kde(Q);

[fmp, xmp] = kde(movP);
[fmq, xmq] = kde(movQ);

subplot(1,3,1);
plot(xp, fp, 'LineWidth', 2);
hold on
plot(xq, fq, 'LineWidth', 2);
hold off
legend({"P", "Q"});

subplot(1,3,2);
plot(xp, fp, 'LineWidth', 2);
hold on
plot(xmq, fmq, 'LineWidth', 2);
hold off
legend({"P", "movQ"});

subplot(1,3,3);
plot(xmp, fmp, 'LineWidth', 2);
hold on
plot(xq, fq, 'LineWidth', 2);
hold off
legend({"movP", "Q"});


%% Test Transport Plan 2D =================================================
clear; close all; clc;

dim = 2;
numPoints = 5e3;
maxIter = 1e4;
maxDelta = 1e-4;
lambda = 125;

% Generate two random distributions ---------------------------------------
numGaussiansP = randi([1, 5]);
muP = 5 * rand(numGaussiansP, dim) + 2.5;
sigmaP = reshape(0.25 * rand(dim, numGaussiansP) + 0.125, [1, dim, numGaussiansP]);
gmP = gmdistribution(muP, sigmaP);
P = random(gmP, numPoints);
PW = ones(numPoints, 1) ./ numPoints;

numGaussiansQ = randi([1, 5]);
muQ = 5 * rand(numGaussiansQ, dim) + 2.5;
sigmaQ = reshape(0.25 * rand(dim, numGaussiansQ) + 0.125, [1, dim, numGaussiansQ]);
gmQ = gmdistribution(muQ, sigmaQ);
Q = random(gmQ, numPoints);
QW = ones(numPoints, 1) ./ numPoints;

% Estimate the regularized optimal transport plan -------------------------
M = pdist2(P, Q, 'squaredeuclidean');
M = M ./ median(M(:));

[D, T] = sinkhornDistance( PW, QW, 'CostMatrix', M, ...
    'Lambda', lambda, 'MaxIterations', maxIter, ...
    'useGPU', true, 'Verbose', true, 'Stability', 'stable', ...
    'MaxDelta', maxDelta);

% Try to transport the points using barycentric averaging -----------------

movP = (T * Q) ./ sum(T,2);
movQ = (T.' * P) ./ sum(T,1).';

%--------------------------------------------------------------------------
% View Results
%--------------------------------------------------------------------------

[X, Y] = meshgrid(linspace(0,10,100), linspace(0,10,100));

fp = reshape(ksdensity(P, [X(:), Y(:)]), size(X));
fq = reshape(ksdensity(Q, [X(:), Y(:)]), size(X));

fmp = reshape(ksdensity(movP, [X(:), Y(:)]), size(X));
fmq = reshape(ksdensity(movQ, [X(:), Y(:)]), size(X));

figure('Color', 'w');

subplot(2,2,1)
hold on
% scatter(P(:,1), P(:,2), [], 'mx');
contourf(X, Y, fp, 'LineWidth', 1, 'Color', 'm');
hold off
axis equal tight
xlim([0 10]);
ylim([0 10]);

colorbar
set(gca, 'Clim', prctile(fp(:), [0 100]));
title('P')

subplot(2,2,2)
hold on
% scatter(Q(:,1), Q(:,2), [], 'cx');
contourf(X, Y, fq, 'LineWidth', 1, 'Color', 'c');
hold off
axis equal tight
xlim([0 10]);
ylim([0 10]);

colorbar
set(gca, 'Clim', prctile(fq(:), [0 100]));
title('Q')


subplot(2,2,3)
hold on
% scatter(movP(:,1), movP(:,2), [], 'mx');
contourf(X, Y, fmp, 'LineWidth', 1, 'Color', 'm');
hold off
axis equal tight
xlim([0 10]);
ylim([0 10]);

colorbar
set(gca, 'Clim', prctile(fmp(:), [0 100]));
title('movP')

subplot(2,2,4)
hold on
% scatter(movQ(:,1), movQ(:,2), [], 'cx');
contourf(X, Y, fmq, 'LineWidth', 1, 'Color', 'c');
hold off
axis equal tight
xlim([0 10]);
ylim([0 10]);

colorbar
set(gca, 'Clim', prctile(fmq(:), [0 100]));
title('movQ')



%%

numGaussians = randi([3,5]);
mu = 5 * rand(numGaussians, 2) + 2.5;
sigma = reshape(0.25 * rand(2,numGaussians) + 0.125, [1, 2, numGaussians]);
gm = gmdistribution(mu, sigma);

X = random(gm, numPoints);