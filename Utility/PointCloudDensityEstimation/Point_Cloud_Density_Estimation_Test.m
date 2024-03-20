%% Point Cloud Density Estimation Test =====================================
%
%   This script tests the functionality of the point cloud density
%   estimation function
%
%   by Dillon Cislo 02/01/2023
%==========================================================================

% RUN THIS FIRST TO MAKE SURE ALL NECESSARY FILES ARE ON THE PATH!
[scriptDir, ~, ~] = fileparts(matlab.desktop.editor.getActiveFilename);
cd(scriptDir);

addpath(genpath(fullfile(scriptDir, '../../..')));

clear scriptDir

%% Generate Point Set/Height Graph ========================================
clear; close all; clc;

%--------------------------------------------------------------------------
% Draw Samples from a Given Probability Distribution
%--------------------------------------------------------------------------

dim = 2;
numPoints = 2000;
% rng(1, twister); % For reproducible random numbers
fprintf('Generating random samples... ');

% Uniform Samples on U[0,10]^2 --------------------------------------------

% X = 10 * rand(numPoints, 2);

% Simple Gaussian ---------------------------------------------------------
% mu = 5 * [1 1]; 
% sigma = [1 0.25; 0.25, 0.5];
% 
% X = mvnrnd(mu, sigma, numPoints);

% Gaussian Mixture --------------------------------------------------------
numGaussians = randi([3,5]);
mu = 5 * rand(numGaussians, 2) + 2.5;
sigma = reshape(0.25 * rand(2,numGaussians) + 0.125, [1, 2, numGaussians]);
gm = gmdistribution(mu, sigma);

X = random(gm, numPoints);

fprintf('Done\n');

%% Perform Point Cloud Density Estimation =================================
close all; clc;

% Option Handling ---------------------------------------------------------

% Set default subsampling options
ssMethod = 'spartan';
numSamples = min(numPoints, 10000);
omtIter = 500;
omtWasserThresh = 0;
qrpType = 'sobol';
qrpLeap = 0;
qrpSkip = 0;
qrpScramble = false;

% Set default neighborhood graph construction options
nGraphType = 'relaxedgabrielgraph';
kNN = -1; % 100;
betaParam = 1;
forceConnectivity = true;
graphOutputFormat = 'symmetric';

% Set default kernel density estimation options
kdsigma = -1;
upsampleEdges = true;
includeNonSamples = true;

% Set topological simplification options
persistenceThreshold = 0;
sizeThreshold = 5;
stabilityThreshold = 0;
distThreshold = 0;
collisionMergeMethod = 'sequential';
maxThreshold = 0;
plotBranchQuality = true;

% Set default general options
verbose = true;

% Run Density Estimation --------------------------------------------------
[density, XOut, XOutIDx, ENG, maxIDx, ...
    saddleIDx, EJT, branchIDx, kdsigma] = pointCloudDensityEstimation(X, ...
    'SubsamplingMethod', ssMethod, 'numSamples', numSamples, ...
    'OMTIterations', omtIter, 'omtWasserThresh', omtWasserThresh, ...
    'QRPType', qrpType, 'QRPLeap', qrpLeap, 'QRPSkip', qrpSkip, ...
    'QRPScramble', qrpScramble, 'NeighborhoodGraphType', nGraphType, ...
    'NumNeighbors', kNN, 'GraphBeta', betaParam, ...
    'ForceConnectivity', forceConnectivity, ...
    'GraphOutputFormat', graphOutputFormat, ...
    'KDSigma', kdsigma, 'UpsampleEdges', upsampleEdges, ...
    'IncludeNonSamples', includeNonSamples, ...
    'PersistenceThreshold', persistenceThreshold, ...
    'SizeThreshold', sizeThreshold, ...
    'StabilityThreshold', stabilityThreshold, ...
    'DistanceThreshold', distThreshold, ...
    'CollisionMergeMethod', collisionMergeMethod, ...
    'MaxThreshold', maxThreshold, ...
    'PlotBranchQuality', plotBranchQuality, 'Verbose', verbose );

fprintf('N = %d maxima detected\n', numel(maxIDx));
fprintf('N0 = %d synthetic peaks\n', size(mu, 1));

%% View Results ===========================================================
close all; clc;

fprintf('N = %d maxima detected\n', numel(maxIDx));
fprintf('N0 = %d synthetic peaks\n', size(mu, 1));

upsampleIDx = isnan(XOutIDx);

clusterIDx = knnsearch(XOut(maxIDx, :), XOut);
% clusterIDx = changem(branchIDx, (1:numel(unique(branchIDx))).', unique(branchIDx));
clusterColors = brewermap(numel(maxIDx), 'Set2');
clusterColors = clusterColors(clusterIDx, :);

figure('Color', 'k');

clear axArray
axArray(1) = subplot(1,2,1);
patch('Faces', ENG, 'Vertices', XOut, 'FaceVertexCData', density, ...
    'FaceColor', 'interp', 'EdgeColor', 'interp', 'LineWidth', 2);
hold on
scatter(XOut(:,1), XOut(:,2), [], density, 'filled');
scatter(mu(:,1), mu(:,2), 100, 's', 'm', 'LineWidth', 3);
scatter(XOut(maxIDx,1), XOut(maxIDx,2), 100, 'filled', 'r');
scatter(XOut(saddleIDx,1), XOut(saddleIDx,2), 100, 'filled', 'g');
scatter(XOut(1,1), XOut(1,2), 100, 'filled', 'w');
hold off
axis equal tight
colorbar('Color', 'w');
set(gca, 'Clim', [0, max(density)]);
set(gca, 'Color', 'k');
set(gca, 'XColor', 'w');
set(gca, 'YColor', 'w');
title('Max/Saddle Structure', 'Color', 'w');
% xlim([0 10]); ylim([0 10]);


axArray(2) = subplot(1,2,2);
patch('Faces', ENG, 'Vertices', XOut, 'FaceVertexCData', density, ...
    'FaceColor', 'interp', 'EdgeColor', 'interp', 'LineWidth', 2);
hold on
scatter(XOut(:,1), XOut(:,2), [], clusterColors, 'filled');
scatter(mu(:,1), mu(:,2), 100, 's', 'm', 'LineWidth', 3);
scatter(XOut(maxIDx,1), XOut(maxIDx,2), 100, 'filled', 'r');
scatter(XOut(saddleIDx,1), XOut(saddleIDx,2), 100, 'filled', 'g');
scatter(XOut(1,1), XOut(1,2), 100, 'filled', 'w');
hold off
axis equal tight
colorbar('Color', 'w');
set(gca, 'Clim', [0, max(density)]);
set(gca, 'Color', 'k');
set(gca, 'XColor', 'w');
set(gca, 'YColor', 'w');
title('Density Based Clustering', 'Color', 'w');
% xlim([0 10]); ylim([0 10]);

linkaxes(axArray);


