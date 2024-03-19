%% Test SPARTAN ===========================================================
%
%   This script tests a the functionality of the SPARTAN algorithim for
%   density aware point cloud subsampling
%
%   by Dillon Cislo 01/29/2023
%==========================================================================

% RUN THIS FIRST TO MAKE SURE ALL NECESSARY FILES ARE ON THE PATH!
[scriptDir, ~, ~] = fileparts(matlab.desktop.editor.getActiveFilename);
cd(scriptDir);

addpath(genpath(fullfile(scriptDir, '../..')));

clear scriptDir

%% Set up Pipeline ========================================================
clear; close all; clc;

numPoints = 5000;

%--------------------------------------------------------------------------
% Generate a probability distribution
%--------------------------------------------------------------------------

[X, Y] = meshgrid(linspace(1,10,100));

% Simple Gaussian ---------------------------------------------------------
% mu = 5 * [1 1]; 
% sigma = [1 0.25; 0.25, 0.5];
% 
% P = mvnrnd(mu, sigma, numPoints);
% probDistFunc = reshape(mvnpdf([X(:), Y(:)], mu, sigma), size(X));

% Gaussian Mixture --------------------------------------------------------
numGaussians = randi([3,6]);
mu = 5 * rand(numGaussians, 2) + 2.5;
sigma = reshape(0.25 * rand(2,numGaussians) + 0.125, [1, 2, numGaussians]);
gm = gmdistribution(mu, sigma);

P = random(gm, numPoints);
probDistFunc = reshape(pdf(gm, [X(:), Y(:)]), size(X));

%--------------------------------------------------------------------------
% Generate Subsamples
%--------------------------------------------------------------------------
% close all; clc;

numSamples = 100;

disp('SPARTAN sub-sampling:');
% profile on
tic; PSS = SPARTAN(P, numSamples); toc;
% profile viewer

disp('Random sub-sampling:');
tic; PSSR = randsample(numPoints, numSamples); PSSR = P(PSSR, :); toc;

%--------------------------------------------------------------------------
% View Results
% -------------------------------------------------------------------------
% close all; clc;

figure('units', 'normalized', 'outerposition', [0 0 0.5 1]);

subplot(1,2,1)
hold on
scatter(P(:,1), P(:,2), [], 0.8 * ones(1,3), 'x' );
scatter(PSS(:,1), PSS(:,2), 'filled', 'k');
contour(X, Y, probDistFunc);
hold off

axis equal
xlim([0 10]);
ylim([0 10]);

title('Density Aware Subsampling');

subplot(1,2,2)
hold on
scatter(P(:,1), P(:,2), [], 0.8 * ones(1,3), 'x' );
scatter(PSSR(:,1), PSSR(:,2), 'filled', 'k');
contour(X, Y, probDistFunc);
hold off

axis equal
xlim([0 10]);
ylim([0 10]);

title('Random Subsampling');

