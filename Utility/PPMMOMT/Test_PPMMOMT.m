%% Test PPMMOMT ===========================================================
%
%   This is a script to test the functionality of an implementation of the
%   projection pursuit Monge map (PPMM) optimal mass transport (OMT)
%   algorithim.
%
%   by Dillon Cislo 01/28/2023
%==========================================================================

%% Data Points with Equal Weights =========================================
clear; close all; clc;

% Generate data point from multivariate Gaussian distribution -------------
numPoints = 10000;
dim = 10;

mu1 = ones(1, dim);
mu2 = -ones(1, dim);

S1 = zeros(dim); S2 = zeros(dim);
for i = 1:dim
    for j = 1:dim
        S1(i,j) = 0.8^(abs(i-j));
        S2(i,j) = 0.5^(abs(i-j));
    end
end

% rng default % For reproducible random numbers
source = mvnrnd(mu1, S1, numPoints);
target = mvnrnd(mu2, S2, numPoints);

% Calculate the true Wasserstein distance
wasserDistTrue = sqrt( sum((mu1-mu2).^2) + ...
    trace( S1 + S2 - 2 * sqrtm(sqrtm(S1) * S2 * sqrtm(S1)) ) );

% Run PPMMOMT -------------------------------------------------------------

maxIter = 100;
wasserDelta = 0;

tic
[moving, wasserDist, allMoving] = ...
    PPMMOMT(source, target, [], [], maxIter, wasserDelta);
toc

% View Results ------------------------------------------------------------

plot([1 maxIter], wasserDistTrue * [1 1], '--k', 'LineWidth', 2);
hold on
plot(1:maxIter, wasserDist, '-b', 'LineWidth', 2);
hold off

xlabel('Number of Iterations');
ylabel('Wasserstein Distance');

legend({'True Wasserstein Distance', 'PPMM (SAVE)'}, ...
    'Location', 'southeast');