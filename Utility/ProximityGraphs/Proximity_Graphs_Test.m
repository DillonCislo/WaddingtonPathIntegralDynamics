%% Proximity Graphs Test ==================================================
%
%   This script tests the functionality of the NGL library to produce a
%   variety of different neighborhood graphs on an N-dimensional point
%   cloud.
%
%   by Dillon Cislo 02/01/2023
%==========================================================================

% RUN THIS FIRST TO MAKE SURE ALL NECESSARY FILES ARE ON THE PATH!
[scriptDir, ~, ~] = fileparts(matlab.desktop.editor.getActiveFilename);
cd(scriptDir);

addpath(genpath(fullfile(scriptDir, '../../..')));

clear scriptDir

%% Exhaustive vs. Approximate Equivalence Tests ===========================
% Use a relatively small number of points for this
clear; close all; clc;

%--------------------------------------------------------------------------
% Draw Samples from a Given Probability Distribution
%--------------------------------------------------------------------------

numPoints = 500;
% rng default
fprintf('Generating random samples... ');

% Uniform Samples on U[0,10]^2 --------------------------------------------

% X = 10 * rand(numPoints, 2);

% Simple Gaussian ---------------------------------------------------------
% mu = 5 * [1 1]; 
% sigma = [1 0.25; 0.25, 0.5];
% 
% X = mvnrnd(mu, sigma, numPoints);

% Gaussian Mixture --------------------------------------------------------
numGaussians = randi([3,6]);
mu = 5 * rand(numGaussians, 2) + 2.5;
sigma = reshape(0.25 * rand(2,numGaussians) + 0.125, [1, 2, numGaussians]);
gm = gmdistribution(mu, sigma);

X = random(gm, numPoints);

fprintf('Done\n');

%--------------------------------------------------------------------------
% Generate Proximity Graphs
%--------------------------------------------------------------------------

graphType = 'rrng';
outputFormat = 'mutual';
kNN = 100;
betaParam = 1;
forceConnectivity = true;

fprintf('\nGenerating exhaustive graph:\n');
tic
[E, DE] = proximityGraphsNGL(X, graphType, -1, ...
    betaParam, forceConnectivity, outputFormat);
toc


fprintf('\nGenerating approximate graph:\n');
tic
[approxE, approxDE] = proximityGraphsNGL(X, graphType, kNN, ...
    betaParam, forceConnectivity, outputFormat);
toc

%--------------------------------------------------------------------------
% Validate Results
%--------------------------------------------------------------------------
disp(' ');

% Check for graph equality
if isequal(sortrows(sort(E,2)), sortrows(sort(approxE, 2)))
    disp('Exhaustive graph and approximate graphs are equal');
else
    disp('Exhaustive graph and approximate graphs are NOT equal');
end

% Check that all approximate edges are contained in the exhaustive graph
if all(ismember(approxE, E, 'rows'))
    disp(['All approximate graph edges are contained in the ' ...
        'exhaustive graphs']);
else
    warning(['Some approximate graph edges are not contained ' ...
        'in the exhaustive graphs!']);
end

% Check that all points are included in both graphs
if all(ismember((1:numPoints), E))
    disp('All points are included in the exhaustive graph');
else
    warning('Some points are NOT included in the exhaustive graph!');
end

if all(ismember((1:numPoints), approxE))
    disp('All points are included in the the approximate graph');
else
    warning('Some points are NOT included in the NGL graph!');
end

% Check that the exhaustive graph is simply connected
if any(strcmpi(outputFormat, {'mutual', 'symmetric'}))
    numCC = graphConnectedComponents(E, numPoints);
    if (numCC == 1)
        disp('Exhaustive graph is simply connected');
    else
        warning('Exhaustive graph is NOT simply connected');
    end
end

% Check that approximate graph is simply connected
if any(strcmpi(outputFormat, {'mutual', 'symmetric'}))
    numCC = graphConnectedComponents(approxE, numPoints);
    if (numCC == 1)
        disp('Approximate graph is simply connected');
    else
        warning('Approximate graph is NOT simply connected');
    end
end

if isempty(E)
    missingEdges = false(size(approxE,1), 1);
    missingNodes = false(numPoints, 1);
else
    missingEdges = ~ismember(E, approxE, 'rows');
    missingNodes = ismember((1:numPoints).', E(missingEdges, :));
end


%--------------------------------------------------------------------------
% View Results - Complete Comparison
%--------------------------------------------------------------------------
close all; % clc;

switch graphType
    case 'nng'
        titleString = 'Nearest Neighbor Graph'; 
    case 'mst'
        titleString = 'Minimum Spanning Tree';
    case 'rng'
        titleString = 'Relative Neighbor Graph';
    case 'gg'
        titleString = 'Gabriel Graph';
    case 'rrng'
        titleString = 'Relaxed Relative Neighbor Graph';
    case 'rgg'
        titleString = 'Relaxed Gabriel Graph';
    otherwise
        error('Invalid graph type');
end

edgeColors = zeros(size(E,1), 3);
edgeColors(missingEdges, :) = repmat([1 0 0], sum(missingEdges), 1);

nodeColors = zeros(size(X,1), 3);
nodeColors(missingNodes, :) = repmat([1 0 0], sum(missingNodes), 1);

figure('units', 'normalized', 'outerposition', [0 0 0.5 1]);

clear axArray
axArray(1) = subplot(1,2,1);
patch('Faces', reshape([1:size(E,1), 1:size(E,1)*2], [], 3), ...
    'Vertices', X(E(:), :), 'FaceVertexCData', repmat(edgeColors, 2, 1), ...
    'FaceColor', 'none', 'EdgeColor', 'interp', 'LineWidth', 2);
hold on
scatter(X(:,1), X(:,2), [], nodeColors, 'filled');
hold off
axis equal tight
xlim([0 10]); ylim([0 10]);
title(sprintf(['Exhaustive Construction:\n' titleString]));

axArray(2) = subplot(1,2,2);
patch('Faces', approxE, 'Vertices', X, 'LineWidth', 2);
hold on
scatter(X(:,1), X(:,2), [], nodeColors, 'filled');
hold off
axis equal tight
xlim([0 10]); ylim([0 10]);
if (kNN <= 0)
    title(sprintf(['Exhaustive Construction:\n' titleString]));
else
    title(sprintf(['Approximate Construction:\n' titleString]));
end

linkaxes(axArray);
