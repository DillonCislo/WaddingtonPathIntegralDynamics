%% Test Nystroem Out-of-Sample Extension for Diffusion Coordinates ========
clear; close all; clc;

numPoints = 5000;
numPointsOOS = 250;

%--------------------------------------------------------------------------
% Randomly Generate Points Inside a "Dumbbell" in R^3
%--------------------------------------------------------------------------
disp('Generating random point sets: ')

L = 10; % Length of the dumbell's handle
R = 3; % Radii of spheres at the end of the dumbbell
rho = 1; % Cylindrical radius of the handle

insideDumbell = @(x) ...
    (sqrt(sum((x-[L/2 0 0]).^2, 2)) < R) | ...
    (sqrt(sum((x+[L/2 0 0]).^2, 2)) < R) | ...
    ((sqrt(sum(x(:, 2:3).^2, 2)) < rho) & (abs(x(:,1)) < L));

X = nan(numPoints+numPointsOOS, 3);
numFound = 0;
while numFound < (numPoints+numPointsOOS)

    progressbar(numFound, numPoints+numPointsOOS);

    newPoint = (L+2*R) * rand(1, 3) - (L/2+R);
    if insideDumbell(newPoint)
        numFound = numFound+1;
        X(numFound, :) = newPoint;
    end
end
assert(~any(isnan(X(:))), 'Bad point generation failed');
XOOS = X((numPoints+1):end, :);
X = X(1:numPoints, :);

clear newPoint numFound

%--------------------------------------------------------------------------
% Generate Diffusion Map Embedding for In-Sample Points
%--------------------------------------------------------------------------

affinityOptions = struct();
affinityOptions.DistanceType = 'euclidean';
affinityOptions.NumNeighbors = 500;
% affinityOptions.SelfTune = 7;
affinityOptions.Sigma = -1;

fprintf('\nGenerating in-sample affinity kernel matrix... ');
[K, ~, ~, sigma] = affinityMatrix(X, affinityOptions);
fprintf('Done\n');

mapOptions = struct();
mapOptions.Normalization = 'Markov';
mapOptions.t = 1;
mapOptions.NumVectors = 30;
mapOptions.NormalizeDensity = false;

fprintf('Generating in-sample diffusion embedding... ');
[diffCoords, lambda] = diffusionMap(K, mapOptions);
fprintf('Done\n');

%--------------------------------------------------------------------------
% Perform Out-of-Sample Extension
%--------------------------------------------------------------------------

fprintf('Performing out-of-sample extension... ');
diffCoordsOOS = nystroemOOS(XOOS, X, diffCoords, lambda, K, ...
    affinityOptions, mapOptions);
fprintf('Done\n');

%--------------------------------------------------------------------------
% View Results
%--------------------------------------------------------------------------

% Diffusion embedding visualization axes
axID1 = 1;
axID2 = 2;
axID3 = 3;

inAlpha = 0.25;

X3D = [diffCoords(:, axID1), diffCoords(:, axID2), diffCoords(:, axID3)];
X3DOOS = [diffCoordsOOS(:, axID1), diffCoordsOOS(:, axID2), diffCoordsOOS(:, axID3)];

inColors = vals2colormap(X(:,1), 'parula', (L/2+R) * [-1 1]);
outColors = vals2colormap(XOOS(:,1), 'parula', (L/2+R) * [-1 1]);

figure('Color', 'k');

subplot(1,2,1)
pcshow(X, inColors, 'MarkerSize', 2);
hold on
pcshow(XOOS, outColors, 'MarkerSize', 300);
hold off
% scatter3(X(:,1), X(:,2), X(:,3), [], inColors, 'x', ...
%     'MarkerEdgeAlpha', inAlpha, 'MarkerFaceAlpha', inAlpha);
% hold on
% scatter3(XOOS(:,1), XOOS(:,2), XOOS(:,3), [], outColors, 'filled');
% hold off
axis equal
set(gca, 'Color', 'k', 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');
camproj('orthographic');
xlabel('X');
ylabel('Y');
zlabel('Z');
title('Original Point Cloud', 'Color', 'w');

subplot(1,2,2)
pcshow(X3D, inColors, 'MarkerSize', 2);
hold on
pcshow(X3DOOS, outColors, 'MarkerSize', 300);
hold off
% scatter3(X3D(:,1), X3D(:,2), X3D(:,3), [], inColors, 'x', ...
%     'MarkerEdgeAlpha', inAlpha, 'MarkerFaceAlpha', inAlpha);
% hold on
% scatter3(X3DOOS(:,1), X3DOOS(:,2), X3DOOS(:,3), [], outColors, 'filled');
% hold off
axis equal
set(gca, 'Color', 'k', 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');
camproj('orthographic');
xlabel(sprintf('\\Phi_%d', axID1));
ylabel(sprintf('\\Phi_%d', axID2));
zlabel(sprintf('\\Phi_%d', axID3));
title('Diffusion Embedding', 'Color', 'w');


