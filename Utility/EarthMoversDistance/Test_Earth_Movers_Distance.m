%% TEST EARTH MOVERS DISTANCE =============================================
% This is a script to test the functionality and speed of
% 'earthMoversDistance' and 'sinkhornDistance' in computing
% optimal transport distances between high dimensional point clouds

%% Compare 'sinkhornDistance' to 'earthMoversDistance' ====================
% This can only be done for small point clouds - 'earthMoversDistance'
% should be exact, but scales too poorly to be used for large point clouds;
clear; close all; clc;

numPoints = 500;
dim = 5;
maxIter = 1e4;
maxDelta = 1e-4;
pointCloudType = 'test';

if strcmpi(pointCloudType, 'random')

    % X = 10 .* rand(numPoints, dim) - 5;
    X = rand(numPoints, dim);

    r = rand(numPoints, 1);
    % r(randperm(numPoints, ceil(numPoints/4))) = 0;
    r = r ./ sum(r);
    c = rand(numPoints, 1); c = c ./ sum(c);

elseif strcmpi(pointCloudType, 'test')
    load('Test_Point_Cloud_5D.mat');

    if (size(X,1) > numPoints)
        keepIDx = randsample(1:size(X,1), numPoints);
        X = X(keepIDx, :);
        r = r(keepIDx); r = r ./ sum(r);
        c = c(keepIDx); c = c ./ sum(c);
    elseif (size(X,1) < numPoints)
        error('Too many points');
    end

else
    error('Invalid point cloud type');
end

M = pdist2(X, X, 'squaredeuclidean');
% M = M ./ median(M(:));

allLambda = [0, 1, 2, 5, 10, 15, 25, 50, 100] ./ median(M(:));
% allLambda = (0:100) ./ median(M(:));
% allLambda = [0, 1, 2, 5, 10, 15, 25, 50, 100];

if (numPoints < 1e3)

    fprintf('Computing true EMD: ');
    tic;
    emd = earthMoversDistance(X, r, X, c, M);
    emdTime = toc;
    fprintf('Elapsed time is %0.5f\n', emdTime);

else

    emd = NaN;
    emdTime = NaN;

end

fprintf('Computing EMD (Sinkhorn)... ')
emdSD = zeros(1, numel(allLambda));
sdTimes = zeros(1, numel(allLambda));
for i = 1:numel(allLambda)
    tic;
    emdSD(i) = sinkhornDistance(r, c, 'CostMatrix', M, ...
        'Lambda', allLambda(i), 'MaxIterations', maxIter, ...
        'useGPU', true, 'Verbose', false, 'Stability', 'stable', ...
        'MaxDelta', maxDelta);
    sdTimes(i) = toc;
end
fprintf('Done\n')

% View Results ------------------------------------------------------------

figure('Color', 'w');

subplot(1,3,1)
plot([allLambda(1) allLambda(end)] * median(M(:)), emd * [1 1], ':k', 'LineWidth', 2);
hold on
plot(allLambda .* median(M(:)), emdSD, '-b', 'LineWidth', 2);
hold off
xlabel('\lambda * q_{50}(M)')
ylabel('Sinkhorn Distance');
xlim([allLambda(1) allLambda(end)] .* median(M(:)));
set(gca, 'YScale', 'log')
axis square

subplot(1,3,2)
plot(allLambda .* median(M(:)), abs(emdSD-emd) ./ emd, '-b', 'LineWidth', 2);
xlabel('\lambda * q_{50}(M)')
ylabel('||SD - EMD|| / EMD');
xlim([allLambda(1) allLambda(end)] .* median(M(:)));
set(gca, 'YScale', 'log')
axis square

subplot(1,3,3)
plot([allLambda(1) allLambda(end)] * median(M(:)), emdTime * [1 1], ':k', 'LineWidth', 2);
hold on
plot(allLambda .* median(M(:)), sdTimes, '-b', 'LineWidth', 2);
hold off
xlabel('\lambda * q_{50}(M)')
ylabel('Computation Time (s)');
xlim([allLambda(1) allLambda(end)] .* median(M(:)));
set(gca, 'YScale', 'log')
axis square

%% Compute Time/Accuracy For Sinkhorn Distances on Large Point Clouds =====
clear; close all; clc;

numPoints = 8000;
dim = 5;
maxIter = 1e4;
maxDelta = 1e-4;
numDists = 250;

% X = 10 .* rand(numPoints, dim) - 5;
X = rand(numPoints, dim);
M = pdist2(X, X, 'euclidean');
% M = M ./ median(M(:));

r = rand(numPoints, 1);
% r(randperm(numPoints, ceil(numPoints/4))) = 0;
r = r ./ sum(r);
c = rand(numPoints, numDists); c = c ./ sum(c, 1);

allLambda = [0, 1, 2, 5, 10, 15, 25, 50, 100] ./ median(M(:));
% allLambda = [0, 1, 2, 5, 10] ./ median(M(:));
% allLambda = (0:100) ./ median(M(:));
% allLambda = [0, 1, 2, 5, 10, 15, 25, 50, 100];
% allLambda = 250  ./ median(M(:));
% allLambda = 100;

fprintf('Computing EMD (Sinkhorn - Stable)... ')
emdSD = zeros(1, numel(allLambda));
sdTimes = zeros(1, numel(allLambda));
for i = 1:numel(allLambda)
    tic;
    % profile on
    emdSD(i) = min(sinkhornDistance(r, c, 'CostMatrix', M, ...
        'Lambda', allLambda(i), 'MaxIterations', maxIter, ...
        'useGPU', true, 'Verbose', false, 'Stability', 'stable'));
    % profile viewer
    sdTimes(i) = toc;
end
fprintf('Done\n')

fprintf('Computing EMD (Sinkhorn - Unstable)... ')
emdSDUnstable = zeros(1, numel(allLambda));
sdTimesUnstable = zeros(1, numel(allLambda));
for i = 1:numel(allLambda)
    tic;
    % profile on
    emdSDUnstable(i) = min(sinkhornDistance(r, c, 'CostMatrix', M, ...
        'Lambda', allLambda(i), 'MaxIterations', maxIter, ...
        'useGPU', true, 'Verbose', false, 'Stability', 'unstable'));
    % profile viewer
    sdTimesUnstable(i) = toc;
end
fprintf('Done\n')


%% View Results ------------------------------------------------------------

figure('Color', 'w');

subplot(1,2,1)
plot(allLambda .* median(M(:)), emdSD, '-b', 'LineWidth', 2);
hold on
plot(allLambda .* median(M(:)), emdSDUnstable, '-r', 'LineWidth', 2);
hold off
xlabel('\lambda * q_{50}(M)')
ylabel('Sinkhorn Distance');
xlim([allLambda(1) allLambda(end)] .* median(M(:)));
set(gca, 'YScale', 'log')
axis square

subplot(1,2,2)
plot(allLambda .* median(M(:)), sdTimes, '-b', 'LineWidth', 2);
hold on
plot(allLambda .* median(M(:)), sdTimesUnstable, '-r', 'LineWidth', 2);
hold off
xlabel('\lambda * q_{50}(M)')
ylabel('Computation Time (s)');
xlim([allLambda(1) allLambda(end)] .* median(M(:)));
set(gca, 'YScale', 'log')
axis square
