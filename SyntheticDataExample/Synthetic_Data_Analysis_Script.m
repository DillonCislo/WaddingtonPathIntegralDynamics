%% Synthetic Data Analysis Script =========================================
%
%   This script illustrates the implementation of the data analysis
%   protocol on a set of synthetic data generated from a flip potential at
%   different parameter values
%
%   SUMMARY OF WORKFLOW (READ ME FIRST):
%       1. Generates synthetic stochastic trajectories under a heteroclinic
%          flip potential and records time-sampled point clouds.
%       2. Subsamples the cloud, estimates pseudopotentials, and infers
%          most-probable transition pathways.
%       3. Fits static potentials across conditions, simulates probability
%          transport, and benchmarks against analytic ground truth.
%       4. Visualizes densities, potentials, and ground states to mirror
%          the analysis presented in the WPID manuscript.
%
%   by Dillon Cislo 2024/11/05
%
%==========================================================================

[scriptDir, ~, ~] = fileparts(matlab.desktop.editor.getActiveFilename);
cd(scriptDir);

addpath(genpath(fullfile(scriptDir, '../../WaddingtonPathIntegralDynamics')));

clear scriptDir

%% ************************************************************************
% *************************************************************************
%           PART 1: SET UP PIPELINE/GENERATE SYNTHETIC DATA
% *************************************************************************
% *************************************************************************

%% Generate Analytic Results for Specified Potentials =====================
clear; close all; clc;

[projectDir, ~, ~] = fileparts(matlab.desktop.editor.getActiveFilename);
cd(projectDir);

syms x y a b
assume(x, 'real'); assume(y, 'real');
assume(a, 'real'); assume(b, 'real');

%--------------------------------------------------------------------------
% Set Synthetic "Experimental" Conditions
%--------------------------------------------------------------------------

% Heteroclininic Flip Potential
U = x.^4 + y.^4 + x.^3 - 2.*x.*y.^2 - x.^2 + a*x + b*y;

% List of "experimental" parameters
% allExpParam(:,1) == a, allExpParam(:,2) == b
allExpParam = [ -1.6, -0.4; -1.4 0.5; -1.1, -0.1];
numExpCond = size(allExpParam, 1);

%--------------------------------------------------------------------------
% Perform Symbolic Analysis of Potential
%--------------------------------------------------------------------------
fprintf('Performing symbolic analysis of potentials... ');

% Derivatives of the potential field
gradU = simplify(gradient(U, [x y]));
% lapU = simplify(laplacian(U, [x y]));

% Generate anonymous functions from analytic statements
UFunc = matlabFunction(U);
gradUFunc = matlabFunction(gradU.');
% lapUFunc = matlabFunction(lapU);

% Loop over each experimental condition
allFixedPts = cell(numExpCond, 1);
allFPLambdas = cell(numExpCond, 1);
allFPEigVecs = cell(numExpCond, 1);
allFPU = cell(numExpCond, 1);
for i = 1:numExpCond

    a_val = allExpParam(i,1);
    b_val = allExpParam(i,2);

    % Substitute current values of a and b into the gradient function
    gradU_sub = subs(gradU, [a, b], [a_val, b_val]);

    % Find fixed points of the deterministic gradient flow numerically
    fixedPts = vpasolve([gradU_sub(1) == 0; gradU_sub(2) == 0], [x y]);
    fixedPts = double([fixedPts.x(:), fixedPts(:).y]);

    % Assess stability of fixed points
    J = simplify([gradient(-gradU_sub(1), [x, y]), gradient(-gradU_sub(2), [x y])].');
    fpLambda = zeros(size(fixedPts,1), 1);
    fpEigVec = cell(size(fixedPts,1),2);
    for j = 1:numel(fpLambda)
        numJ = double(vpa(subs(J, {x, y}, {fixedPts(j,1), fixedPts(j,2)})));
        [V, D] = eig(numJ); D = diag(D);
        fpLambda(j) = max(real(D));
        fpEigVec{j,1} = V; fpEigVec{j,2} = D;
    end

    % Re-order fixed points by largest (real) eigenvalue
    [fpLambda, sortOrder] = sort(fpLambda, 'ascend');
    fixedPts = fixedPts(sortOrder, :);
    fpEigVec = fpEigVec(sortOrder, :);
    fpU = UFunc(a_val, b_val, fixedPts(:,1), fixedPts(:,2));

    allFixedPts{i} = fixedPts;
    allFPLambdas{i} = fpLambda;
    allFPEigVecs{i} = fpEigVec;
    allFPU{i} = fpU;

end

fprintf('Done\n');

clear i j a_val b_val gradU_sub fixedPts J fpLambda fpEigVec
clear numJ V D fpU sortOrder
clear a b x y U gradU lapU % Clear symbolic functions

%--------------------------------------------------------------------------
% View Results
%--------------------------------------------------------------------------

xLim = 2 * [-1 1];
yLim = 2 * [-1 1];
numPts = 100;
[X, Y] = meshgrid(linspace(xLim(1), xLim(2), numPts), ...
    linspace(yLim(1), yLim(2), numPts));

figure('Color', 'w');

for i = 1:numExpCond

    a = allExpParam(i,1);
    b = allExpParam(i,2);
    fpLambda = allFPLambdas{i};

    sinkIDx = fpLambda < 0;
    saddleIDx = fpLambda >= 0;

    subplot(1,3,i)
    imagesc(X(:), Y(:), UFunc(a, b, X, Y))

    hold on

    scatter(allFixedPts{i}(sinkIDx, 1), allFixedPts{i}(sinkIDx, 2), ...
        'filled', 'm', 'MarkerEdgeColor', 'k');
    scatter(allFixedPts{i}(saddleIDx, 1), allFixedPts{i}(saddleIDx, 2), ...
        'filled', 'c', 'MarkerEdgeColor', 'k');

    hold off

    axis equal square
    xlim(xLim);
    ylim(yLim);

    colorbar
    set(gca, 'Clim', [-1.5 2]);
    set(gca, 'Colormap', brewermap(256, '*YlGnBu'));
    set(gca, 'YDir', 'normal');

    title(sprintf('U(a = %0.2f, b = %0.2f)', a, b));

end

clear xLim yLim numPts X Y i a b fpLambda sinkIDx saddleIDx

%% Generate Points by Simulating Stochastic ODE ===========================
% The stochastic ODE is simulated using basic Euler-Maruyama. Tracks are
% terminated upon sampling so that no two tracks are sampled at different
% times/conditions.
close all; clc;

rng(1, 'twister'); % For reproducible random numbers

numTotalPoints = 1.8e5; % The total number of points across all times/conditions
numExpTimes = 5; % The total number of sampled time points (including T = 0)

assert((rem(numTotalPoints, numExpCond) == 0) && ...
    (rem(numTotalPoints, numExpTimes) == 0), ['Please pick a total ' ...
    'number of points that is divisible by both the number of ' ...
    'experimental conditions and the number of sample times. It''s ' ...
    'just easier.']);

sodeSettings = struct();
sodeSettings.DSODE = 1; % THe diffusion coefficient for the stochastic ODE simulation
sodeSettings.dtSODE = 1e-4; % The time step for the stochastic ODE simulation
sodeSettings.initSigma = 0.05; % The variance of the Gaussian initial condition

% The number of simulation time steps. The final time at the end of the
% simulation is dtSODE * numSimTimes
sodeSettings.numSimTimes = 5e4;

% The times at which the simulations will be sampled to produce the
% synthetic data
sodeSettings.sampleTimes = linspace(0, ...
    sodeSettings.dtSODE * sodeSettings.numSimTimes, numExpTimes);
sodeSettings.sampleTimes = [0 0.2 0.8 1.6 5];

% The number of points sampled at each sampling time for each experimental
% condition
numPointsPerSample = numTotalPoints / (numExpTimes * numExpCond);

allX = cell(numExpCond, numExpTimes);
allA = cell(numExpCond, numExpTimes);
allB = cell(numExpCond, numExpTimes);
allT = cell(numExpCond, numExpTimes);
for i = 1:numExpCond

   % The current experimental parameters
    a = allExpParam(i,1);
    b = allExpParam(i,2);
    fprintf('Generating synthetic data for a = %0.2f, b = %0.2f\n:', a, b);

    % The fixed point locations for the current experimental condition
    curFP = allFixedPts{i};
    if ~any(curFP(:,1) < 0)
        curFP = vertcat(allFixedPts{:});
        curFP = curFP(vertcat(allFPLambdas{:}) < 0, :);
        curFP = curFP(curFP(:,1) < 0, :);
        curFP = curFP(curFP(:,1) == max(curFP(:,1)), :);
    end

    % The total number of points generated for the current experimental
    % condition
    curNumPoints = numTotalPoints / numExpCond;

    % The initial point locations are drawn i.i.d. from a Gaussian
    % distribution centered at the left-most stable fixed point with a
    % specified variance
    [~, P0] = min(curFP(:,1));
    P0 = curFP(P0, :);
    X = mvnrnd(P0, sodeSettings.initSigma * eye(2), curNumPoints);

    % Extract the synthetic data for the initial condition
    allX{i,1} = X(1:numPointsPerSample, :);
    allA{i,1} = a .* ones(numPointsPerSample, 1);
    allB{i,1} = b .* ones(numPointsPerSample, 1);
    allT{i,1} = zeros(numPointsPerSample, 1);

    % Remove the sampled tracks
    X(1:numPointsPerSample, :) = [];

    % Run stochastic simulation
    j = 2;
    for tidx = 1:sodeSettings.numSimTimes

        progressbar(tidx, sodeSettings.numSimTimes);
        t = sodeSettings.dtSODE * tidx;

        % Euler-Maruyama step
        gradUX = gradUFunc(a, b, X(:,1), X(:,2));
        X = X - sodeSettings.dtSODE * gradUX + ...
            sqrt(2 * sodeSettings.dtSODE * sodeSettings.DSODE) .* ...
            normrnd(0, 1, [size(X,1) 2]);

        if any(abs(t-sodeSettings.sampleTimes) < 1e-12)

            % Extract the synthetic data for the current time point
            allX{i,j} = X(1:numPointsPerSample, :);
            allA{i,j} = a .* ones(numPointsPerSample, 1);
            allB{i,j} = b .* ones(numPointsPerSample, 1);
            % allT{i,j} = t .* ones(numPointsPerSample, 1);
            allT{i,j} = sodeSettings.sampleTimes(knnsearch( ...
                sodeSettings.sampleTimes.', t)) .* ...
                ones(numPointsPerSample, 1);
            
            
            X(1:numPointsPerSample, :) = []; % Remove the sampled tracks
            j = j+1; % Increment time point counter

        end

    end

    assert(isempty(X), 'Simulation did not run properly');

end

% Assemble synthetic data into a single data table
allX = allX.'; allX = vertcat(allX{:});
allA = allA.'; allA = vertcat(allA{:});
allB = allB.'; allB = vertcat(allB{:});
allT = allT.'; allT = vertcat(allT{:});

dataTable = table(allX, allT, [allA, allB], 'VariableNames', ...
    {'X', 'T', 'Param'});
timePoints = unique(dataTable.T);

clear allX allA allB allT i j X gradUX curFP P0 curNumPoints a b
clear numPointsPerSample t tidx

%--------------------------------------------------------------------------
% View Results
%--------------------------------------------------------------------------

generateVisualization = false;
if generateVisualization

    close all;
    fprintf('\nGenerating Figure Panels:\n')

    xLim = 2.1 * [-1 1];
    yLim = 2.1 * [-1 1];

    figure('Color', 'k');

    for i = 1:numExpCond

        a = allExpParam(i,1);
        b = allExpParam(i,2);
        fpLambda = allFPLambdas{i};

        sinkIDx = fpLambda < 0;
        saddleIDx = fpLambda >= 0;

        for j = 1:numExpTimes

            progressbar(j, numExpTimes);

            t = sodeSettings.sampleTimes(j);
            subplot(numExpCond, numExpTimes, j + (i-1) * numExpTimes)

            curIDx = ismember(dataTable.Param, [a b], 'rows') & ...
                (dataTable.T == t);
            curX = dataTable.X(curIDx, :);
            curT = dataTable.T(curIDx, :);

            tcrange = [min(sodeSettings.sampleTimes), ...
                max(sodeSettings.sampleTimes)];
            timeColors = vals2colormap(curT, 'parula', tcrange);

            hold on

            pcshow([curX, zeros(size(curX,1), 1)], timeColors);

            scatter(allFixedPts{i}(sinkIDx, 1), allFixedPts{i}(sinkIDx, 2), ...
                'filled', 'm', 'MarkerEdgeColor', 'k');
            scatter(allFixedPts{i}(saddleIDx, 1), allFixedPts{i}(saddleIDx, 2), ...
                'filled', 'c', 'MarkerEdgeColor', 'k');

            hold off

            axis equal square
            xlim(xLim);
            ylim(yLim);

            view([0 90]);
            camproj('orthographic');

            cb = colorbar('Color', 'w');
            cb.Ticks = sodeSettings.sampleTimes;
            set(gca, 'Clim', tcrange);
            set(gca, 'Colormap', parula(numel(sodeSettings.sampleTimes)));

            set(gca, 'Color', 'k');
            set(gca, 'YDir', 'normal');
            set(gca, 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');

            title(sprintf('a = %0.2f, b = %0.2f, T = %0.2f', a, b, t), ...
                'Color', 'w');

        end

    end

    sgtitle('Synthetic Data', 'Color', 'w')

end

clear xLim yLim i j cb a b fpLambda sinkIDx saddleIDx curX curT
clear tcrange timeColors t curIDx generateVisualization

%% ************************************************************************
% *************************************************************************
%       PART 2: POINT CLOUD SUBSAMPLING/PSEUDOPOTENTIAL ESTIMATION
% *************************************************************************
% *************************************************************************

%% Subsample Point Cloud ==================================================
close all; clc;

% Set subsampling options
samplingOptions = struct();
samplingOptions.samplingMethod = 'SPARTAN';
samplingOptions.omtIter = 1000;
samplingOptions.verbose = false;

% Set the number of points to retain from each day during subsampling
samplingOptions.numSamplesPerDay = [200 300 300 300 900];

rng(42, 'twister'); % For reproducible random numbers

XDE = [];
for i = 1:numExpCond

    % The current experimental parameters
    a = allExpParam(i,1);
    b = allExpParam(i,2);
    fprintf('Subsampling data for a = %0.2f, b = %0.2f\n:', a, b);

    for tidx = 1:numel(timePoints)

        progressbar(tidx, numel(timePoints));
        t = timePoints(tidx);

        curIDx = ismember(dataTable.Param, [a b], 'rows') & ...
            (dataTable.T == t);
        curX = dataTable.X(curIDx, :);
        assert(size(curX,1) >= samplingOptions.numSamplesPerDay(tidx), ...
            'Not enough points at T = %0.2f for a = %0.2f, b = %0.2f', ...
            t, a, b);

        if strcmpi(samplingOptions.samplingMethod, 'Uniform')

            curXDE = curX(randsample((1:size(curX,1)).', ...
                samplingOptions.numSamplesPerDay(tidx)), :);

        elseif strcmpi(samplingOptions.samplingMethod, 'SPARTAN')

            curXDE = SPARTAN(curX, samplingOptions.numSamplesPerDay(tidx), ...
                'OMTIterations', samplingOptions.omtIter, ...
                'Verbose', samplingOptions.verbose);

        else

            error('Invalid subsampling method');

        end

        XDE = [XDE; curXDE];

    end

end

if any(abs(XDE(:)) >= 2, 'all')
    warning('Some subsampled points lie outside the bounding box');
end

% The number of subsampled points
numPoints = sum(samplingOptions.numSamplesPerDay) * numExpCond ;

assert(size(XDE, 1) == numPoints, ...
    'Subsampling did not produce the correct number of output points');
assert(size(unique(XDE, 'rows'), 1) == size(XDE, 1), ...
    'Subsampling produced duplicate points');

clear a b curX curIDx curXDE i tidx t 

%% Estimate Pseudopotential From Subsampled Point Cloud ===================
% We assume that all of points are drawn from some probability density
% specified by a 'pseudopotential'. This potential is independent of the
% actual dynamics governing how probability jumps between points.
% Functionally, this quantity will set an appropriate 'volume' per point in
% the sums approximating integrals in our transition matrix.
clc;

% This will be the "dynamical" time step parameter which we will hold
% fixed for the rest of the script
dt = 2e-2;

recalculatePseudopotential = true;
if ~exist('U0DE', 'var') || recalculatePseudopotential

    fprintf('Estimating pseudopotential on subsampled point cloud... ');
    U0DE = -log(gaussianKDE(XDE, XDE, [], sqrt(2 * dt), false, [], [], true, true));
    fprintf('Done\n');

end

% Check that the associated transition matrix is irreducible
T = computeTransitionMatrix(XDE, U0DE, dt);
isIrreducible = ~isTransitionMatrixReducible(T);
assert(isIrreducible, ['Point set potential transition matrix is ' ...
    'reducible. Choose a larger time step']);

clear recalculatePseudopotential isIrreducible T

%% Estimate Density Maxima ================================================
close all; clc;

densityOptions = struct();

% Set subsampling options
densityOptions.ssMethod = 'spartan';
densityOptions.numSamples = -1;
densityOptions.omtIter = 5000;
densityOptions.omtWasserThresh = 0;
densityOptions.qrpType = 'sobol';
densityOptions.qrpLeap = 0;
densityOptions.qrpSkip = 0;
densityOptions.qrpScramble = false;

% Set neighborhood graph construction options
densityOptions.nGraphType = 'relaxedgabrielgraph';
densityOptions.kNN = 500;
densityOptions.betaParam = 1;
densityOptions.forceConnectivity = true;
densityOptions.graphOutputFormat = 'symmetric';

% Set kernel density estimation options
densityOptions.kdsigma = sqrt(2*dt);
densityOptions.upsampleEdges = true;
densityOptions.includeNonSamples = true;

% Set topological simplification options
densityOptions.collisionMergeMethod = 'sequential';
densityOptions.persistenceThreshold = 0.05;
densityOptions.sizeThreshold = 400;
densityOptions.stabilityThreshold = 40;
densityOptions.distThreshold = 0.5;
densityOptions.maxThreshold = 0;
densityOptions.plotBranchQuality = true;


% Set default general options ---------------------------------------------
densityOptions.verbose = true;

[pointDensity, XOut, XOutIDx, ENG, maxIDxDE, ...
    saddleIDxDE, EJT, branchIDx, kdsigma] = ...
    pointCloudDensityEstimation(XDE, ...
    'SubsamplingMethod', densityOptions.ssMethod, ...
    'numSamples', densityOptions.numSamples, ...
    'OMTIterations', densityOptions.omtIter, ...
    'omtWasserThresh', densityOptions.omtWasserThresh, ...
    'QRPType', densityOptions.qrpType, ...
    'QRPLeap', densityOptions.qrpLeap, ...
    'QRPSkip', densityOptions.qrpSkip, ...
    'QRPScramble', densityOptions.qrpScramble, ...
    'NeighborhoodGraphType', densityOptions.nGraphType, ...
    'NumNeighbors', densityOptions.kNN, ...
    'GraphBeta', densityOptions.betaParam, ...
    'ForceConnectivity', densityOptions.forceConnectivity, ...
    'GraphOutputFormat', densityOptions.graphOutputFormat, ...
    'KDSigma', densityOptions.kdsigma, ...
    'UpsampleEdges', densityOptions.upsampleEdges, ...
    'IncludeNonSamples', densityOptions.includeNonSamples, ...
    'CollisionMergeMethod', densityOptions.collisionMergeMethod, ...
    'PersistenceThreshold', densityOptions.persistenceThreshold, ...
    'SizeThreshold', densityOptions.sizeThreshold, ...
    'StabilityThreshold', densityOptions.stabilityThreshold, ...
    'DistanceThreshold', densityOptions.distThreshold, ...
    'MaxThreshold', densityOptions.maxThreshold, ...
    'PlotBranchQuality', densityOptions.plotBranchQuality, ...
    'Verbose', densityOptions.verbose );

% Store the output concisely
densityEstimationOutput = struct();
densityEstimationOutput.pointDensity = pointDensity;
densityEstimationOutput.XOut = XOut;
densityEstimationOutput.XOutIDx = XOutIDx;
densityEstimationOutput.ENG = ENG;
densityEstimationOutput.maxIDxDE = maxIDxDE;
densityEstimationOutput.saddleIDxDE = saddleIDxDE;
densityEstimationOutput.EJT = EJT;
densityEstimationOutput.branchIDx = branchIDx;
densityEstimationOutput.kdsigma = kdsigma;

clear pointDensity XOut XOutIDx ENG maxIDxDE saddleIDxDE
clear EJT branchIDx kdsigma rmIDx

fprintf('N = %d maxima detected\n', ...
    numel(densityEstimationOutput.maxIDxDE));

%% Estimate Unstable Manifolds From Pseudopotential =======================
% We calculate the "unstable manifolds" from the pseudopotential for the
% purpose of computing the interpolated potential used in constructing the
% actual dynamical potential

recalculateUnstableManifolds = true;

% Point IDs of local density maxima on downsampled point cloud
locMaxIDx = densityEstimationOutput.maxIDxDE;
locMaxIDx = densityEstimationOutput.XOut(locMaxIDx, :);
locMaxIDx = knnsearch(XDE, locMaxIDx);
% numUniqueMaxPairs = nchoosek(numel(locMaxIDx), 2);

if ~exist('allPaths', 'var') || recalculateUnstableManifolds

    fprintf('Estimating unstable manifolds from pseudopotential... ');

    logTUM = computeLogTransitionMatrix(XDE, U0DE, dt, ...
        'VolumeElementType', 'GraphLaplacian', 'ClipThreshold', 0, ...
        'ScalarMetric', 0.6, 'DiffusionCoefficient', 1.25e-2);
    
    % Estimate Unstable Manifolds Using Transition Matrix -----------------
    % Unstable manifolds are the most probable paths between points in the
    % point cloud calculated using Djikstra's method on graph defined by
    % the transition matrix

    % Unique pairs of density maxima
    maxPairIDx = nchoosek((1:numel(locMaxIDx)).', 2);
    numUniqueMaxPairs = size(maxPairIDx, 1);
    maxPairIDx = [maxPairIDx; fliplr(maxPairIDx)];

    [allPaths, ~, allPathLengths] = ...
        computeMostProbablePaths(XDE, [], locMaxIDx(maxPairIDx), [], ...
        'LogTransitionMatrix', logTUM);
    
    % Check if all paths are symmetric (i.e. the most probable path from
    % i->j is identical to the most probably path from j->i). This is not
    % guaranteed to be the case for a generic transition matrix
    for i = 1:numUniqueMaxPairs
        if ~isequal(allPaths{i}, flipud(allPaths{i+numUniqueMaxPairs}))
            warning('Most probable path between %d and %d is not symmetric', ...
                maxPairIDx(i,1), maxPairIDx(i,2));
        end
    end

    fprintf('Done\n');
    clear gradU0DE2 Ti normT i logTUM
    clear curPath curPathWeights curPathLength
    clear allPathsSymmetric diffGraph maxPairIDx
    
end

clear recalculateUnstableManifolds numUniqueMaxPairs locMaxIDx

%% View Subsampling Results ===============================================
close all; clc;

% The IDs of the density maxima in the subsampled point cloud
if exist('densityEstimationOutput', 'var')
    locMaxIDx = densityEstimationOutput.maxIDxDE;
    locMaxIDx = densityEstimationOutput.XOut(locMaxIDx, :);
    locMaxIDx = knnsearch(XDE, locMaxIDx);
else
    locMaxIDx = [];
end

% The IDs of the paths to view
% pathVisIDx = 1:numel(allPaths);
pathVisIDx = [1 3 5];
% titleString = ' ';

% Choose Point Color Scheme -----------------------------------------------

% View Psuedopotential
cmap = parula(256);
% crange = [min(U0DE) max(U0DE)];
crange = prctile(U0DE, [0 98]);
cellColors = vals2colormap(U0DE, 'parula', crange);
cbString = 'Pseudopotential U_0';

maxColors = repmat([0 1 1], numel(locMaxIDx), 1);

%--------------------------------------------------------------------------
% Generate Figure
%--------------------------------------------------------------------------

figure('Color', 'k');

pcshow([XDE, zeros(numPoints, 1)], cellColors, 'MarkerSize', 15);

hold on

if exist('allPaths', 'var')
    for i = 1:numel(pathVisIDx)
        line( XDE(allPaths{pathVisIDx(i)}, 1), ...
            XDE(allPaths{pathVisIDx(i)}, 2), 'Color', 'm', ...
            'LineWidth', 2, 'Marker', 'o', 'MarkerFaceColor', 'm', ...
            'MarkerEdgeColor', 'none');
    end
end

scatter(XDE(locMaxIDx, 1), XDE(locMaxIDx,2), ...
    100, maxColors, 'filled');

hold off

axis equal square
xlim([-2 2]);
ylim([-2 2]);
view([0 90]);
camproj('orthographic')

cb = colorbar('Color', 'w');
cb.Label.String = cbString;
set(gca, 'Clim', crange);
set(gca, 'Colormap', cmap);
if size(cmap, 1)  == numExpCond
    cb.Ticks = 1:numExpCond;
end

clear i cb pathVisIDx cbString cellColors cmap crange maxColors
clear titleString

%% ************************************************************************
% *************************************************************************
%              PART 3: POTENTIAL DYNAMICS ESTIMATION
% *************************************************************************
% *************************************************************************

% This is a logical place for a checkpoint. If you like, save the results
% of the previous sections and just reload fresh here to avoid having to
% simulate all of those points again.
clear; close all; clc;
load('Point_Set_Manifold_dt2e-2.mat');

%%  Split Paths At Saddle Points ==========================================
% This code consolidates the dense path list down to the minimal set of
% included paths and then splits those paths at 'saddles'. Saddles are
% assumed to be the point along the path with the maximum point set
% pseuodopotential (i.e. minimum density)

fitSaddles = true; % Whether to fit saddle heights as free parameters

incPathIDx = [1 3 5]; % The sparse set of paths we retain to fit
allPaths = allPaths(incPathIDx);
allPathLengths = allPathLengths(incPathIDx);

oldPathEndPoints = cellfun(@(x) [x(1); x(end)], allPaths, 'Uni', false);
oldPathEndPoints = cell2mat(oldPathEndPoints);
oldPathEndPoints = unique(oldPathEndPoints(:));

if fitSaddles

    saddleIDx = cellfun(@(x) 1+find(U0DE(x(2:(end-1))) == ...
        max(U0DE(x(2:(end-1))))), allPaths, 'Uni', false);

    [allPaths, allPathLengths, ~] = ...
        splitPathsAtIDx(allPaths, saddleIDx, XDE, []);

end

fixInPathIDx = cellfun(@(x) [x(1); x(end)], allPaths, 'Uni', false);
fixInPathIDx = cell2mat(fixInPathIDx);
[fixPointIDx, ~, fixInPathIDx] = unique(reshape(fixInPathIDx.', [], 1));
fixInPathIDx = reshape(fixInPathIDx, [2 numel(allPaths)]).';

isSaddle = ~ismember(fixPointIDx, oldPathEndPoints);

clear incPathIDx oldPathEndPoints saddleIDx TUM

%--------------------------------------------------------------------------
% View Results
%--------------------------------------------------------------------------
close all; clc;

% Generate Point Cloud Visualization --------------------------------------

% Color range for visualizing pseudopotential
crange = prctile(U0DE, [0 95]);

% Colors for visualizing distance along paths
pathColors = cell(numel(allPaths), 1);
for i = 1:numel(allPaths)
    if fitSaddles
        cmap = brewermap(256, '*PuRd');
        if (mod(i,2) == 0), cmap = flipud(cmap);end
        pathColors{i} = vals2colormap( ...
            allPathLengths{i} ./ allPathLengths{i}(end), cmap);
    else
        pathColors{i} = repmat([1 0 1], numel(allPaths{i}), 1);
    end
end

fixPtColors = brewermap(2, '*PuRd');
fixPtColors = fixPtColors(isSaddle+1, :);

subplot(1,2,1)
pcshow([XDE, zeros(numPoints, 1)], U0DE)

hold on

for i = 1:numel(allPaths)

    pathEdges = [1:(numel(allPaths{i})-1); 2:numel(allPaths{i})].';
    patch('Faces', pathEdges(:, [1 1 2]), 'Vertices', XDE(allPaths{i}, :), ...
        'FaceVertexCData', pathColors{i}, 'LineWidth', 2, ...
        'FaceColor', 'interp', 'EdgeColor', 'interp');
    scatter(XDE(allPaths{i}, 1), XDE(allPaths{i}, 2), ...
        30, pathColors{i}, 'filled');

end

scatter(XDE(fixPointIDx, 1), XDE(fixPointIDx, 2), ...
    80, fixPtColors, 'filled');

hold off


axis equal square
xlim([-2 2]);
ylim([-2 2])
view([0 90]);
camproj('orthographic');

cb = colorbar('Color', 'w');
cb.Label.String = 'Pseudopotential U_0';
set(gca, 'Clim', crange);

% Generate Potential Plot Visualization -----------------------------------

plotPaths = allPaths;
plotPathLengths = allPathLengths;

plotPaths = vertcat(plotPaths{:});
plotPathLengths = cumsum(vertcat(plotPathLengths{:}));
plotPathLengths = plotPathLengths ./ plotPathLengths(end);

plotMinIDx = ismember(plotPaths, fixPointIDx(~isSaddle));
plotMinIDx = (plotMinIDx & xor(plotMinIDx, circshift(plotMinIDx, [1 0]))) | ...
    [plotMinIDx(1); false(numel(plotMinIDx)-1, 1)];

plotSaddleIDx = ismember(plotPaths, fixPointIDx(isSaddle));
plotSaddleIDx = plotSaddleIDx & xor(plotSaddleIDx, circshift(plotSaddleIDx, [1 0])) | ...
    [plotSaddleIDx(1); false(numel(plotSaddleIDx)-1, 1)];

subplot(1,2,2)
plot(plotPathLengths, U0DE(plotPaths), '-xb', 'LineWidth', 3)

hold on

scatter(plotPathLengths(plotMinIDx), U0DE(plotPaths(plotMinIDx)), ...
    80, [1 0 1], 'filled');
scatter(plotPathLengths(plotSaddleIDx), U0DE(plotPaths(plotSaddleIDx)), ...
    80, [0 1 1], 'filled');

hold off

axis square
xlim([0 1]);
set(gca, 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w', 'Color', 'k');

xlabel('Fraction Length Along Loop', 'Color', 'w');
ylabel('Pseudopotential U_0');

clear plotPaths plotPathLengths plotMinIDx plotSaddleIDx
clear cb cmap crange fixPtColors i pathColors pathEdges

%% Generate Differential Operators on Subsampled Point Cloud ==============
% Differential operators are constructed on a FULLY CONNECTED affinity
% matrix in order to avoid potential complications in the continuum limit
% that can arise for partially connected affinity matrices (i.e. kNN
% graphs)
close all; clc;

% NOTE: The time step for generating the Laplacian/mass matrix should not
% necessarily be equal to the dynamical time step. If the dynamical time
% step is relatively large (i.e. to generate "smoother" dynamics in space),
% you may have to set this time step lower to recover appropriate numerical
% accuracy
dtL = 1e-3;

recalculateDiffOperators = true;
if ( ~exist('LBDE', 'var') || ~exist('MDE', 'var') || recalculateDiffOperators )
    
    fprintf('Constructing diffusion map Laplacian... ')

    [LDE, MDE] = diffusionMapLaplacian(XDE, dtL);

    % Some Tikhonov regularization on the Laplacian to make it
    % negative-definite
    LDE = LDE - 1e-16 * eye(size(LDE));

    fprintf('Done\n');

end

clear recalculateDiffOperators

%% Estimate Measured Probability Densities/Probabilities ==================
% We are using the 'GraphLaplacian' volume element here
close all; clc;

allMeasDensities = cell(numExpCond, 1);
allMeasProb = cell(numExpCond, 1);

volumeElement = exp(U0DE) ./ numPoints;

for i = 1:numExpCond

    % The current experimental parameters
    a = allExpParam(i,1);
    b = allExpParam(i,2);
    fprintf('Extracting measured probability for a = %0.2f, b = %0.2f ...', a, b);

    measDensities = nan(numPoints, numExpTimes);
    % measProb = nan(numPoints, numExpTimes);
    for tidx = 1:numExpTimes

        t = timePoints(tidx);

        % Find points with the current parameter values/times
        curIDx = ismember(dataTable.Param, [a b], 'rows') & ...
            (dataTable.T == t);
        curX = dataTable.X(curIDx, :);

        measDensities(:, tidx) = gaussianKDE(curX, XDE, [], ...
            sqrt(2 * dt), false, [], [], true, true);

    end

    measProb = measDensities .* repmat(volumeElement, 1, numExpTimes);
    measProb = measProb ./ sum(measProb, 1);

    allMeasDensities{i} = measDensities;
    allMeasProb{i} = measProb;

    fprintf('Done\n');

end

clear i a b volumeElement measDensities tidx t curIDx curX measProb

%% ========================================================================
%--------------------------------------------------------------------------
% FIT STATIC LANDSCAPE (LOOP OVER ALL CONDITIONS)
%--------------------------------------------------------------------------
close all; clc;

% Set the base potential
if ~exist('UBDE', 'var'), UBDE = U0DE; end

% The number of time to re-run the fits using different random initial
% conditions
numRandIter = 5;
overwriteFit = false;
overwriteFitOptions = false;

% Set up fit results output directory
fitFileDir = fullfile(projectDir, ['Static_Fits_symKLD_' date]);
if ~exist(fitFileDir, 'dir'), mkdir(fitFileDir); end
fitFileBase = fullfile(fitFileDir, ...
    ['Static_Fit_dt5e-2_a_%0.2f_b_%0.2f_RI%d_' date '.mat']);

% Set Options -------------------------------------------------------------
fitOptions = struct();
fitOptions.initialConditions = {};
fitOptions.numSimTimes = 250;
fitOptions.enforceSaddles = false;
fitOptions.constHeightSum = 0;
fitOptions.simTimeHandling = 'none';
fitOptions.constDiffusionCoefficient = [];
fitOptions.constScalarMetric = [];
fitOptions.enforcePositiveDiffusion = true;
fitOptions.enforcePositiveMetric = true;
fitOptions.precomputeQuadProg = true;
fitOptions.volumeElementType = 'GraphLaplacian';
fitOptions.upperBounds = [inf(numel(isSaddle), 1); 5; 5];
fitOptions.useGPU = true;
fitOptions.verbose = true;
fitOptions.errorType = 'symKLD';
fitOptions.constFixedHeights = [];
fitOptions.pointDiffusionCoefficient = 1;
fitOptions.dtL = dtL;

optOptions = {'Display', 'iter', 'UseParallel', false, ...
    'FiniteDifferenceType', 'forward', ...
    'PlotFcn', {'optimplotx', 'optimplotfval'}, ...
    'MaxIterations', 1000, 'MaxFunctionEvaluations', 15000};

fitOptionsFile = fullfile(fitFileDir, 'fitOptions.mat');
if ~exist(fitOptionsFile, 'file') || overwriteFitOptions
    save(fitOptionsFile, 'fitOptions', 'optOptions');
end

%--------------------------------------------------------------------------
% Fit Data
%--------------------------------------------------------------------------

rng(100, 'twister'); % For reproducible random numbers
for i = 1:numExpCond

    a = allExpParam(i,1);
    b = allExpParam(i,2);

    for randIter = 1:numRandIter

        % Set initial guess (we want to do this before skipping already
        % completed fits in order to have the reproducible random numbers
        % always match
        initGuess = 10 * rand(numel(isSaddle), 1) - 5;
        initGuess = [ (initGuess-sum(initGuess)/numel(initGuess)); 1; 1];

        % Check if fit has already been completed
        fitFile = sprintf(fitFileBase, a, b, randIter);
        if (exist(fitFile, 'file') && ~overwriteFit), continue; end

        % Consolidate data for current condition
        dataProb = allMeasProb(i);
        dataTimes = {timePoints};

        % Run optimization
        [optErr, fixHeights, D, scalarMetric, ...
            timeScale, fitTimes, optOutput] = ...
            fitStaticLandscape(XDE, dataProb, dataTimes, dt, allPaths, ...
            'InitialGuess', initGuess, ...
            'InitialConditions', fitOptions.initialConditions, ...
            'NumSimTimes', fitOptions.numSimTimes, ...
            'IsSaddle', isSaddle, ...
            'PointDiffusionCoefficient', fitOptions.pointDiffusionCoefficient, ...
            'EnforceSaddles', fitOptions.enforceSaddles, ...
            'ConstHeightSum', fitOptions.constHeightSum, ...
            'SimTimeHandling', fitOptions.simTimeHandling, ...
            'OptimizationOptions', optOptions, ...
            'ConstDiffusionCoefficient', fitOptions.constDiffusionCoefficient, ...
            'ConstFixedHeights', fitOptions.constFixedHeights, ...
            'ConstScalarMetric', fitOptions.constScalarMetric, ...
            'EnforcePositiveDiffusion', fitOptions.enforcePositiveDiffusion, ...
            'EnforcePositiveMetric', fitOptions.enforcePositiveMetric, ...
            'PrecomputeQuadProg', fitOptions.precomputeQuadProg, ...
            'PointPotential', U0DE, ...
            'BasePotential', UBDE, ...
            'Laplacian', LDE, ...
            'MassMatrix', MDE, ...
            'PathLengths', allPathLengths, ...
            'VolumeElementType', fitOptions.volumeElementType, ...
            'UpperBounds', fitOptions.upperBounds, ...
            'UseGPU', fitOptions.useGPU, ...
            'Verbose', fitOptions.verbose, ...
            'ErrorType', fitOptions.errorType);

        save(fitFile, 'fitTimes', 'fixHeights', 'optErr', ...
            'optOutput', 'D', 'scalarMetric', 'timeScale', 'initGuess');

        close all; clc;

    end
end

clear i overwriteFit overwriteFitOptions numRandIter fitFileDir
clear fitFileBase fitOptionsFile a b randIter dataProb dataTimes
clear initGuess optOptions

%% Consolidate Fit Results and Simulate Data ==============================
close all; clc;

% Uncomment to start from a fresh workspace (save right before running the
% "Fit Static Landscape" section)
% clear; load('Point_Set_Manifold_PreFit_dt2e-2_dtL1e-3.mat');

% Set the base potential
if ~exist('UBDE', 'var'), UBDE = U0DE; end

% Set up fit results output directory
fitFileDir = fullfile(projectDir, ['Static_Fits_symKLD_' date]);
if ~exist(fitFileDir, 'dir'), mkdir(fitFileDir); end
fitFileBase = fullfile(fitFileDir, ...
    ['Static_Fit_dt5e-2_a_%0.2f_b_%0.2f_RI%d_' date '.mat']);

% Load fit options
load(fullfile(fitFileDir, 'fitOptions.mat'), 'fitOptions');

if ~exist('numRandIter', 'var'), numRandIter = 5; end

allSimDensities = cell(numExpCond, 1);
allSimProb = cell(numExpCond, 1);
allSimGroundStateDensities = cell(numExpCond, 1);
allSimGroundStateProb = cell(numExpCond, 1);
allAnaDensities = cell(numExpCond, 1);
allAnaProb = cell(numExpCond, 1);
allAnaGroundStateDensities = cell(numExpCond, 1);
allAnaGroundStateProb = cell(numExpCond, 1);
allFitTimes = cell(numExpCond, 1);
allAnaFitTimes = cell(numExpCond, 1);
allFixHeights = cell(numExpCond, 1);
allFitFiles = cell(numExpCond, 1);
allD = cell(numExpCond, 1);
allScalarMetrics = cell(numExpCond, 1);
allUIDE = cell(numExpCond, 1);
allUDE = cell(numExpCond, 1);
for i = 1:numExpCond

    a = allExpParam(i,1);
    b = allExpParam(i,2);

    %----------------------------------------------------------------------
    % Extract Best Result for Current Experimental Parameters
    %----------------------------------------------------------------------

    fprintf('Extracting best results for a = %0.2f and b = %0.2f... ', a, b);

    optErr = Inf;
    fitFile = '';
    for randIter = 1:numRandIter

        curFitFile = sprintf(fitFileBase, a, b, randIter);
        curOptErr = load(curFitFile, 'optErr');
        curOptErr = curOptErr.optErr;

        if (curOptErr < optErr)
            optErr = curOptErr;
            fitFile = curFitFile;
        end

    end

    % Load best results
    load(fitFile, 'fitTimes', 'fixHeights', 'scalarMetric', 'D');
    fitTimes = fitTimes{1};
    
    clear randIter curFitFile curOptErr

    fprintf('Done\n');

    %----------------------------------------------------------------------
    % Compute Interpolated Potential
    %----------------------------------------------------------------------

    endPointVals = fixHeights(fixInPathIDx);

    [knownU, knownIDx] = interpolateValuesAlongPath(endPointVals, ...
        allPaths, 'PathLengths', allPathLengths);

    UIDE = interpolatePotentialKHarmonic(XDE, knownU, knownIDx, ...
            2, 'Laplacian', LDE, 'MassMatrix', MDE, 'Verbose', true);

    % UIDE = interpolatePotentialKHarmonic(XDE, knownU, knownIDx, ...
    %         2, 'Laplacian', [], 'MassMatrix', [], ...
    %         'Verbose', true, 'TimeStep', dtL);

    % Compute the dynamical potential
    UDE = UBDE + UIDE;

    clear endPointVals knownU knownIDx

    %----------------------------------------------------------------------
    % Simulate Potential Dynamics (Static Landscape)
    %----------------------------------------------------------------------

    % Initial probability is taken directly from synthetic data
    initProb = allMeasProb{i}(:,1);

    fprintf('Generating dynamical transition matrix... ');

    [T, volumeElement] = computeTransitionMatrix(XDE, UDE, dt, ...
        'PointPotential', U0DE, 'ClipThreshold', 1e-14, ...
        'ScalarMetric', scalarMetric, 'DiffusionCoefficient', D, ...
        'VolumeElementType', fitOptions.volumeElementType);

    fprintf('Done\n');

    fprintf('Time evolving probabilities... ');

    % The 'viewTimes' field is the same for all experimental conditions
    [simProb, viewTimes] = evolveProbabilities(initProb, T, ...
        fitOptions.numSimTimes-1, 'NumViewTimes', -1, 'TimeStep', dt);

    % volumeElement = exp(U0DE);
    simDensity = simProb ./ volumeElement;

    fprintf('Done\n');

    fprintf('Computing ground state probabilities... ')

    % How is this not properly normalized? Tired of the floating point
    % mismatch warnings...
    uniformProb = ones(numPoints, 1) ./ numPoints;
    uniformProb = uniformProb ./ sum(uniformProb(:));

    simGroundStateProb = evolveProbabilities( ...
        uniformProb, T, 1e4, 'NumViewTimes', 1, ...
        'TimeStep', dt);

    simGroundStateDensity = simGroundStateProb ./ volumeElement;

    fprintf('Done\n')

    clear T % volumeElement initProb unifomProb

    %----------------------------------------------------------------------
    % Simulate Potential Dynamics for Analytic Potential
    %----------------------------------------------------------------------

    fprintf('Generating analytic transition matrix... ');

    T = computeTransitionMatrix(XDE, UFunc(a, b, XDE(:,1), XDE(:,2)), ...
        dt, 'DiffusionCoefficient', sodeSettings.DSODE, ...
        'PointPotential', U0DE, 'ClipThreshold', 1e-14, ...
        'ScalarMetric', 1, 'VolumeElementType', 'GraphLaplacian');

    fprintf('Done\n');

    fprintf('Time evolving analytic probabilities... ');

    % The 'viewTimes' field is the same for all experimental conditions
    [anaProb, ~] = evolveProbabilities(initProb, T, ...
        fitOptions.numSimTimes-1, 'NumViewTimes', -1, 'TimeStep', dt);

    volumeElement = exp(U0DE);
    anaDensity = anaProb ./ volumeElement;

    fprintf('Done\n');

    fprintf('Finding analytic fit times... ')

    anaFitTimes = nan(size(fitTimes));
    for j = 1:size(allMeasProb{i}, 2)

        % Symmetric KL-divergence error
        err = (allMeasProb{i}(:,j) - anaProb) ...
            .* log(allMeasProb{i}(:,j) ./ anaProb);
        err(isnan(err)) = 0;
        err = sum(err, 1);

        [~, anaFitTimes(j)] = min(err);
        anaFitTimes(j) = viewTimes(anaFitTimes(j));

    end

    fprintf('Done\n');

    fprintf('Computing analytic ground state probabilities... ')

    anaGroundStateProb = evolveProbabilities( ...
        uniformProb, T, 1e4, 'NumViewTimes', 1, ...
        'TimeStep', dt);

    anaGroundStateDensity = anaGroundStateProb ./ volumeElement;

    fprintf('Done\n')

    clear T volumeElement initProb uniformProb
    clear j err

    %----------------------------------------------------------------------
    % Store Results
    %----------------------------------------------------------------------

    allFitFiles{i} = fitFile;
    allFitTimes{i} = fitTimes;
    allFixHeights{i} = fixHeights;
    allD{i} = D;
    allScalarMetrics{i} = scalarMetric;
    allUIDE{i} = UIDE;
    allUDE{i} = UDE;

    allSimDensities{i} = simDensity;
    allSimProb{i} = simProb;
    allSimGroundStateProb{i} = simGroundStateProb;
    allSimGroundStateDensities{i} = simGroundStateDensity;

    allAnaDensities{i} = anaDensity;
    allAnaProb{i} = anaProb;
    allAnaFitTimes{i} = anaFitTimes;
    allAnaGroundStateProb{i} = anaGroundStateProb;
    allAnaGroundStateDensities{i} = anaGroundStateDensity;

    clear fitFile fitTimes fixHeights scalarMetric UIDE UDE
    clear simDensity simProb anaDensity anaProb anaFitTimes
    clear simGroundStateProb simGroundStateDensity
    clear anaGroundStateProb anaGroundStateDensity

    disp(' ');

end

allSimResults = struct('fitFile', allFitFiles, 'fitTimes', allFitTimes, ...
    'fixHeights', allFixHeights, 'scalarMetric', allScalarMetrics, ...
    'UIDE', allUIDE,'UDE', allUDE, 'simDensity', allSimDensities, ...
    'simProb', allSimProb, 'anaDensity', allAnaDensities, ...
    'anaProb', allAnaProb, 'anaFitTimes', allAnaFitTimes, ...
    'simGroundStateProb', allSimGroundStateProb, ...
    'anaGroundStateProb', allAnaGroundStateProb, ...
    'simGroundStateDensity', allSimGroundStateDensities, ...
    'anaGroundStateDensity', allAnaGroundStateDensities );

clear i allFitFiles allFitTimes allFixHeights allScalarMetrics
clear allUDE allUIDE fitFileDir fitFileBase numRandIter
clear allSimDensities allSimProb a b optErr allAnaDensities
clear allAnaFitTimes allAnaProb allAnaGroundStateDensities
clear allSimGroundStateDensities allAnaGroundStateProb
clear allSimGroundStateProb


%% ************************************************************************
% *************************************************************************
%                           PART 4: VIEW RESULTS
% *************************************************************************
% *************************************************************************

%% View Measured Probabilities/Densities For All Conditions ===============
clc; % close all;

plotType = 'Density';
if ~ismember(plotType, {'Density', 'Probability'})
    error('Invalid plot type');
end

if ~exist('locMaxIDx', 'var')
    if exist('densityEstimationOutput', 'var')
        locMaxIDx = densityEstimationOutput.maxIDxDE;
        locMaxIDx = densityEstimationOutput.XOut(locMaxIDx, :);
        locMaxIDx = knnsearch(XDE, locMaxIDx);
    else
        locMaxIDx = [];
    end
end

disp(['Generating Measured ' plotType ' Figure Panels:']);

xLim = 2 * [-1 1];
yLim = 2 * [-1 1];

make3DPlot = false;

figure('Color', 'k');

for i = 1:numExpCond

    a = allExpParam(i,1);
    b = allExpParam(i,2);
    fpLambda = allFPLambdas{i};

    sinkIDx = fpLambda < 0;
    saddleIDx = fpLambda >= 0;

    for tidx = 1:numExpTimes

        progressbar(tidx, numExpTimes);
        t = timePoints(tidx);

        if strcmpi(plotType, 'Probability')
            plotProb = allMeasProb{i}(:, tidx);
        elseif strcmpi(plotType, 'Density')
            plotProb = allMeasDensities{i}(:, tidx);
        else
            error('Invalid plot type');
        end

        % if tidx == (numExpTimes-1)
        %     plotProb = exp(-UFunc(a, b, XDE(:,1), XDE(:,2)));
        %     plotProb = plotProb ./ sum(plotProb) .* sum(allMeasDensities{i}(:, end));
        % end

        % crange = prctile(plotProb, [0 98]);
        crange = [0, prctile(plotProb, 98)];

        subplot(numExpCond, numExpTimes, tidx + (i-1) * numExpTimes)

        hold on

        if make3DPlot

            pcshow([XDE, plotProb], plotProb);

            scatter3(allFixedPts{i}(sinkIDx, 1), allFixedPts{i}(sinkIDx, 2), ...
                plotProb(knnsearch(XDE, [allFixedPts{i}(sinkIDx, 1), ...
                allFixedPts{i}(sinkIDx, 2)])), ...
                'filled', 'm', 'MarkerEdgeColor', 'k');
            scatter3(allFixedPts{i}(saddleIDx, 1), allFixedPts{i}(saddleIDx, 2),  ...
                plotProb(knnsearch(XDE, [allFixedPts{i}(saddleIDx, 1), ...
                allFixedPts{i}(saddleIDx, 2)])), ...
                'filled', 'c', 'MarkerEdgeColor', 'k');

        else

            pcshow([XDE, zeros(numPoints, 1)], plotProb);

            scatter(allFixedPts{i}(sinkIDx, 1), allFixedPts{i}(sinkIDx, 2), ...
                'filled', 'm', 'MarkerEdgeColor', 'k');
            scatter(allFixedPts{i}(saddleIDx, 1), allFixedPts{i}(saddleIDx, 2), ...
                'filled', 'c', 'MarkerEdgeColor', 'k');

        end

        hold off

        axis equal square
        xlim(xLim);
        ylim(yLim);

        view([0 90]);
        camproj('orthographic');

        cb = colorbar('Color', 'w');
        set(gca, 'Clim', crange);
        set(gca, 'Colormap', brewermap(256, '*YlGnBu'));

        set(gca, 'Color', 'k');
            set(gca, 'YDir', 'normal');
            set(gca, 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');

            title(sprintf('a = %0.2f, b = %0.2f, T = %0.2f', a, b, t), ...
                'Color', 'w');

    end

end

sgtitle(['''Measured'' ', plotType], 'Color', 'w');

clear xLim yLim t tidx sinkIDx saddleIDx plotType plotProb
clear i fpLambda crange cb  a b make3DPlot

%% View Simulated Probabilities/Densities For All Conditions ==============
clc; % close all;

plotType = 'Density';
if ~ismember(plotType, {'Density', 'Probability'})
    error('Invalid plot type');
end

if ~exist('locMaxIDx', 'var')
    if exist('densityEstimationOutput', 'var')
        locMaxIDx = densityEstimationOutput.maxIDxDE;
        locMaxIDx = densityEstimationOutput.XOut(locMaxIDx, :);
        locMaxIDx = knnsearch(XDE, locMaxIDx);
    else
        locMaxIDx = [];
    end
end

disp(['Generating Simulated ' plotType ' Figure Panels:']);

xLim = 2 * [-1 1];
yLim = 2 * [-1 1];

figure('Color', 'k');

for i = 1:numExpCond

    a = allExpParam(i,1);
    b = allExpParam(i,2);
    fpLambda = allFPLambdas{i};

    sinkIDx = fpLambda < 0;
    saddleIDx = fpLambda >= 0;

    for tidx = 1:numExpTimes

        progressbar(tidx, numExpTimes);
        if (tidx == 1)
            t = 0;
        else
            t = allSimResults(i).fitTimes(tidx-1);
        end

        if strcmpi(plotType, 'Probability')
            plotProb = allSimResults(i).simProb(:, knnsearch(viewTimes.', t));
        elseif strcmpi(plotType, 'Density')
            plotProb = allSimResults(i).simDensity(:, knnsearch(viewTimes.', t));
            plotProb = plotProb ./ sum(plotProb) .* sum(allMeasDensities{i}(:, tidx));
        else
            error('Invalid plot type');
        end

        % crange = prctile(plotProb, [0 98]);
        crange = [0, prctile(plotProb, 98)];

        subplot(numExpCond, numExpTimes, tidx + (i-1) * numExpTimes)

        hold on

        pcshow([XDE, zeros(numPoints, 1)], plotProb);

        scatter(allFixedPts{i}(sinkIDx, 1), allFixedPts{i}(sinkIDx, 2), ...
            'filled', 'm', 'MarkerEdgeColor', 'k');
        scatter(allFixedPts{i}(saddleIDx, 1), allFixedPts{i}(saddleIDx, 2), ...
            'filled', 'c', 'MarkerEdgeColor', 'k');

        hold off

        axis equal square
        xlim(xLim);
        ylim(yLim);

        view([0 90]);
        camproj('orthographic');

        cb = colorbar('Color', 'w');
        set(gca, 'Clim', crange);
        set(gca, 'Colormap', brewermap(256, '*YlGnBu'));

        set(gca, 'Color', 'k');
        set(gca, 'YDir', 'normal');
        set(gca, 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');

        title(sprintf('a = %0.2f, b = %0.2f, T = %0.2f', a, b, t), ...
            'Color', 'w');

    end

end

sgtitle(['Simulated ', plotType], 'Color', 'w');

clear xLim yLim t tidx sinkIDx saddleIDx plotType plotProb
clear i fpLambda crange cb  a b

%% View Analytic Potential Probabilities/Densities For All Conditions =====
clc; % close all;

plotType = 'Density';
if ~ismember(plotType, {'Density', 'Probability'})
    error('Invalid plot type');
end

if ~exist('locMaxIDx', 'var')
    if exist('densityEstimationOutput', 'var')
        locMaxIDx = densityEstimationOutput.maxIDxDE;
        locMaxIDx = densityEstimationOutput.XOut(locMaxIDx, :);
        locMaxIDx = knnsearch(XDE, locMaxIDx);
    else
        locMaxIDx = [];
    end
end

disp(['Generating Analytic ' plotType ' Figure Panels:']);

xLim = 2 * [-1 1];
yLim = 2 * [-1 1];

figure('Color', 'k');

for i = 1:numExpCond

    a = allExpParam(i,1);
    b = allExpParam(i,2);
    fpLambda = allFPLambdas{i};

    sinkIDx = fpLambda < 0;
    saddleIDx = fpLambda >= 0;

    for tidx = 1:numExpTimes

        progressbar(tidx, numExpTimes);
        if (tidx == 1)
            t = 0;
        else
            t = allSimResults(i).anaFitTimes(tidx);
        end

        if strcmpi(plotType, 'Probability')
            plotProb = allSimResults(i).anaProb(:, knnsearch(viewTimes.', t));
        elseif strcmpi(plotType, 'Density')
            plotProb = allSimResults(i).anaDensity(:, knnsearch(viewTimes.', t));
            plotProb = plotProb ./ sum(plotProb) .* sum(allMeasDensities{i}(:, tidx));
        else
            error('Invalid plot type');
        end

        % crange = prctile(plotProb, [0 98]);
        crange = [0, prctile(plotProb, 98)];

        subplot(numExpCond, numExpTimes, tidx + (i-1) * numExpTimes)

        hold on

        pcshow([XDE, zeros(numPoints, 1)], plotProb);

        scatter(allFixedPts{i}(sinkIDx, 1), allFixedPts{i}(sinkIDx, 2), ...
            'filled', 'm', 'MarkerEdgeColor', 'k');
        scatter(allFixedPts{i}(saddleIDx, 1), allFixedPts{i}(saddleIDx, 2), ...
            'filled', 'c', 'MarkerEdgeColor', 'k');

        hold off

        axis equal square
        xlim(xLim);
        ylim(yLim);

        view([0 90]);
        camproj('orthographic');

        cb = colorbar('Color', 'w');
        set(gca, 'Clim', crange);
        set(gca, 'Colormap', brewermap(256, '*YlGnBu'));

        set(gca, 'Color', 'k');
        set(gca, 'YDir', 'normal');
        set(gca, 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');

        title(sprintf('a = %0.2f, b = %0.2f, T = %0.2f', a, b, t), ...
            'Color', 'w');

    end

end

sgtitle(['Analytic ', plotType], 'Color', 'w');

clear xLim yLim t tidx sinkIDx saddleIDx plotType plotProb
clear i fpLambda crange cb  a b


%% View Base/Interpolated/Dynamical Potential Fields (Voronoi) ============
close all; clc;

paramID = 3;
assert(ismember(paramID, [1 2 3]), 'Invalid parameter set');

a = allExpParam(paramID, 1);
b = allExpParam(paramID, 2);
fpLambda = allFPLambdas{paramID};

sinkIDx = fpLambda < 0;
saddleIDx = fpLambda >= 0;

xLim = 2.05 * [-1 1];
yLim = 2.05 * [-1 1];

%--------------------------------------------------------------------------
% Generate Clipped Voronoi Diagram of Point Set
%--------------------------------------------------------------------------

bdyPoly = [xLim(1) yLim(1); xLim(2) yLim(1); xLim(2) yLim(2); xLim(1) yLim(2)];
DT = delaunayTriangulation(XDE);
[v, c] = clippedVoronoiDiagram(DT, bdyPoly);
% cv = cellfun(@(x) v([x, x(1)], :), c, 'Uni', false);
% careas = cellfun(@(x) area(polyshape(x(:,1), x(:,2))), cv, 'Uni', true);

f = max(cellfun(@numel, c, 'Uni', true));
f = cell2mat(cellfun(@(x) [x, nan(1, f-numel(x)+1)], c, 'Uni', false));

%--------------------------------------------------------------------------
% Generate Figure Panels
%--------------------------------------------------------------------------

cmap = brewermap(256, '*YlGnBu');

crange_ub = prctile(UBDE, [0 95]);
crange_ui = prctile(allSimResults(paramID).UIDE, [0 95]);
crange_ud = prctile(allSimResults(paramID).UDE, [0 95]); 

figure('Color', 'k')

% Base Potential ----------------------------------------------------------
subplot(1,3,1)

patch('Faces', f, 'Vertices', v, 'EdgeColor', 'k', ...
    'FaceVertexCData', UBDE, ...
    'FaceColor', 'flat', 'LineWidth', 1, 'EdgeColor', 'none');

hold on

scatter(allFixedPts{paramID}(sinkIDx, 1), allFixedPts{paramID}(sinkIDx, 2), ...
    'filled', 'm', 'MarkerEdgeColor', 'k');
scatter(allFixedPts{paramID}(saddleIDx, 1), allFixedPts{paramID}(saddleIDx, 2), ...
    'filled', 'c', 'MarkerEdgeColor', 'k');

hold off

axis equal square
xlim(xLim);
ylim(yLim);

cb_ub = colorbar('Color', 'w');
cb_ub.Label.String = 'U_{base}';
set(gca, 'Clim', crange_ub);
set(gca, 'Colormap', cmap);
set(gca, 'Color', 'k');
set(gca, 'YDir', 'normal');
set(gca, 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');

title(sprintf('Base Potential\na = %0.2f, b = %0.2f', ...
    a, b), 'Color', 'w');

% Interpolated Potential --------------------------------------------------
subplot(1,3,2)

patch('Faces', f, 'Vertices', v, 'EdgeColor', 'k', ...
    'FaceVertexCData', allSimResults(paramID).UIDE, ...
    'FaceColor', 'flat', 'LineWidth', 1, 'EdgeColor', 'none');

hold on

scatter(allFixedPts{paramID}(sinkIDx, 1), allFixedPts{paramID}(sinkIDx, 2), ...
    'filled', 'm', 'MarkerEdgeColor', 'k');
scatter(allFixedPts{paramID}(saddleIDx, 1), allFixedPts{paramID}(saddleIDx, 2), ...
    'filled', 'c', 'MarkerEdgeColor', 'k');

hold off

axis equal square
xlim(xLim);
ylim(yLim);

cb_ui = colorbar('Color', 'w');
cb_ui.Label.String = 'U_{int}';
set(gca, 'Clim', crange_ui);
set(gca, 'Colormap', cmap);

set(gca, 'Color', 'k');
set(gca, 'YDir', 'normal');
set(gca, 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');

title(sprintf('Interpolated Potential\na = %0.2f, b = %0.2f', ...
    a, b), 'Color', 'w');

% Dynamical Potential -----------------------------------------------------
subplot(1,3,3)

patch('Faces', f, 'Vertices', v, 'EdgeColor', 'k', ...
    'FaceVertexCData', allSimResults(paramID).UDE, ...
    'FaceColor', 'flat', 'LineWidth', 1, 'EdgeColor', 'none');

hold on

scatter(allFixedPts{paramID}(sinkIDx, 1), allFixedPts{paramID}(sinkIDx, 2), ...
    'filled', 'm', 'MarkerEdgeColor', 'k');
scatter(allFixedPts{paramID}(saddleIDx, 1), allFixedPts{paramID}(saddleIDx, 2), ...
    'filled', 'c', 'MarkerEdgeColor', 'k');

hold off

axis equal square
xlim(xLim);
ylim(yLim);

cb_ud = colorbar('Color', 'w');
cb_ud.Label.String = 'U';
set(gca, 'Clim', crange_ud);
set(gca, 'Colormap', cmap);

set(gca, 'Color', 'k');
set(gca, 'YDir', 'normal');
set(gca, 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');

title(sprintf('Dynamical Potential\na = %0.2f, b = %0.2f', ...
    a, b), 'Color', 'w');


clear a b fpLambda sinkIDx saddleIDx xLim yLim bdyPoly DT v c cv careas
clear f cmap crange_ub crange_ui crange_ud cb_ub cb_ui cb_ud paramID


%% View Base/Interpolated/Dynamical Potential Fields (Point Cloud) ========
close all; clc;

paramID = 3;
assert(ismember(paramID, [1 2 3]), 'Invalid parameter set');

includeScalarMetric = true;
curUDE = allSimResults(paramID).UDE;
if includeScalarMetric
    curUDE = curUDE ./ allSimResults(paramID).scalarMetric;
end

a = allExpParam(paramID, 1);
b = allExpParam(paramID, 2);

curUTrue = UFunc(a, b, XDE(:,1), XDE(:,2));

xLim = 2 * [-1 1];
yLim = 2 * [-1 1];

%--------------------------------------------------------------------------
% Generate Figure Panels
%--------------------------------------------------------------------------

allPathIDx = unique(vertcat(allPaths{:}));

cmap = brewermap(256, '*YlGnBu');

crange_ub = prctile(UBDE, [0 95]);
crange_ui = prctile(allSimResults(paramID).UIDE, [0 95]);
crange_ud = prctile(curUDE, [0 95]);
crange_ut = prctile(curUTrue, [0 95]);

UBColors = vals2colormap(UBDE, cmap, crange_ub);
UIColors = vals2colormap(allSimResults(paramID).UIDE, cmap, crange_ui);
UDColors = vals2colormap(curUDE, cmap, crange_ud);
UTColors = vals2colormap(curUTrue, cmap, crange_ut);

figure('Color', 'k')

% Base Potential ----------------------------------------------------------
subplot(2,2,1)

pcshow([XDE, UBDE], UBDE);

hold on

scatter3(XDE(allPathIDx, 1), XDE(allPathIDx, 2), UBDE(allPathIDx), ...
    50, UBColors(allPathIDx, :), 'filled')

scatter3(XDE(fixPointIDx, 1), XDE(fixPointIDx, 2), UBDE(fixPointIDx), ...
    100, UBColors(fixPointIDx, :), 'filled', 'MarkerEdgeColor', 'm')

hold off

% axis equal % square
xlim(xLim);
ylim(yLim);
camproj('orthographic')

cb_ub = colorbar('Color', 'w');
cb_ub.Label.String = 'U_{base}';
set(gca, 'Clim', crange_ub);
set(gca, 'Colormap', cmap);
set(gca, 'Color', 'k');
set(gca, 'YDir', 'normal');
set(gca, 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');

title(sprintf('Base Potential\na = %0.2f, b = %0.2f', ...
    a, b), 'Color', 'w');

% Interpolated Potential --------------------------------------------------
subplot(2,2,2)

pcshow([XDE, allSimResults(paramID).UIDE], allSimResults(paramID).UIDE);

hold on

scatter3(XDE(allPathIDx, 1), XDE(allPathIDx, 2), ...
    allSimResults(paramID).UIDE(allPathIDx), ...
    50, UIColors(allPathIDx, :), 'filled')

scatter3(XDE(fixPointIDx, 1), XDE(fixPointIDx, 2), ...
    allSimResults(paramID).UIDE(fixPointIDx), ...
    100, UIColors(fixPointIDx, :), 'filled', 'MarkerEdgeColor', 'm')

hold off

% axis equal % square
xlim(xLim);
ylim(yLim);
camproj('orthographic')

cb_ui = colorbar('Color', 'w');
cb_ui.Label.String = 'U_{int}';
set(gca, 'Clim', crange_ui);
set(gca, 'Colormap', cmap);

set(gca, 'Color', 'k');
set(gca, 'YDir', 'normal');
set(gca, 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');

title(sprintf('Interpolated Potential\na = %0.2f, b = %0.2f', ...
    a, b), 'Color', 'w');

% Dynamical Potential -----------------------------------------------------
subplot(2,2,3)

pcshow([XDE, curUDE], curUDE);

hold on

scatter3(XDE(allPathIDx, 1), XDE(allPathIDx, 2), ...
    curUDE(allPathIDx), 50, UDColors(allPathIDx, :), 'filled')

scatter3(XDE(fixPointIDx, 1), XDE(fixPointIDx, 2), curUDE(fixPointIDx), ...
    100, UDColors(fixPointIDx, :), 'filled', 'MarkerEdgeColor', 'm')

hold off

% axis equal % square
xlim(xLim);
ylim(yLim);
camproj('orthographic')

cb_ud = colorbar('Color', 'w');
if includeScalarMetric
    cb_ud.Label.String = 'U / g';
else
    cb_ud.Label.String = 'U';
end
set(gca, 'Clim', crange_ud);
set(gca, 'Colormap', cmap);

set(gca, 'Color', 'k');
set(gca, 'YDir', 'normal');
set(gca, 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');

title(sprintf('Dynamical Potential\na = %0.2f, b = %0.2f', ...
    a, b), 'Color', 'w');

% "True" Potential ----------------------------------------------===-------
subplot(2,2,4)

pcshow([XDE, curUTrue], curUTrue);

hold on

scatter3(XDE(allPathIDx, 1), XDE(allPathIDx, 2), ...
    curUTrue(allPathIDx), 50, UTColors(allPathIDx, :), 'filled')

scatter3(XDE(fixPointIDx, 1), XDE(fixPointIDx, 2), curUTrue(fixPointIDx), ...
    100, UTColors(fixPointIDx, :), 'filled', 'MarkerEdgeColor', 'm')

hold off

% axis equal % square
xlim(xLim);
ylim(yLim);
camproj('orthographic')

cb_ut = colorbar('Color', 'w');
cb_ut.Label.String = 'U';
set(gca, 'Clim', crange_ut);
set(gca, 'Colormap', cmap);

set(gca, 'Color', 'k');
set(gca, 'YDir', 'normal');
set(gca, 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');

title(sprintf('''True'' Potential\na = %0.2f, b = %0.2f', ...
    a, b), 'Color', 'w');

clear a b fpLambda sinkIDx saddleIDx xLim yLim bdyPoly DT v c cv careas
clear f cmap crange_ub crange_ui crange_ud cb_ub cb_ui cb_ud cb_ut
clear allPathIDx  UBColors UIColors UDColors curUDE
clear includeScalarMetric curUTrue crange_ut UTColors

%% View Error For All Conditions ==========================================
close all; clc;

errorType = 'symKLD';

allH = zeros(numExpCond, numExpTimes);

cmap = parula(numel(timePoints));
crange = [1 numel(timePoints)];
cticks = 1:numel(timePoints);
cticklabels = cellfun(@num2str, num2cell(timePoints), 'Uni', false);
cbString = 'Measured Time';

yLim = [0 0.125];
xLim = [viewTimes(1) viewTimes(end)];

figure('Color', 'w');

for i = 1:numExpCond

    a = allExpParam(i,1);
    b = allExpParam(i,2);

    fprintf('Processing error for a = %0.2f, b = %0.2f:\n', a, b);

    for tidx = 1:numExpTimes

        % Current measured probability
        measProb = allMeasProb{i}(:, tidx);
        allH(i, tidx) = -sum(measProb .* log(measProb));
        fprintf('H(t = %0.2f) = %0.3f\n', timePoints(tidx), allH(i, tidx));

        % Compute simulated error
        simErr = computeSimulationError(measProb, ...
            allSimResults(i).simProb, errorType, XDE);

        subplot(2, numExpCond, i)
        hold on
        plot(viewTimes.', simErr, 'LineWidth', 2, 'Color', cmap(tidx,:));
        hold off
    
        % Compute analytic potential error
        anaErr = computeSimulationError(measProb, ...
            allSimResults(i).anaProb, errorType, XDE);

        subplot(2, numExpCond, i+numExpCond)
        hold on
        plot(viewTimes.', anaErr, 'LineWidth', 2, 'Color', cmap(tidx,:));
        hold off

    end

    subplot(2, numExpCond, i)
    axis square
    xlim(xLim);
    ylim(yLim);
    xlabel('Simulation Time');

    if exist('cmap', 'var')
        cb = colorbar;
        cb.Color = 'k';
        set(gca, 'Clim', crange);
        set(gca, 'Colormap', cmap);
        if exist('cticks', 'var'), cb.Ticks = cticks; end
        if exist('cticklabels', 'var'), cb.TickLabels = cticklabels; end
        if exist('cbString', 'var'), cb.Label.String = cbString; end
    end

    title(sprintf('Fit Potential Error\na = %0.2f, b = %0.2f', ...
    a, b), 'Color', 'k');

    subplot(2, numExpCond, i+numExpCond)
    axis square
    xlim(xLim);
    ylim(yLim);
    xlabel('Simulation Time')

    if exist('cmap', 'var')
        cb = colorbar;
        cb.Color = 'k';
        set(gca, 'Clim', crange);
        set(gca, 'Colormap', cmap);
        if exist('cticks', 'var'), cb.Ticks = cticks; end
        if exist('cticklabels', 'var'), cb.TickLabels = cticklabels; end
        if exist('cbString', 'var'), cb.Label.String = cbString; end
    end

    title(sprintf('Analytic Potential Error\na = %0.2f, b = %0.2f', ...
    a, b), 'Color', 'k');

    disp(' ');

end

sgtitle(errorType);

clear a b measProb anaErr simErr xLim yLim