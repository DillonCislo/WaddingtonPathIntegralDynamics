%% TEST FIT STATIC LANDSCAPE WITH ORBITS ==================================
%
%   This is a canned example to test the functionality of the
%   'fitStaticLandscape' function with various differenct constraints
%   and periodic orbits
%
%   by Dillon Cislo 2024/04/01
%==========================================================================

[projectDir, ~, ~] = fileparts(matlab.desktop.editor.getActiveFilename);
cd(projectDir);

addpath(genpath('../'))
rmpath(genpath('../External/gptoolbox'))

addpath(genpath('/home/dillon/Documents/MATLAB/tubular'));
rmpath(genpath('/home/dillon/Documents/MATLAB/tubular/external/gptoolbox'));
addpath(genpath('/home/dillon/Documents/MATLAB/gptoolbox'));
rmpath(genpath('/home/dillon/Documents/MATLAB/tubular/external/CGAL_Code'));
addpath(genpath('/home/dillon/Documents/MATLAB/CGAL_Code'));
addpath(genpath('/home/dillon/Documents/MATLAB/GeometryCentralCode'));
addpath('/home/dillon/Documents/MATLAB/GrowSurface/MeshCreation');
addpath(genpath('/home/dillon/Documents/MATLAB/StreamlineMesh3D'));
addpath(genpath('/home/dillon/Documents/MATLAB/mesh2d'))
addpath(genpath('/home/dillon/Documents/Python_Workspace/point-cloud-moving-least-squares'))

%% ************************************************************************
% *************************************************************************
%               Generate Discrete Dynamical Manifold 
% *************************************************************************
% *************************************************************************
% The manifold is designed to ba a roughly tubular surface in 3D evocative
% of a possible dynamical manifold in nematode chemotaxis. This section
% uses some code that is not shipped with this package. The results can
% just be loaded from the corresponding data file


%% Build Base Surface Mesh/Potential ======================================
clear; close all; clc;

[projectDir, ~, ~] = fileparts(matlab.desktop.editor.getActiveFilename);
cd(projectDir);

% Surface mesh options
numMidlinePoints = 75;
numCircPoints = 75;
midlineLength = 15;

h = 0.25;
l = 9.75;
s = 0.05;
z0 = 40;

bottomLoopHeight = 4;
topLoopHeight = 13.5;
sepLoopHeight = (bottomLoopHeight+topLoopHeight)/2; % midlineLength/2;

bottomU = 1;
topU = 1;
% sepU = 1.25;
sepU = 1.1;
topBdyU = 1.25;
bottomBdyU = 1.5;

% Generate Surface Mesh ---------------------------------------------------
fprintf('Builing surface mesh... ')

% The midline is parallel to the z-axis
midlinePoints = [zeros(numMidlinePoints, 2), ...
    linspace(0, midlineLength, numMidlinePoints).'];

% Define the curve forming the surface of rotation
midlineRadii = s * (sinh((l/midlineLength) .* ...
    midlinePoints(1:(end-1),3) - asinh(z0)) + z0) + h;

% Build mesh
[F, V, ~, ~] = ...
    tubeMesh( midlinePoints, 'Radii', midlineRadii, ...
    'NumCircPoints', numCircPoints, ...
    'AddStartCap', false, 'AddEndCap', false, ...
    'ApplyIsoRotation', false );
F = bfs_orient(F);

% COM = barycenter(V, F);
% FN = faceNormal(triangulation(F,V));
% VN = per_vertex_normals(V, F, 'Weighting', 'angle');

% Extract mesh differential operators
% [V2F, F2V] = meshAveragingOperators(F, V, 'angle');
L = cotmatrix(V, F);
% M = massmatrix(V, F, 'barycentric');
% normM = M ./ max(abs(diag(M)));

% Scale midline to unit length. Sets the overall scale of the problem
V = V ./ midlineLength;
bottomLoopHeight = bottomLoopHeight ./ midlineLength;
topLoopHeight = topLoopHeight ./ midlineLength;
sepLoopHeight = sepLoopHeight ./ midlineLength;

% Extract the loops defining orbits, repellers, and separatrices
bottomLoopHeight = V(knnsearch(V(:,3), bottomLoopHeight), 3);
bottomLoopIDx = find(abs(V(:,3)-bottomLoopHeight) < 1e-4);
assert(numel(bottomLoopIDx) == numCircPoints, ...
    'Bottom loop ID number mismatch')
bottomLoopAngle = wrapTo2Pi(atan2(V(bottomLoopIDx,2), V(bottomLoopIDx,1)));
[~, sortIDx] = sort(bottomLoopAngle);
bottomLoopIDx = bottomLoopIDx(sortIDx);

bottomBdyLoopIDx = find(abs(V(:,3)-min(V(:,3))) < 1e-4);
assert(numel(bottomBdyLoopIDx) == numCircPoints, ...
    'Bottom boundary loop ID number mismatch')
bottomBdyLoopAngle = wrapTo2Pi(atan2(V(bottomBdyLoopIDx,2), ...
    V(bottomBdyLoopIDx,1)));
[~, sortIDx] = sort(bottomBdyLoopAngle);
bottomBdyLoopIDx = bottomBdyLoopIDx(sortIDx);

topLoopHeight = V(knnsearch(V(:,3), topLoopHeight), 3);
topLoopIDx = find(abs(V(:,3)-topLoopHeight) < 1e-4);
assert(numel(topLoopIDx) == numCircPoints, ...
    'Top loop ID number mismatch')
topLoopAngle = wrapTo2Pi(atan2(V(topLoopIDx,2), V(topLoopIDx,1)));
[~, sortIDx] = sort(topLoopAngle);
topLoopIDx = topLoopIDx(sortIDx);

topBdyLoopHeight = V(knnsearch(V(:,3), max(V(:,3))), 3);
topBdyLoopIDx = find(abs(V(:,3)-topBdyLoopHeight) < 1e-4);
assert(numel(topBdyLoopIDx) == numCircPoints, ...
    'Top boundary loop ID number mismatch')
topBdyLoopAngle = wrapTo2Pi(atan2(V(topBdyLoopIDx,2), V(topBdyLoopIDx,1)));
[~, sortIDx] = sort(topBdyLoopAngle);
topBdyLoopIDx = topBdyLoopIDx(sortIDx);

sepLoopHeight = V(knnsearch(V(:,3), sepLoopHeight), 3);
sepLoopIDx = find(abs(V(:,3)-sepLoopHeight) < 1e-4);
assert(numel(sepLoopIDx) == numCircPoints, ...
    'Separatrix loop ID number mismatch')
sepLoopAngle = wrapTo2Pi(atan2(V(sepLoopIDx,2), V(sepLoopIDx,1)));
[~, sortIDx] = sort(sepLoopAngle);
sepLoopIDx = sepLoopIDx(sortIDx);

fprintf('Done\n')

% Build Base Potential ----------------------------------------------------
fprintf('Computing base potential... ')

interpIDx = [bottomLoopIDx; topLoopIDx; sepLoopIDx; ...
    topBdyLoopIDx; bottomBdyLoopIDx];
interpVals = [bottomU .* ones(numCircPoints, 1); ...
    topU .* ones(numCircPoints, 1); sepU .* ones(numCircPoints, 1); ...
    topBdyU .* ones(numCircPoints, 1); bottomBdyU .* ones(numCircPoints, 1)];

% if normalizeMassMatrix
%     Q = L * (normM \ L);
% else
%     Q = L * (M \ L);
% end

U0 = min_quad_with_fixed(-L, zeros(size(V,1), 1), interpIDx, interpVals);
A = monotonicity_matrix(U0, F);
UV = monotonic_biharmonic(V, F, interpIDx, interpVals, ...
    'MonotonicityGraph', A, 'MonotonicityEpsilon', 1e-3);

fprintf('Done\n')

%--------------------------------------------------------------------------
% View Results
%--------------------------------------------------------------------------
gridIDx = reshape(1:size(V,1), numCircPoints, []);

gridLengths = sqrt(sum((V(gridIDx(:, 2:end), :) - ...
    V(gridIDx(:, 1:(end-1)), :)).^2, 2));
gridLengths = [zeros(numCircPoints,1), ...
    reshape(gridLengths, numCircPoints, [])];
gridLengths = cumsum(gridLengths, 2);

gridAngles = reshape(wrapTo2Pi(atan2(V(gridIDx,2), V(gridIDx,1))), ...
    numCircPoints, []) ./ (2 * pi);

rmFIDx = any(ismember(F, gridIDx(1,:)), 2) & ...
    any(ismember(F, gridIDx(end,:)), 2);
F2D = F(~rmFIDx, :);
V2D = [gridLengths(:), gridAngles(:)];

loopColors = orderedcolors('glow');

figure('Color', 'k');

% View Surface in 3D ------------------------------------------------------
subplot(1,2,1);

trisurf(triangulation(F,V), 'EdgeColor', 'k', 'FaceVertexCData', UV, ...
    'FaceColor', 'interp');

hold on

plot3(V(bottomLoopIDx,1), V(bottomLoopIDx,2), V(bottomLoopIDx,3), ...
    'Color', loopColors(1,:), 'LineWidth', 5)

plot3(V(topLoopIDx,1), V(topLoopIDx,2), V(topLoopIDx,3), ...
    'Color', loopColors(2,:), 'LineWidth', 5)

plot3(V(sepLoopIDx,1), V(sepLoopIDx,2), V(sepLoopIDx,3), ...
    'Color', loopColors(3,:), 'LineWidth', 5)

hold off

axis equal

cb1 = colorbar('Color', 'w');
cb1.Label.String = 'U';

set(gca, 'Clim', [min(UV), mean(UV(topBdyLoopIDx))]);

% View Surface in 2D Arc Length Parameterization --------------------------
subplot(1,2,2)

trisurf(triangulation(F2D, [V2D, UV]), ...
    'EdgeColor', 'none', 'FaceColor', 'interp');

hold on

plot3(V2D(bottomLoopIDx,1), V2D(bottomLoopIDx,2), UV(bottomLoopIDx), ...
    'Color', loopColors(1,:), 'LineWidth', 5)

plot3(V2D(topLoopIDx,1), V2D(topLoopIDx,2), UV(topLoopIDx), ...
    'Color', loopColors(2,:), 'LineWidth', 5)

plot3(V2D(sepLoopIDx,1), V2D(sepLoopIDx,2), UV(sepLoopIDx), ...
    'Color', loopColors(3,:), 'LineWidth', 5)

hold off

axis equal tight

cb2 = colorbar('Color', 'w');
cb2.Label.String = 'U';

set(gca, 'Clim', [min(UV), mean(UV(topBdyLoopIDx))]);

clear loopColors bottomLoopAngle topLoopAngle sepLoopAngle h l s z0
clear cb1 cb2 U0 A bdyIDx bdyU COM F2D V2D gridIDx
clear gridAngles gridLengths bottomU topU bdyU sepU interpIDx interpVals
clear L F2V FN M normM Q V2F VN numCircPoints numMidlinePoints rmFIDx
clear midlineLength midlineRadii  sortIDx rmFIDx bottomBdyU
clear topBdyLoopHeight topBdyLoopIDx topBdyLoopAngle topBdyU
clear bottomBdyLoopHeight bottomBdyLoopIDx bottomBdyLoopAngle
% clear topLoopIDx bottomLoopIDx sepLoopIDx

%% Generate Points by Rejection Sampling ==================================
% Points are generated by simulating drift-diffusion dynamics using the
% BASE potential
close all; clc;

% rng(88, 'twister');
rng(25, 'twister'); % For reproducible random numbers

% Some point cloud options
confFactor = 100;
D0 = 0.5; % Diffusion coefficient
numPoints = 5000; % Total number of points
normalizeMassMatrix = true;
volumeElementType = 'GraphLaplacian';

xLim = [-0.75 0.75];
yLim = [-0.75 0.75];
zLim = [-0.25 1.25];
maxVal = 0.2;

X = nan(numPoints, 3);
while any(isnan(X))

    unsetIDx = find(any(isnan(X), 2));
    numUnset = numel(unsetIDx);
    progressbar(numPoints-numUnset, numPoints)

    candidateX = [diff(xLim) * rand([numUnset, 1]) + xLim(1), ...
        diff(yLim) * rand([numUnset, 1]) + yLim(1), ...
        diff(zLim) * rand([numUnset, 1]) + zLim(1)];

    [candidateProb, ~] = sampleProbFunction( ...
        candidateX, F, V, UV, D0, confFactor);

    goodCandidates = (maxVal * rand([numUnset, 1])) < candidateProb;

    X(unsetIDx(1:sum(goodCandidates)), :) = candidateX(goodCandidates, :);

end

% Estimate Base Potential From Point Cloud -------------------------------_

% This will be the "dynamical" time step parameter which we will hold
% fixed for the rest of the script
dt = 1e-3; % Smooth for 1e4 points
% dt = 0.5e-3;

recalculateBasePotential = true;
if ~exist('UB', 'var') || recalculateBasePotential

    fprintf('\nEstimating pseudopotential on subsampled point cloud... ');
    UB = -log(gaussianKDE(X, X, [], sqrt(2 * dt), false, [], [], true, true));
    fprintf('Done\n');

end

% Check that the associated transition matrix is irreducible
T = computeTransitionMatrix(X, UB, dt);
isIrreducible = ~isTransitionMatrixReducible(T);
assert(isIrreducible, ['Point set potential transition matrix is ' ...
    'reducible. Choose a larger time step']);

clear recalculateBasePotential isIrreducible T

%--------------------------------------------------------------------------
% View Results
%--------------------------------------------------------------------------
loopColors = orderedcolors('glow');

figure('Color', 'k');

pcshow(X, UB, 'MarkerSize', 50);

hold on

plot3(V(bottomLoopIDx,1), V(bottomLoopIDx,2), V(bottomLoopIDx,3), ...
    'Color', loopColors(1,:), 'LineWidth', 5)

plot3(V(topLoopIDx,1), V(topLoopIDx,2), V(topLoopIDx,3), ...
    'Color', loopColors(2,:), 'LineWidth', 5)

plot3(V(sepLoopIDx,1), V(sepLoopIDx,2), V(sepLoopIDx,3), ...
    'Color', loopColors(3,:), 'LineWidth', 5)


hold off

colorbar

set(gca, 'Clim', prctile(UB, [0, 95]))

axis equal

clear loopColors xLim yLim zLim maxVal unsetIDx numUnset candidateX
clear candidateProb goodCandidates

%% Extract Orbits/Separatrices From Base Potential Transition Matrix ======
close all; clc;

numCircSubsample = 24;

recalculateManifolds = true;
if ~exist('allPaths', 'var') || ~exist('allLoops', 'var') || ...
        recalculateManifolds

    fprintf('Estimating unstable manifolds from pseudopotential... ');

    logTUM = computeLogTransitionMatrix(X, UB, dt, ...
        'VolumeElementType', volumeElementType, 'ClipThreshold', 0, ...
        'ScalarMetric', 0.1, 'DiffusionCoefficient', 1e-2);
        % 'ScalarMetric', 0.5, 'DiffusionCoefficient', 2e-2);

    ssIDx = round(linspace(1, numel(topLoopIDx)+1, numCircSubsample+1));
    ssIDx(end) = [];
    assert(isequal(ssIDx, unique(ssIDx)), 'Bad index subsampling');

    topPathIDx = unique(knnsearch(X, V(topLoopIDx(ssIDx), :)), 'stable');
    bottomPathIDx = unique(knnsearch(X, V(bottomLoopIDx(ssIDx), :)), 'stable');
    sepPathIDx = unique(knnsearch(X, V(sepLoopIDx(ssIDx), :)), 'stable');

    % Extract the points comprising the top loop
    pathPairIDx = (1:numel(topPathIDx)).';
    pathPairIDx = [pathPairIDx, circshift(pathPairIDx, [-1 0])];
    [topPath, ~, ~] = ...
        computeMostProbablePaths(X, [], topPathIDx(pathPairIDx), [], ...
        'LogTransitionMatrix', logTUM);
    topPath = cellfun(@(x) x(2:end), topPath, 'Uni', false);
    topPath = circshift(vertcat(topPath{:}), [1 0]);
    assert(isequal(topPath, unique(topPath, 'stable')), ...
        'Duplicate points in top path');

    % Extract the points comprising the bottom loop
    pathPairIDx = (1:numel(bottomPathIDx)).';
    pathPairIDx = [pathPairIDx, circshift(pathPairIDx, [-1 0])];
    [bottomPath, ~, ~] = ...
        computeMostProbablePaths(X, [], bottomPathIDx(pathPairIDx), [], ...
        'LogTransitionMatrix', logTUM);
    bottomPath = cellfun(@(x) x(2:end), bottomPath, 'Uni', false);
    bottomPath = circshift(vertcat(bottomPath{:}), [1 0]);
    assert(isequal(bottomPath, unique(bottomPath, 'stable')), ...
        'Duplicate points in bottom path');

    % Extract the points comprising the separatrix loop
    pathPairIDx = (1:numel(sepPathIDx)).';
    pathPairIDx = [pathPairIDx, circshift(pathPairIDx, [-1 0])];
    [sepPath, ~, ~] = ...
        computeMostProbablePaths(X, [], sepPathIDx(pathPairIDx), [], ...
        'LogTransitionMatrix', logTUM);
    sepPath = cellfun(@(x) x(2:end), sepPath, 'Uni', false);
    sepPath = circshift(vertcat(sepPath{:}), [1 0]);
    assert(isequal(sepPath, unique(sepPath, 'stable')), ...
        'Duplicate points in separatrix path');

    % Break up the separatrix loop into four chunks
    ssIDx = round(linspace(1, numel(topLoopIDx)+1, 4+1));
    ssIDx(end) = [];
    assert(isequal(ssIDx, unique(ssIDx)), 'Bad index subsampling');

    topEndPointIDx = topPath(knnsearch(X(topPath, :), V(topLoopIDx(ssIDx), :)));
    bottomEndPointIDx = bottomPath(knnsearch(X(bottomPath, :), V(bottomLoopIDx(ssIDx), :)));
    sepEndPointInPathIDx = knnsearch(X(sepPath, :), V(sepLoopIDx(ssIDx), :));
    sepEndPointIDx = sepPath(sepEndPointInPathIDx);

    assert(numel(topEndPointIDx) == 4, 'Top endpoint number mismatch');
    assert(numel(bottomEndPointIDx) == 4, 'Bottom endpoint number mismatch');
    assert(numel(sepEndPointIDx) == 4, 'Separatrix endpoint number mismatch');

    sepPaths = mat2cell([sepEndPointInPathIDx, ...
        circshift(sepEndPointInPathIDx, [-1 0])], ...
        ones(numel(sepEndPointInPathIDx),1), 2);
    for i = 1:numel(sepPaths)
        curPath = circshift(sepPath, [-sepPaths{i}(1)+1, 0]);
        curPath = curPath(1:mod(diff(sepPaths{i})+1, numel(sepPath)));
        sepPaths{i} = curPath;
    end

    % Find the paths from the separatrix anchor points to the loop anchor
    % points
    pathPairIDx = [sepEndPointIDx, topEndPointIDx; ...
        sepEndPointIDx, bottomEndPointIDx];
    [allPaths, ~, ~] = ...
        computeMostProbablePaths(X, [], pathPairIDx, [], ...
        'LogTransitionMatrix', logTUM);

    for i = 1:numel(allPaths)
        rmIDx = ismember(allPaths{i}, [bottomPath; topPath]);
        rmIDx = [false; rmIDx(2:end) & rmIDx(1:(end-1))];
        allPaths{i} = allPaths{i}(~rmIDx);
    end

    allPaths = vertcat(allPaths, sepPaths);

    allLoops = cell(2,1);
    allLoops{1} = bottomPath;
    allLoops{2} = topPath;

    fprintf('Done\n');

end

fprintf('Number of points in separatrix path = %d\n', numel(sepPath))
fprintf('Number of points in bottom path = %d\n', numel(bottomPath))
fprintf('Number of points in top path = %d\n', numel(topPath))

%--------------------------------------------------------------------------
% View Results
%--------------------------------------------------------------------------
% close all; clc;
loopColors = orderedcolors('glow');

figure('Color', 'k');

pcshow(X, UB, 'MarkerSize', 50);

hold on

scatter3(X(allLoops{1},1), X(allLoops{1},2), X(allLoops{1},3), 80, ...
    loopColors(1,:), 'filled')
plot3(X([allLoops{1}; allLoops{1}(1)],1), ...
    X([allLoops{1}; allLoops{1}(1)],2), ...
    X([allLoops{1}; allLoops{1}(1)],3), ...
    'Color', loopColors(1,:), 'LineWidth', 3)

scatter3(X(allLoops{2},1), X(allLoops{2},2), X(allLoops{2},3), 80, ...
    loopColors(2,:), 'filled')
plot3(X([allLoops{2}; allLoops{2}(1)],1), ...
    X([allLoops{2}; allLoops{2}(1)],2), ...
    X([allLoops{2}; allLoops{2}(1)],3), ...
    'Color', loopColors(2,:), 'LineWidth', 3)

for i = 1:numel(allPaths)
    scatter3(X(allPaths{i},1), X(allPaths{i},2), X(allPaths{i},3), 70, ...
        loopColors(3,:), 'filled')
    plot3(X(allPaths{i},1), X(allPaths{i},2), X(allPaths{i},3), ...
        'Color', loopColors(3,:), 'LineWidth', 3)
end

hold off

colorbar

set(gca, 'Clim', prctile(UB, [0, 95]))

axis equal

clear numCircSubsample recalculateManifolds logTUM ssIDx sepPath
clear topPathIDx bottomPathIDx sepPathIDx pathPairIDx topPath
clear bottomPath topEndPointIDx bottomEndPointIDx sepEndPointInPathIDx
clear sepEndPointIDx sepPaths i curPath rmIDx loopColors

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
dtL = dt;
% dtL = 1e-3;
% dtL = 5e-4;

regSigma = 1e-12;
normalizeMassMatrix = true;
recalculateDiffOperators = true;
if ( ~exist('L', 'var') || ~exist('M', 'var') || recalculateDiffOperators )
    
    fprintf('Constructing diffusion map Laplacian... ')

    [L, M] = diffusionMapLaplacian(X, dtL);

    % Some Tikhonov regularization on the Laplacian to make it
    % negative-definite
    L = L - regSigma * eye(size(L));

    if normalizeMassMatrix, M = M ./ max(abs(diag(M))); end

    % Build the quadratic problem
    Q = L*(M\L);
    if ~issymmetric(Q), Q = (Q + Q.') ./ 2; end % Correct for roundoff error
    if (regSigma > 0), Q = Q + regSigma .* eye(size(Q)); end

    fprintf('Done\n');

end

clear recalculateDiffOperators


%% Compute Interpolated Potential and Rotational Velocity =================
close all; clc;

trueBottomHeight = -3;
trueTopHeight = 1;
trueHiSepHeight = 0.6;
trueLoSepHeight = 0.4;

smoothIters = 5;
rotMethod = 'bilaplacian-flat';
trueTopRotSpeed = 50;
trueBottomRotSpeed = -50;

%--------------------------------------------------------------------------
% Path Indexing Helper Variables
%--------------------------------------------------------------------------

% Extract lengths of edges in each path
allPathLengths = cell(size(allPaths));
for i = 1:numel(allPaths)
    curPath = [allPaths{i}(1:(end-1)), allPaths{i}(2:end)];
    curPathLengths = X(curPath(:,2), :) - X(curPath(:,1), :);
    curPathLengths = sqrt(sum(curPathLengths.^2, 2));
    % curPathLengths = cumsum(curPathLengths);
    allPathLengths{i} = curPathLengths;
end

% Unique vertices comprising loops
allLoopIDx = vertcat(allLoops{:});
allLoopIDx = allLoopIDx(:);
assert(numel(allLoopIDx) == numel(unique(allLoopIDx)), ...
    'Loops must not share points');

% fixPointIDx: The unique set of indices in 'X' corresponding to
% minima/saddles/loop termini, i.e. numel(fixPointIDx) == (# minima) + (#
% saddles) + (# loops)
% fixInPathIDx: #Px2 array of indices into fixPointIDx
% mapping minima/saddles/loop back into their corresponding paths
fixInPathIDx = cellfun(@(x) [x(1); x(end)], allPaths, 'Uni', false);
fixInPathIDx = cell2mat(fixInPathIDx);
fixPointIDx = unique(reshape(fixInPathIDx.', [], 1));
fixPointIDx(ismember(fixPointIDx, vertcat(allLoops{:}))) = [];
fixPointIDx = [fixPointIDx; cellfun(@(x) x(1), allLoops, 'Uni', true)];
for i = 1:numel(allLoops)
    fixInPathIDx(ismember(fixInPathIDx, allLoops{i})) = allLoops{i}(1);
end
[~, fixInPathIDx] = ismember(fixInPathIDx, fixPointIDx);
fixInPathIDx = reshape(fixInPathIDx, [2 numel(allPaths)]).';
numFixPoints = numel(fixPointIDx);
numNonLoopFixPoints = numFixPoints - numel(allLoops);
assert(all(fixInPathIDx(:) > 0) && ...
    all(ismember((1:numNonLoopFixPoints).', fixInPathIDx(:))), ...
    'Failed to assign path endpoints');

fixIsLoopIDx = ismember(fixPointIDx, allLoopIDx);

%--------------------------------------------------------------------------
% Compute Interpolated Potential
%--------------------------------------------------------------------------
fprintf('Computing interpolated potential... ')

trueFixHeights = nan(numel(fixPointIDx), 1);
trueFixHeights(ismember(fixPointIDx, allLoops{1})) = trueBottomHeight;
trueFixHeights(ismember(fixPointIDx, allLoops{2})) = trueTopHeight;

% Heuristic - definitely not robust
loIDx = ~fixIsLoopIDx & (abs(X(fixPointIDx, 2)) < 0.025);
hiIDx = ~fixIsLoopIDx & (abs(X(fixPointIDx, 1)) < 0.025);
assert((sum(loIDx) == 2) && (sum(hiIDx) == 2) && ~any(loIDx & hiIDx), ...
    'Failed to identify hi/lo separatrix fix points');
trueFixHeights(loIDx) = trueLoSepHeight;
trueFixHeights(hiIDx) = trueHiSepHeight;

[knownU, knownIDx] = interpolateValuesAlongPath( ...
    trueFixHeights(fixInPathIDx), allPaths, 'PathLengths', allPathLengths);
loopU = cellfun(@(x, y)  x .* ones(numel(y), 1), ...
    {trueBottomHeight; trueTopHeight}, allLoops, 'Uni', false);
knownU = [knownU; vertcat(loopU{:})];
[knownIDx, uniqueIDx, ~] = unique([knownIDx; allLoopIDx], 'stable');
knownU = knownU(uniqueIDx);

trueUI = min_quad_with_fixed(Q, zeros(numPoints, 1), ...
    knownIDx, knownU, [], [], []);
curOutlierThreshold = [min(knownU), max(knownU)] + 1e-14 * [-1 1];
trueUI = removeScalarOutliersFromPointCloud( ...
    X, trueUI, curOutlierThreshold, 10);
trueU = trueUI + UB;

fprintf('Done\n');

try

    fprintf('Computing potential gradient... ')
    [UMLS, gradUMLS, ~] = pcmls(X, X, trueU, 'PolynomialOrder', 2, ...
        'WeightFunction', 'gaussian', 'WeightMethod', 'knn', ...
        'WeightParam', 18, 'Vectorized', false, 'ComputeGradients', true, ...
        'ComputeHessians', false, 'Verbose', false);
    fprintf('Done\n');

catch

    fprintf('Potential gradient computation failed\n')
    UMLS = zeros(size(X,1), 1);
    gradUMLS = zeros(size(X));

end


%--------------------------------------------------------------------------
% Compute Rotational Velocity 
%--------------------------------------------------------------------------
fprintf('Computing rotational velocity... ');

allLoopTangentVectors = cellfun(@(x) ...
    computePathTangentVectors(X, x, smoothIters, true), allLoops, ...
    'Uni', false);

if strcmpi(rotMethod, 'bilaplacian-flat')

    loopVelocities = [trueBottomRotSpeed .* allLoopTangentVectors{1}; ...
        trueTopRotSpeed .* allLoopTangentVectors{2}];

    trueRotV = zeros(size(X));
    for i = 1:size(X,2)
        trueRotV(:,i) = min_quad_with_fixed(Q, zeros(numPoints, 1), ...
            allLoopIDx, loopVelocities(:,i), [], [], []);
    end


else

    error('Rotation velocity method not yet implemented');

end

% Handle interpolated velocity outliers. NOTE: This is NOT geometry
% aware for non-flat problems
for i = 1:size(X,2)

    curOutlierThreshold = [min(loopVelocities(:,i)), ...
        max(loopVelocities(:,i))] + 1e-14 * [-1 1];

    trueRotV(:,i) = removeScalarOutliersFromPointCloud( ...
        X, trueRotV(:,i), curOutlierThreshold, 10);
end

fprintf('Done\n');

%--------------------------------------------------------------------------
% View Results
%--------------------------------------------------------------------------
close all; clc;

normGradUMLS = sqrt(sum(gradUMLS.^2, 2));
if (max(normGradUMLS) - min(normGradUMLS)) > 1e-3
    incIDx = (prctile(normGradUMLS, 5) < normGradUMLS) & ...
        (normGradUMLS < prctile(normGradUMLS, 80));
else
    incIDx = true(size(X,1), 1);
end
incIDx = farthest_points(X(incIDx,:), 1000, 'Distance', 'euclidean');
incIDx = knnsearch(X, incIDx);

fig = figure;
axArray = [];

% Gradient Velocity Figure ------------------------------------------------
axArray(1) = subplot(1,3,1);

hold on

plot3(X([allLoops{1}; allLoops{1}(1)],1), ...
    X([allLoops{1}; allLoops{1}(1)],2), ...
    X([allLoops{1}; allLoops{1}(1)],3), ...
    'Color', 'c', 'LineWidth', 3)

plot3(X([allLoops{2}; allLoops{2}(1)],1), ...
    X([allLoops{2}; allLoops{2}(1)],2), ...
    X([allLoops{2}; allLoops{2}(1)],3), ...
    'Color', 'c', 'LineWidth', 3)

for i = 1:numel(allPaths)
    if any(ismember(allPaths{i}, allLoopIDx)), continue; end
    plot3(X(allPaths{i},1), X(allPaths{i},2), X(allPaths{i},3), ...
        'Color', 'c', 'LineWidth', 3)
end


quiver3(X(incIDx,1), X(incIDx,2), X(incIDx,3), ...
    -gradUMLS(incIDx,1), -gradUMLS(incIDx,2), ...
    -gradUMLS(incIDx,3), 1, 'Color', 'm', 'LineWidth', 1.5);

hold off

axis equal tight

title('Gradient Velocity')

% Rotational Velocity Figure ----------------------------------------------
axArray(2) = subplot(1,3,2);

hold on

plot3(X([allLoops{1}; allLoops{1}(1)],1), ...
    X([allLoops{1}; allLoops{1}(1)],2), ...
    X([allLoops{1}; allLoops{1}(1)],3), ...
    'Color', 'c', 'LineWidth', 3)

plot3(X([allLoops{2}; allLoops{2}(1)],1), ...
    X([allLoops{2}; allLoops{2}(1)],2), ...
    X([allLoops{2}; allLoops{2}(1)],3), ...
    'Color', 'c', 'LineWidth', 3)

for i = 1:numel(allPaths)
    if any(ismember(allPaths{i}, allLoopIDx)), continue; end
    plot3(X(allPaths{i},1), X(allPaths{i},2), X(allPaths{i},3), ...
        'Color', 'c', 'LineWidth', 3)
end


quiver3(X(incIDx,1), X(incIDx,2), X(incIDx,3), ...
    trueRotV(incIDx,1), trueRotV(incIDx,2), ...
    trueRotV(incIDx,3), 1, 'Color', 'm', 'LineWidth', 1.5);

hold off

axis equal tight

title('Rotational Velocity')

% Rotational Velocity Figure ----------------------------------------------
axArray(3) = subplot(1,3,3);

hold on

plot3(X([allLoops{1}; allLoops{1}(1)],1), ...
    X([allLoops{1}; allLoops{1}(1)],2), ...
    X([allLoops{1}; allLoops{1}(1)],3), ...
    'Color', 'c', 'LineWidth', 3)

plot3(X([allLoops{2}; allLoops{2}(1)],1), ...
    X([allLoops{2}; allLoops{2}(1)],2), ...
    X([allLoops{2}; allLoops{2}(1)],3), ...
    'Color', 'c', 'LineWidth', 3)

for i = 1:numel(allPaths)
    if any(ismember(allPaths{i}, allLoopIDx)), continue; end
    plot3(X(allPaths{i},1), X(allPaths{i},2), X(allPaths{i},3), ...
        'Color', 'c', 'LineWidth', 3)
end


quiver3(X(incIDx,1), X(incIDx,2), X(incIDx,3), ...
    trueRotV(incIDx,1) - gradUMLS(incIDx,1), ...
    trueRotV(incIDx,2) - gradUMLS(incIDx,2), ...
    trueRotV(incIDx,3) - gradUMLS(incIDx,3), ...
    1, 'Color', 'm', 'LineWidth', 1.5);

hold off

axis equal tight

title('Total Velocity')

Link = linkprop(axArray, {'CameraUpVector', 'CameraPosition', ...
    'CameraTarget', 'XLim', 'YLim', 'ZLim'});
setappdata(fig, 'StoreTheLink', Link);

clear i curPath curPathLengths loIDx hiIDx curLoop loopEdges
clear loopTangentVectors loopEdgeLengths smoothIters fig gradCrange
clear rotCrange j interpIDx loopVelocities knownU knownIDx normGradUMLS
clear normRotV axArray Link UMLS uniqueIDx loopU curOutlierThreshold

%% Generate Single Cell Tracked Data ======================================
close all; clc;

rng(123, 'twister'); % For reproducible random numbers

% trueScalarMetric = 0.5;
% trueD = 0.15;

trueScalarMetric = 0.25; % 0.125;
trueD = 1; % For fitting
% trueD = 0.25; % For display
D0 = 1; % <-- NOTE THIS IS RESET HERE (best practice to set equal to 1)

numTracks = 1e3;
trackLength = 250;

viewResults = false;

fprintf('Building transition matrix... ')
[T, volumeElement] = computeTransitionMatrix(X, trueU, dt, ...
    'PointPotential', UB,  'VolumeElementType', volumeElementType, ...
    'ClipThreshold', 0, 'VectorField', trueRotV, 'UseGPU', true, ...
    'ScalarMetric', trueScalarMetric, 'DiffusionCoefficient', trueD);
fprintf('Done\n')

% Compute a probability based on the dynamic potential
dynProb = volumeElement .* exp(-trueU ./ trueD);
dynProb = dynProb ./ sum(dynProb);
dynProbCDF = cumsum(dynProb);
dynProbCDF(end) = 1;

% Sample initial conditions from the dynamic probability
dataTracks = zeros(numTracks, trackLength);
% dataTracks(:,1) = sum(repmat(dynProbCDF, [1, numTracks]) < ...
%     rand(1, numTracks), 1).' + 1;

if numTracks > numPoints
    dataTracks(:,1) = randsample(1:size(X,1), numTracks, true);
else
    dataTracks(:,1) = randsample(1:size(X,1), numTracks, false);
end


% Sample tracks
for tid = 2:trackLength
    progressbar(tid, trackLength)
    trackCDF = cumsum(T(:, dataTracks(:, tid-1)), 1); % numPoints x numTracks
    dataTracks(:,tid) = sum(trackCDF < rand(1, numTracks), 1).' + 1;
end

%--------------------------------------------------------------------------
% View Results
%--------------------------------------------------------------------------

if viewResults

    close all; clc;
    trackColors = distinguishable_colors(numTracks, [0 0 0; 1 1 1; 0 1 1]);
    num_trail_points = 5;

    h_fig = figure('Units', 'normalized', 'OuterPosition', [0 0 1 1]);
    h_ax = axes('Parent', h_fig, 'NextPlot', 'add');
    xlabel('x'); ylabel('y'); zlabel('z');
    axis(h_ax, 'equal', 'tight')
    camproj('orthographic')
    box on

    axes(h_ax)
    pcshow(X, 0.8 * ones(1,3), 'MarkerSize', 10)

    hold on

    plot3(X([allLoops{1}; allLoops{1}(1)],1), ...
        X([allLoops{1}; allLoops{1}(1)],2), ...
        X([allLoops{1}; allLoops{1}(1)],3), ...
        'Color', 'c', 'LineWidth', 1.5)

    plot3(X([allLoops{2}; allLoops{2}(1)],1), ...
        X([allLoops{2}; allLoops{2}(1)],2), ...
        X([allLoops{2}; allLoops{2}(1)],3), ...
        'Color', 'c', 'LineWidth', 1.5)

    for i = 1:numel(allPaths)
        if any(ismember(allPaths{i}, allLoopIDx)), continue; end
        plot3(X(allPaths{i},1), X(allPaths{i},2), X(allPaths{i},3), ...
            'Color', 'c', 'LineWidth', 3)
    end

    axis equal tight vis3d
    box on

    % view([0, 90]);

    h_trails = gobjects(numTracks, 1);
    h_heads = gobjects(1, 1);

    for tid = 1:trackLength

        delete(h_trails(isgraphics(h_trails)))
        delete(h_heads(isgraphics(h_heads)))

        if num_trail_points > 0
            trail_col_idx = (tid - num_trail_points):tid;
            trail_col_idx(trail_col_idx < 1) = [];
            trail_size = numel(trail_col_idx);
            for k = 1:numTracks
                trail = dataTracks(k, trail_col_idx);
                trail = X(trail, :);
                h_trails(k) = plot3(h_ax, trail(:, 1), trail(:, 2), ...
                    trail(:, 3), 'Color', trackColors(k,:), 'LineWidth', 2);
            end
        end

        h_heads = scatter3(h_ax, X(dataTracks(:, tid), 1), X(dataTracks(:, tid), 2), ...
            X(dataTracks(:, tid), 3), 100, trackColors, 'filled');

        title(sprintf('Step %d/%d', tid, trackLength))

        drawnow

    end

end

% clear trueScalarMetric trueD
clear numTracks trackLength T volumeElement
clear dynProb dynProbCDF tid trackCDF trackColors 
clear num_trail_oints trail_col_idx trail_size k trail j alpha col
clear h_fig h_ax i h_trails h_heads viewResults

%% Generate Simulated Probability Density Data ============================
close all; clc;

rng(123, 'twister'); % For reproducible random numbers

trueScalarMetric = 0.25;
trueD = 1; % For fitting
% trueD = 0.25; % For display
D0 = 1; % <-- NOTE THIS IS RESET HERE (best practice to set equal to 1)

probSigma = 0.05;
numSimTimes = 125;
numDataSets = 3;
assert(numDataSets <= 5, 'Please choose a number of data sets <= 5');

fprintf('Generating simulated data set... ')

[trueT, volumeElement] = computeTransitionMatrix(X, trueU, dt, ...
    'PointPotential', UB,  'VolumeElementType', volumeElementType, ...
    'ClipThreshold', 0, 'VectorField', trueRotV, 'UseGPU', true, ...
    'ScalarMetric', trueScalarMetric, 'DiffusionCoefficient', trueD);

dataProb = cell(numDataSets, 1);
dataTimes = cell(numDataSets, 1);
if (numDataSets == 1)
    initIDx = 5;
else
    initIDx = randsample((1:size(fixPointIDx, 1)), numDataSets);
end
initIDx = fixPointIDx(initIDx);
initIDx = rand(numDataSets, size(X,2));
initIDx(:,3) = initIDx(:,3) + 0.6;
initIDx = knnsearch(X, initIDx);

initIDx = [initIDx; knnsearch(X, [1, 0, mean(X(allLoops{1}, 3), 1)])];
numDataSets = numDataSets + 1;

assert(numel(initIDx) == numDataSets, 'Duplicate initial conditions');

for i = 1:numDataSets

    initProb = X - repmat(X(initIDx(i), :), [size(X,1), 1]);
    initProb = sum(initProb.^2, 2);
    initProb = exp(-initProb ./ (2 * probSigma.^2));
    initProb = initProb ./ (2 * pi * probSigma).^(size(X,2)/2);
    initProb = exp(UB ./ D0) .* initProb ./ size(X, 1);
    initProb = initProb ./ sum(initProb);

    [allProbabilities, viewTimes] = evolveProbabilities( ...
        initProb, trueT, numSimTimes-1, 'TimeStep', dt, 'UseGPU', true, ...
        'StrictNormalization', true);

    % Choose the number of data sets to retain (not including the initial
    % condition)
    numTimePoints = 10; % randi([3 5]);
    keepIDx = knnsearch(viewTimes.', ...
        linspace(0, max(viewTimes), numTimePoints).');
    dataProb{i} = allProbabilities(:, keepIDx);
    dataTimes{i} = viewTimes(keepIDx);

end

clear initIDx probSigma initProb allProbabilities viewTimes
clear numTimePoints keepIDx i

fprintf('Done\n')

%% View Simulated Probability Time Course =================================
close all; clc;

viewID = 4;
viewProb = dataProb{viewID};
viewDensity = exp(-UB ./ D0) .* viewProb;
viewTimes = dataTimes{viewID};

figure('Color', 'k', 'Units', 'normalized', 'OuterPosition', ...
    [0 0 1 1]);

for tidx = 1:size(viewProb, 2)

    subplot(1, 2, 1, 'replace');

    probCRange = prctile(viewProb(:, tidx), [0 98]);

    pcshow(X, viewProb(:, tidx));
    % view([0 90]);
    colorbar('Color', 'w');
    set(gca, 'Clim', probCRange);
    axis square vis3d
    camproj('orthographic')
    title(sprintf(' Probability T = %0.5f', viewTimes(tidx)), ...
        'Color', 'w');
    xlabel('x'); ylabel('y'); zlabel('z')

    subplot(1, 2, 2, 'replace');

    densCRange = prctile(viewDensity(:, tidx), [0 98]);

    pcshow(X, viewDensity(:, tidx));
    % view([0 90]);
    colorbar('Color', 'w');
    set(gca, 'Clim', densCRange);
    axis square vis3d
    camproj('orthographic')
    title(sprintf('Density T = %0.5f', viewTimes(tidx)), ...
        'Color', 'w');
    xlabel('x'); ylabel('y');

    drawnow

    pause(1);

end

clear viewID viewProb viewTimes probCRange viewDensity densCRange

%% ************************************************************************
% *************************************************************************
%                  FIT STATIC LANDSCAPE WITH ORBITS
% *************************************************************************
% *************************************************************************
close all; clc;

% Set optimization options
optOptions = {'Display', 'iter', 'FiniteDifferenceType', 'forward', ...
    'UseParallel', false, 'PlotFcn', {'optimplotx', 'optimplotfval'}};

% Generate an initial guess that satisfies the constraints (just a
% potential height sum constraint here)
% initGuess = 0.25 * rand(numel(isSaddle), 1) + 0.5;
% initGuess = sum(trueFixHeights) * (initGuess ./ sum(initGuess));
% initGuess = [initGuess; 1; 1];

% Generic initial guess (no constraints)
% initGuess = [zeros(numel(fixPointIDx) + numel(allLoops), 1); 1; 1];
initGuess = [trueFixHeights; trueBottomRotSpeed; trueTopRotSpeed; 1; 1];

% Specify fixed height constraints (if any)
% constFixHeights = nan(size(trueFixHeights));
% constFixHeights(end) = 0;
constFixHeights = [];
constLoopHeights = [];
constLoopSpeeds = [];

[optErr, fixHeights, loopHeights, loopSpeeds, D, scalarMetric, ...
    timeScale, fitTimes, optOutput, optRotV] = ...
    fitStaticLandscapeWithOrbits( ...
    X, dataProb, dataTimes, dt, allPaths, allLoops, ...
    'InitialGuess', initGuess, 'InitialConditions', {}, ...
    'NumSimTimes', numSimTimes, 'IsSaddle', false(numel(fixPointIDx), 1), ...
    'EnforceSaddles', false, 'ConstHeightSum', 0, ...
    'SimTimeHandling', 'none', 'OptimizationOptions', optOptions, ...
    'ConstFixedHeights', constFixHeights, 'ConstScalarMetric', trueScalarMetric, ...
    'ConstDiffusionCoefficient', trueD, 'EnforcePositiveDiffusion', true, ...
    'EnforcePositiveMetric', true, 'PointDiffusionCoefficient', D0, ...
    'PointPotential', UB, 'BasePotential', UB, ...
    'Laplacian', L, 'MassMatrix', M, 'PathLengths', allPathLengths, ...
    'VolumeElementType', volumeElementType, 'ClipThreshold', 0, ...
    'RotationProblemType', 'bilaplacian-flat', 'PrecomputeQuadProg', true, ...
    'UseGPU', true, 'Verbose', true, 'ErrorType', 'symKLD', ...
    'IntrinsicDimension', [], 'LoopTangentVectors', allLoopTangentVectors, ...
    'LoopSmoothIters', 5, 'TikhonovRegularization', regSigma);

%% ************************************************************************
% *************************************************************************
%          FIT STATIC LANDSCAPE WITH ORBITS FROM TRACKED DATA
% *************************************************************************
% *************************************************************************
close all; clc;

% Set optimization options
optOptions = {'Display', 'iter', 'FiniteDifferenceType', 'forward', ...
    'UseParallel', false, 'PlotFcn', {'optimplotx', 'optimplotfval'}};

% Generate an initial guess that satisfies the constraints (just a
% potential height sum constraint here)
% initGuess = 0.25 * rand(numel(isSaddle), 1) + 0.5;
% initGuess = sum(trueFixHeights) * (initGuess ./ sum(initGuess));
% initGuess = [initGuess; 1; 1];

% Generic initial guess (no constraints)
initGuess = [zeros(numel(fixPointIDx) + numel(allLoops), 1); 1; 1];
% initGuess = [trueFixHeights; trueBottomRotSpeed; trueTopRotSpeed; 1; 1];

% Specify fixed height constraints (if any)
% constFixHeights = nan(size(trueFixHeights));
% constFixHeights(end) = 0;
constFixHeights = [];
constLoopHeights = [];
constLoopSpeeds = [];

[optErr, fixHeights, loopHeights, loopSpeeds, ...
    D, scalarMetric, optOutput, optRotV] = ...
    fitStaticLandscapeWithOrbitsTrackedData( ...
    X, dataTracks, 1, dt, allPaths, allLoops, ...
    'InitialGuess', initGuess, 'IsSaddle', false(numel(fixPointIDx), 1), ...
    'EnforceSaddles', false, 'ConstHeightSum', 0, ...
    'OptimizationOptions', optOptions, 'UseGPU', true, 'Verbose', true, ...
    'ConstFixedHeights', constFixHeights, 'ConstScalarMetric', trueScalarMetric, ...
    'ConstDiffusionCoefficient', trueD, 'EnforcePositiveDiffusion', true, ...
    'EnforcePositiveMetric', true, 'PointDiffusionCoefficient', D0, ...
    'PointPotential', UB, 'BasePotential', UB, ...
    'Laplacian', L, 'MassMatrix', M, 'PathLengths', allPathLengths, ...
    'VolumeElementType', volumeElementType, 'ClipThreshold', 0, ...
    'RotationProblemType', 'bilaplacian-flat', 'PrecomputeQuadProg', true, ...
    'IntrinsicDimension', [], 'LoopTangentVectors', allLoopTangentVectors, ...
    'LoopSmoothIters', 5, 'TikhonovRegularization', regSigma);


%% View Optimization Results ==============================================
close all; clc;

smoothIters = 5;
rotMethod = 'bilaplacian-flat';

%--------------------------------------------------------------------------
% Compute Interpolated Potential
%--------------------------------------------------------------------------
fprintf('Computing interpolated potential... ')

optHeights = [fixHeights; loopHeights];
[knownU, knownIDx] = interpolateValuesAlongPath( ...
    optHeights(fixInPathIDx), allPaths, 'PathLengths', allPathLengths);
% loopU = cellfun(@(x, y)  x .* ones(numel(y), 1), ...
%     num2cell(loopHeights), allLoops, 'Uni', false);
% knownU = [knownU; vertcat(loopU{:})];
% [knownIDx, uniqueIDx, ~] = unique([knownIDx; allLoopIDx], 'stable');
% knownU = knownU(uniqueIDx);

optUI = min_quad_with_fixed(Q, zeros(numPoints, 1), ...
    knownIDx, knownU, [], [], []);
curOutlierThreshold = [min(knownU), max(knownU)] + 1e-14 * [-1 1];
optUI = removeScalarOutliersFromPointCloud( ...
    X, optUI, curOutlierThreshold, 10);
optU = optUI + UB;

fprintf('Done\n');

try

    fprintf('Computing potential gradient... ')
    [optUMLS, optGradUMLS, ~] = pcmls(X, X, optU, 'PolynomialOrder', 2, ...
        'WeightFunction', 'gaussian', 'WeightMethod', 'knn', ...
        'WeightParam', 18, 'Vectorized', false, 'ComputeGradients', true, ...
        'ComputeHessians', false, 'Verbose', true);
    fprintf('Done\n');

catch

    fprintf('Potential gradient computation failed\n')
    optUMLS = zeros(size(X,1), 1);
    optGradUMLS = zeros(size(X));

end


%--------------------------------------------------------------------------
% Generate Figures
%--------------------------------------------------------------------------
close all; clc;

normGradUMLS = sqrt(sum(optGradUMLS.^2, 2));
if (max(normGradUMLS) - min(normGradUMLS)) > 1e-3
    incIDx = (prctile(normGradUMLS, 5) < normGradUMLS) & ...
        (normGradUMLS < prctile(normGradUMLS, 80));
else
    incIDx = true(size(X,1), 1);
end
incIDx = farthest_points(X(incIDx,:), 1000, 'Distance', 'euclidean');
incIDx = knnsearch(X, incIDx);

fig = figure;
axArray = [];

% Gradient Velocity Figure ------------------------------------------------
axArray(1) = subplot(1,3,1);

hold on

plot3(X([allLoops{1}; allLoops{1}(1)],1), ...
    X([allLoops{1}; allLoops{1}(1)],2), ...
    X([allLoops{1}; allLoops{1}(1)],3), ...
    'Color', 'c', 'LineWidth', 3)

plot3(X([allLoops{2}; allLoops{2}(1)],1), ...
    X([allLoops{2}; allLoops{2}(1)],2), ...
    X([allLoops{2}; allLoops{2}(1)],3), ...
    'Color', 'c', 'LineWidth', 3)

for i = 1:numel(allPaths)
    if any(ismember(allPaths{i}, allLoopIDx)), continue; end
    plot3(X(allPaths{i},1), X(allPaths{i},2), X(allPaths{i},3), ...
        'Color', 'c', 'LineWidth', 3)
end


quiver3(X(incIDx,1), X(incIDx,2), X(incIDx,3), ...
    -optGradUMLS(incIDx,1), -optGradUMLS(incIDx,2), ...
    -optGradUMLS(incIDx,3), 1, 'Color', 'm', 'LineWidth', 1.5);

hold off

axis equal tight

title('Gradient Velocity')

% Rotational Velocity Figure ----------------------------------------------
axArray(2) = subplot(1,3,2);

hold on

plot3(X([allLoops{1}; allLoops{1}(1)],1), ...
    X([allLoops{1}; allLoops{1}(1)],2), ...
    X([allLoops{1}; allLoops{1}(1)],3), ...
    'Color', 'c', 'LineWidth', 3)

plot3(X([allLoops{2}; allLoops{2}(1)],1), ...
    X([allLoops{2}; allLoops{2}(1)],2), ...
    X([allLoops{2}; allLoops{2}(1)],3), ...
    'Color', 'c', 'LineWidth', 3)

for i = 1:numel(allPaths)
    if any(ismember(allPaths{i}, allLoopIDx)), continue; end
    plot3(X(allPaths{i},1), X(allPaths{i},2), X(allPaths{i},3), ...
        'Color', 'c', 'LineWidth', 3)
end


quiver3(X(incIDx,1), X(incIDx,2), X(incIDx,3), ...
    optRotV(incIDx,1), optRotV(incIDx,2), ...
    optRotV(incIDx,3), 1, 'Color', 'm', 'LineWidth', 1.5);

hold off

axis equal tight

title('Rotational Velocity')

% Rotational Velocity Figure ----------------------------------------------
axArray(3) = subplot(1,3,3);

hold on

plot3(X([allLoops{1}; allLoops{1}(1)],1), ...
    X([allLoops{1}; allLoops{1}(1)],2), ...
    X([allLoops{1}; allLoops{1}(1)],3), ...
    'Color', 'c', 'LineWidth', 3)

plot3(X([allLoops{2}; allLoops{2}(1)],1), ...
    X([allLoops{2}; allLoops{2}(1)],2), ...
    X([allLoops{2}; allLoops{2}(1)],3), ...
    'Color', 'c', 'LineWidth', 3)

for i = 1:numel(allPaths)
    if any(ismember(allPaths{i}, allLoopIDx)), continue; end
    plot3(X(allPaths{i},1), X(allPaths{i},2), X(allPaths{i},3), ...
        'Color', 'c', 'LineWidth', 3)
end


quiver3(X(incIDx,1), X(incIDx,2), X(incIDx,3), ...
    optRotV(incIDx,1) - optGradUMLS(incIDx,1), ...
    optRotV(incIDx,2) - optGradUMLS(incIDx,2), ...
    optRotV(incIDx,3) - optGradUMLS(incIDx,3), ...
    1, 'Color', 'm', 'LineWidth', 1.5);

hold off

axis equal tight

title('Total Velocity')

Link = linkprop(axArray, {'CameraUpVector', 'CameraPosition', ...
    'CameraTarget', 'XLim', 'YLim', 'ZLim'});
setappdata(fig, 'StoreTheLink', Link);

clear i curPath curPathLengths loIDx hiIDx curLoop loopEdges
clear loopTangentVectors loopEdgeLengths smoothIters fig gradCrange
clear rotCrange j interpIDx loopVelocities knownU knownIDx normGradUMLS
clear normRotV axArray Link optUMLS uniqueIDx loopU curOutlierThreshold


%% ************************************************************************
% *************************************************************************
%                   Helper Functions/Visualizations
% *************************************************************************
% *************************************************************************

%% SAMPLEPROBFUNCTION =====================================================

function [sampleProb, samplePotential] = sampleProbFunction(...
    P, F, V, U, D, confFactor)
% SAMPLEPROBFUNCTION Computes the (unnormalizeed) probability of sampling a
% point at a given location

[sqrD, I, C] = point_mesh_squared_distance(P, V, F);
B = barycentric_coordinates(C, V(F(I,1),:), V(F(I,2),:), V(F(I,3),:));

UF = U(F);
% samplePotential = sum(B .* UF(I,:), 2) + confFactor .* sqrD.^2;
samplePotential = sum(B .* UF(I,:), 2) + (exp(confFactor .* sqrD)-1);
% samplePotential = sum(B .* UF(I,:), 2) + (exp(confFactor .* sqrt(sqrD))-1);

sampleProb = exp(-samplePotential ./ D);

end