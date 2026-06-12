addpath('../MeshHandling');
addpath('../SpecialFunctions');
addpath('../PointCloudHandling');

%% Test Parallel Transport Dijkstra (2D) ==================================
clear; close all; clc;

fprintf('Building triangulation... ');

% Build a triangulation of the unit disk
diskTri = diskTriangulation(20);
F = diskTri.ConnectivityList;
V = diskTri.Points;

% Extract edge list and associated adjacency matrix
E = edges(diskTri);
A = sparse(E, fliplr(E), 1, size(V,1), size(V,1));

% Extract edge lengths
L = edge_lengths(V, E);

% Build graph objects for distance computations
W = sparse(E, fliplr(E), [L, L], size(V,1), size(V,1));

G = graph(W); % Graph weighted by 2D edge lengths

fprintf('Done\n');

% Build tangent space basis in 2D (trivial)
fprintf('Building tangent spaces... ');
allBases = buildPointCloudTangentSpace(V, 2, 6, A);
fprintf('Done\n');

% Extract distance matrices using parallel transport Dijkstra
fprintf('Computing 2D distance matrix (parallel transport): ');
tic; D = parallelTransportDijkstra(V, E, allBases); toc;

% Distances in 2D should match exactly
trueD = squareform(pdist(V));  % Euclidean pairwise distances
fprintf('Max discrepancy from true distances = %.3e\n', ...
    max(abs(D(:) - trueD(:))));

%% Test Parallel Transport Dijkstra =======================================
clear; close all; clc;

fprintf('Building triangulations... ');

% Build a triangulation of the unit sphere

% Basic icosahedral subdivision
% [V, F] = subdivided_sphere(4);

% Uniformly random sampling
% V = randsphere(3000);
% F = freeBoundary(delaunayTriangulation(V));

% Blue noise sampling
[V, F] = subdivided_sphere(6);
V = random_points_on_mesh(V, F, 3000, 'Color', 'blue');
F = freeBoundary(delaunayTriangulation(V));

% Normalize vertex positions for good measure
V = V ./ sqrt(sum(V.^2, 2));

% Extract edge list from mesh
E = edges(triangulation(F,V));

% Build k-nn graph on the points
% k = 8;
% E = knnsearch(V, V, 'K', k+1);
% E = E(:, 2:end);
% E = [reshape(repmat((1:size(V,1))',1,k), [], 1), E(:)];
% E = unique(sort(E,2),'rows');

% Build associated adjacency matrix
A = sparse(E, fliplr(E), 1, size(V,1), size(V,1));

% Extract edge lengths
L = edge_lengths(V, E);

% Build graph objects for distance computations
W = sparse(E, fliplr(E), [L, L], size(V,1), size(V,1));

G = graph(W); % Graph weighted by 3D edge lengths

fprintf('Done\n');

% Build tangent space basis in 2D (trivial) and 3D
fprintf('Building tangent spaces... ');

% Fit tangent spaces using local PCA
% allBases = buildPointCloudTangentSpace(V, 2, 6, A);

% Compute analytic tangent spaces for points on the unit sphere
theta = acos(V(:,3)); phi = atan2(V(:,2), V(:,1));
etheta = [cos(theta) .* cos(phi), cos(theta) .* sin(phi), -sin(theta)];
ephi = [-sin(theta) .* sin(phi), sin(theta) .* cos(phi), zeros(size(phi))];
if any(sum(ephi, 2) == 0)
    ephi(sum(ephi, 2) == 0, :) = [0 1 0];
end
etheta = etheta ./ sqrt(sum(etheta.^2, 2));
ephi = ephi ./ sqrt(sum(ephi.^2, 2));
allBases = cellfun(@(x,y) [x.', y.'], ...
    mat2cell(etheta, ones(1, size(V,1)), 3), ...
    mat2cell(ephi, ones(1, size(V,1)), 3), 'Uni', false);

fprintf('Done\n');

% Extract distance matrices using vanilla Dijkstra
topoD = distances(G, 'Method', 'unweighted'); % Topological distances
fprintf('Computing 3D distance matrix (MATLAB): ');
tic; DM = distances(G, 'Method', 'positive'); toc;

% Extract distance matrices using parallel transport Dijkstra
fprintf('Computing 3D distance matrix (parallel transport): ');
tic; D = parallelTransportDijkstra(V, E, allBases); toc;

% Compute true geodesic distances on the hemispherical cap
trueD = pdist2(V, V, ...
    @(x, y) real(acos(dot(repmat(x, [size(y,1), 1]), y, 2))));

pairIDx = nchoosek(1:size(V,1), 2);
pairIDx = sub2ind(size(D), pairIDx(:,1), pairIDx(:,2));

pairTopoD = topoD(pairIDx);
pairDMErr = 100 * abs(DM(pairIDx) - trueD(pairIDx)) ./ trueD(pairIDx);
pairDErr = 100 * abs(D(pairIDx) - trueD(pairIDx)) ./ trueD(pairIDx);

fprintf('Average percent error (Dijkstra) = %0.3f pm %0.3f\n', ...
    mean(pairDMErr), std(pairDMErr));
fprintf('Average percent error (PT-Dijkstra) = %0.3f pm %0.3f\n', ...
    mean(pairDErr), std(pairDErr));

DMErr = nan(1, max(pairTopoD)); DMStd = DMErr;
DErr = nan(1, max(pairTopoD)); DStd = DErr;
for i = 1:max(pairTopoD)
    curPairIDx = pairTopoD == i;
    DMErr(i) = mean(pairDMErr(curPairIDx));
    DErr(i) = mean(pairDErr(curPairIDx));
    DMStd(i) = std(pairDMErr(curPairIDx));
    DStd(i) = std(pairDErr(curPairIDx));
end 

figure('Color', 'k');

subplot(1,2,1)
trisurf(triangulation(F, V));
axis equal tight
xlabel('x'); ylabel('y'); zlabel('z');

subplot(1,2,2);
errorbar(DMErr, DMStd);
hold on
errorbar(DErr, DStd);
hold off
axis tight square
grid on
xlabel('Topological Distance');
ylabel('Relative Errors in Length');
legend({'Dijkstra', 'PT-Dijkstra'});



