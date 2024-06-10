%% Test Point Cloud Vector Field Push Forward =============================
close all; clc;

k = 15; % Number of nearest-neighbors defining local tangent space
dim = 3; % Intrinsic dimensionality of the manifold

% Generate points on the sphere
% numPoints = 8000;
% X = randsphere(numPoints);
[X, F] = subdivided_sphere(4);
numPoints = size(X,1);

% Transform points to an ellipsoid
ellipseAxes = [3 2 1];
XPrime = repmat(ellipseAxes, numPoints, 1) .* X;

% Design a vector field on the unit sphere
x = X(:,1); y = X(:,2); z = X(:,3);
trueV = [ x .* z .* ( z.^2 - 1/4 ) - y, ...
    y .* z .* ( z.^2 - 1/4 ) + x, ...
    -( x.^2 + y.^2 ) .* ( z.^2 - 1/4 ) ];
clear x y z

% Push forward this vector field to the ellipse analytically
trueVPrime = trueV * diag(ellipseAxes);

% Push forward this vector field using the numerical method
[testVPrime, ~, ~, ~, ~] = pushForwardPointCloudVectorField(X, XPrime, ...
    trueV, dim, k, true);

% Pull back the transformed vector field for completeness, since this is a
% diffeomorphism anyway
[testV, ~, ~, ~, ~] = pushForwardPointCloudVectorField(XPrime, X, ...
    trueVPrime, dim, k, true);

pushErr = sqrt(sum((trueVPrime-testVPrime).^2, 2)) ./ ...
    sqrt(sum(trueVPrime.^2, 2));
fprintf('Max push forward error = %0.5e\n', max(pushErr));

pullErr = sqrt(sum((trueV-testV).^2, 2)) ./ sqrt(sum(trueV.^2, 2));
fprintf('Max pull back error = %0.5e\n', max(pullErr));

%% View Results ===========================================================
close all; clc;

subplot(1,2,1)
trisurf(triangulation(F,X), 'FaceVertexCData', pullErr, ...
    'EdgeColor', 'none', 'FaceColor', 'interp');
hold on
pcshow(X)
quiver3(X(:,1), X(:,2), X(:,3), trueV(:,1), trueV(:,2), trueV(:,3), ...
    1, 'Color', 'm', 'LineWidth', 2);
quiver3(X(:,1), X(:,2), X(:,3), testV(:,1), testV(:,2), testV(:,3), ...
    1, 'Color', 'c', 'LineWidth', 2);
hold off
axis equal
colorbar('Color', 'w');
set(gca, 'Clim', [0 0.1]);
xlabel('X')
ylabel('Y')
zlabel('Z')

subplot(1,2,2)
trisurf(triangulation(F,XPrime), 'FaceVertexCData', pushErr, ...
    'EdgeColor', 'none', 'FaceColor', 'interp');
hold on
pcshow(XPrime)
quiver3(XPrime(:,1), XPrime(:,2), XPrime(:,3), ...
    trueVPrime(:,1), trueVPrime(:,2), trueVPrime(:,3), ...
    1, 'Color', 'm', 'LineWidth', 2);
quiver3(XPrime(:,1), XPrime(:,2), XPrime(:,3), ...
    testVPrime(:,1), testVPrime(:,2), testVPrime(:,3), ...
    1, 'Color', 'c', 'LineWidth', 2);
hold off
axis equal
colorbar('Color', 'w');
set(gca, 'Clim', [0 0.1]);
xlabel('X')
ylabel('Y')
zlabel('Z')

