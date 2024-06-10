%% Test Diffusion Map Laplacian on a Sphere ===============================
% We include functionality to compare the diffusion map Laplacian/gradient
% to the well studied cotan Laplacian/FEM gradient operator, respectively
clear; close all; clc;

% Generate random points on the sphere
% numPoints = 2000;
% X = randsphere(numPoints); F = [];

% Generate a spherical mesh (needed for mesh operators)
[X, F] = subdivided_sphere(4);
numPoints = size(X, 1);

%--------------------------------------------------------------------------
% Generate Analytic Results
%--------------------------------------------------------------------------

fprintf('Generating analytic results... ');

syms theta phi x y z
assume( theta, 'real' ); assume( phi, 'real' );
assume( x, 'real'); assume( y, 'real'); assume( z, 'real' );

% The equation of the surface
R = [ sin(theta) * cos(phi); sin(theta) * sin(phi); cos(theta) ];

% The tangent vectors
Etheta = [ gradient(R(1), theta); gradient(R(2), theta); gradient(R(3), theta) ];
Ephi = [ gradient(R(1), phi); gradient(R(2), phi); gradient(R(3), phi) ];

%--------------------------------------------------------------------------
% Enter your favorite scalar field in Cartesian or spherical coordinates

% S = 1 / (1 + (x + 1/sqrt(2))^2 + z^2 ); DIVERGENT LAPLACIAN AT THETA = 0
% S = (3/32) .* sqrt(77/pi) * sin(theta)^5 * cos(5*phi); % A spherical harmonic
% S = 1+x+y.^2+x.^2.*y+x.^4+y.^5+(x.*y.*z).^2;
S = cos(4*phi).*cos(theta).*sin(theta).^4-cos(theta).^2;

%--------------------------------------------------------------------------

% Transform to spherical coordinates if necessary
S = simplify(subs( S, [x y z], [R(1) R(2) R(3)] ));

% Calculate the gradient of the scalar field
gradS = simplify( gradient(S, theta) * Etheta + ...
    ( gradient(S, phi) / sin(theta) ) * ( Ephi / sin(theta) ) );

% Calculate the Laplacian of the scalar field
lapS = simplify( ...
    gradient( sin(theta) * gradient(S, theta), theta ) / sin(theta) + ...
    gradient( gradient( S, phi ), phi ) / sin(theta)^2 );

fprintf('Done\n');

%--------------------------------------------------------------------------
% Convert Analytic Results to Numerical Quantities
%--------------------------------------------------------------------------

fprintf('Substituting numerical values for symbolic variables... ');

% (theta, phi) for each vertex
NTheta = acos(X(:,3));
NPhi = atan2(X(:,2), X(:,1));

% Quantities associated with the scalar field -----------------------------
% S = double(vpa(subs(S, {theta, phi}, {NTheta, NPhi})));
% gradS = double(vpa(subs(gradS.', {theta, phi}, {NTheta, NPhi})));
% lapS = double(vpa(subs(lapS, {theta, phi}, {NTheta, NPhi})));

S = matlabFunction(S, 'Vars', {theta, phi}); S = S(NTheta, NPhi);
gradS = matlabFunction(gradS.', 'Vars', {theta, phi}); gradS = gradS(NTheta, NPhi);
lapS = matlabFunction(lapS, 'Vars', {theta, phi}); lapS = lapS(NTheta, NPhi);

% Account for numerical roundoff
S(abs(S) < eps) = 0;
gradS(abs(gradS) < eps) = 0;
lapS(abs(lapS) < eps) = 0;

% Account for any divisions by zero
if any(isnan(S) | isinf(S))
    warning('Replacing NaN/Inf in S');
    S(isnan(S) | isinf(S)) = 0;
end
if any(isnan(gradS) | isinf(gradS))
    warning('Replacing NaN/Inf in gradS');
    gradS(isnan(gradS) | isinf(gradS)) = 0;
end
if any(isnan(lapS) | isinf(lapS))
    warning('Replacing NaN/Inf in lapS');
    lapS(isnan(lapS) | isinf(gradS)) = 0;
end

fprintf('Done\n');

clear Ephi Etheta NPhi NTheta phi R theta x y z

%% Compute the Laplacian of the Scalar Field ==============================
close all; clc;

useMeshOperators = false;
if useMeshOperators 

    L = cotmatrix(X, F);
    M = massmatrix(X, F, 'voronoi');

else   

   [L, M] = diffusionMapLaplacian(X, 7e-3, -1); % subdivided_sphere(4);
    % [L, M] = diffusionMapLaplacian(X, 2e-3, -1); % subdivided_sphere(5);
    % [L, M, lapSigma] = diffusionMapLaplacian(X, -1, -1);

end

NLapS = (M \ L) * S;

% The relative error
lapErr = abs(lapS - NLapS) ./ abs(lapS);

% Account for division by 0
lapErr(isinf(lapErr)) = 0;
lapErr(isnan(lapErr)) = 0;

fprintf('SCALAR LAPLACIAN ERROR MEASUREMENTS:\n')
fprintf('RMS Relative Error = %f\n', sqrt( mean( lapErr.^2 ) ));
fprintf('Max Relative Error = %f\n', max(lapErr));
fprintf('Median Relative Error = %f\n', median(lapErr));

% View Results ------------------------------------------------------------
figure('color', 'w');

subplot(2,2,1);
funcColors = vals2colormap(S);
scatter3(X(:,1), X(:,2), X(:,3), [], funcColors, 'filled');
axis equal
colorbar
title('Original S');
colorbar
set(gca, 'Clim', [min(S), max(S)]);

subplot(2,2,2);
trueLapColors = vals2colormap(lapS);
scatter3(X(:,1), X(:,2), X(:,3), [], trueLapColors, 'filled');
axis equal
colorbar
title('True \nabla^2 S');
set(gca, 'Clim', [min(lapS), max(lapS)]);

subplot(2,2,3);
testLapColors = vals2colormap(NLapS, 'parula', ...
    [min(lapS), max(lapS)]);
scatter3(X(:,1), X(:,2), X(:,3), [], testLapColors, 'filled');
axis equal
colorbar
title('Computed \nabla^2 S');
set(gca, 'Clim', [min(lapS), max(lapS)]);

subplot(2,2,4)
errColors = vals2colormap(lapErr, 'parula', [0 0.5]);
scatter3(X(:,1), X(:,2), X(:,3), [], errColors, 'filled');
axis equal
colorbar
set(gca, 'Clim', [0 0.5]);
title('Relative Error');

clear funcColors trueLapColors testLapColors errColors

%% Compute the Gradient of the Scalar Field ===============================
close all; clc;

useMeshOperators = false;
if useMeshOperators 

    NGradS = reshape(grad(X, F) * S, [size(F,1), 3]);
    [~, F2V] = meshAveragingOperators(F, X);
    NGradS = F2V * NGradS;

else   

    NGradS = diffusionGradient(X, S, 'IntrinsicDimension', 2, ...
        'NumEigenvectors', 250, 'EigMethod', 'complete', ...
        'KNN', 15, 'TimeStep', 2.5, 'Verbose', false, ...
        'LaplacianSigma', 7.5e-2, 'LaplacianKNN', -1);

    % NGradS = diffusionGradient(X, S, 'IntrinsicDimension', 2, ...
    %     'NumEigenvectors', 250, 'EigMethod', 'complete', ...
    %     'KNN', 15, 'TimeStep', 1.1, 'Verbose', true, ...
    %     'LaplacianSigma', -1, 'LaplacianKNN', -1);

end

% The relative error
gradErr = gradS - NGradS;
gradErr = sqrt(sum(gradErr.^2, 2)) ./ sqrt(sum(gradS.^2, 2));

% Account for division by 0
gradErr(isinf(gradErr)) = 0;
gradErr(isnan(gradErr)) = 0;

fprintf('SCALAR GRADIENT ERROR MEASUREMENTS:\n')
fprintf('RMS Relative Error = %f\n', sqrt( mean( gradErr.^2 ) ));
fprintf('Max Relative Error = %f\n', max(gradErr));
fprintf('Median Relative Error = %f\n', median(gradErr));

% View Results ------------------------------------------------------------
close all
figure('color', 'w');

ssf = 1;

normGradS = sqrt(sum(gradS.^2, 2));
normNGradS = sqrt(sum(NGradS.^2, 2));

subplot(2,2,1);
funcColors = vals2colormap(S);
scatter3(X(:,1), X(:,2), X(:,3), [], funcColors, 'filled');
axis equal
colorbar
title('Original S');
colorbar
set(gca, 'Clim', [min(S), max(S)]);

subplot(2,2,2);
trueGradColors = vals2colormap(normGradS);
scatter3(X(:,1), X(:,2), X(:,3), [], trueGradColors, 'filled');
hold on
quiver3(X(1:ssf:end, 1), X(1:ssf:end, 2), X(1:ssf:end, 3), ...
    gradS(1:ssf:end, 1), gradS(1:ssf:end, 2), gradS(1:ssf:end, 3), ...
    1, 'LineWidth', 1, 'Color', 'k');
hold off
axis equal
colorbar
title('True \nabla S');
set(gca, 'Clim', [min(normGradS), max(normGradS)]);


subplot(2,2,3);
testGradColors = vals2colormap(normNGradS, 'parula', ...
    [min(normNGradS), max(normNGradS)]);
scatter3(X(:,1), X(:,2), X(:,3), [], testGradColors, 'filled');
hold on
quiver3(X(1:ssf:end, 1), X(1:ssf:end, 2), X(1:ssf:end, 3), ...
    NGradS(1:ssf:end, 1), NGradS(1:ssf:end, 2), NGradS(1:ssf:end, 3), ...
    1, 'LineWidth', 1, 'Color', 'k');
hold off
axis equal
colorbar
title('True \nabla S');
set(gca, 'Clim', [min(normNGradS), max(normNGradS)]);

subplot(2,2,4)
errCrange = [0 0.5];
errColors = vals2colormap(gradErr, 'parula', errCrange);
scatter3(X(:,1), X(:,2), X(:,3), [], errColors, 'filled');
axis equal
colorbar
set(gca, 'Clim', errCrange);
title('Relative Error');

clear funcColors trueGradColors testGradColors errColors errCrange




