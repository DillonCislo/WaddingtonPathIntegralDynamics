%% Test Diffusion Map Laplacian on a Sphere ===============================
% We include functionality to compare the diffusion map Laplacian to the
% well studied cotangent Laplacian on a spherical mesh
clear; close all; clc;

numPoints = 8000;
% X = randsphere(numPoints);
[X, F] = subdivided_sphere(4);

% Compute a spherical harmonic on the point cloud
l = randi(4)+1;
m = randsample(-l:l, 1);
Ylm = sphericalHarmonics(X, l, m, 'real');
trueLapYlm = -l * (l+1) * Ylm;

% Compute the discrete Laplacian of the spherical harmonic
useCotanLaplacian = false;
if useCotanLaplacian   
    L = cotmatrix(X, F);
    M = massmatrix(X, F, 'voronoi');
else   
    [L, M] = diffusionMapLaplacian(X, -1, -1);
end
lapYlm = (M \ L) * Ylm;

% lapErr = abs(lapYlm - trueLapYlm) ./ sqrt(mean(trueLapYlm.^2));
lapErr = abs(lapYlm - trueLapYlm) ./ abs(trueLapYlm);
lapErr(isinf(lapErr)) = NaN;

fprintf('Max error = %0.5e\n', max(lapErr));
fprintf('Mean error = %0.5e\n', mean(lapErr, 'omitnan'));
fprintf('Median error = %0.5e\n', median(lapErr, 'omitnan'));

% View Results ------------------------------------------------------------
figure('color', 'w', 'units', 'normalized', 'outerposition', [0.5 0 0.5 1]);

subplot(2,2,1);
funcColors = vals2colormap(Ylm);
scatter3(X(:,1), X(:,2), X(:,3), [], funcColors, 'filled');
axis equal
colorbar
title(['Y_{' num2str(l) num2str(m) '}(\theta, \phi)']);
colorbar
set(gca, 'Clim', [min(Ylm), max(Ylm)]);

subplot(2,2,2);
trueLapColors = vals2colormap(trueLapYlm);
scatter3(X(:,1), X(:,2), X(:,3), [], trueLapColors, 'filled');
axis equal
colorbar
title(['True \nabla^2 Y_{' num2str(l) num2str(m) '}(\theta, \phi)']);
set(gca, 'Clim', [min(-l * (l+1) * Ylm), max(-l * (l+1) * Ylm)]);

subplot(2,2,3);
testLapColors = vals2colormap(lapYlm, 'parula', ...
    [min(-l * (l+1) * Ylm), max(-l * (l+1) * Ylm)]);
scatter3(X(:,1), X(:,2), X(:,3), [], testLapColors, 'filled');
axis equal
colorbar
title(['Computed \nabla^2 Y_{' num2str(l) num2str(m) '}(\theta, \phi)']);
set(gca, 'Clim', [min(-l * (l+1) * Ylm), max(-l * (l+1) * Ylm)]);

subplot(2,2,4)
errColors = vals2colormap(lapErr, 'parula', [0 1]);
scatter3(X(:,1), X(:,2), X(:,3), [], errColors, 'filled');
axis equal
colorbar
set(gca, 'Clim', [0 1]);
title('Relative Error');