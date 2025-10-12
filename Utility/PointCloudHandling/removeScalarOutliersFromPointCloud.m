function U = removeScalarOutliersFromPointCloud( ...
    X, Uin, outlierThreshold, outlierNNSize)
%REMOVESCALAROUTLIERSFROMPOINTCLOUD Given a D-dimensional point cloud with
%a scalar field defined on each point, this function removes outliers
%(defined by a user specified global threshold) by setting the invalid data
%points equal to the average of its valid nearest neighborhood
%
%   INPUT PARAMETERS:
%
%       - X:                    #N x dim set of input points
%
%       - Uin:                  #N x 1 vector of scalar function values
%
%       - outlierThreshold:     The threshold above and below which
%                               outliers are removed
%
%       - outlierNNSize:        The size of the nearest-neighbor averaging
%                               neighborhood
%
%   OUTPUT PARAMETERS:
%
%       - U:                    #N x 1 processed vector of scalar function
%                               values with outliers removed
%
%   by Dillon Cislo 2024/04/04

%--------------------------------------------------------------------------
% INPUT PROCESSING
%--------------------------------------------------------------------------
if (nargin < 4), outlierNNSize = 10; end

validateattributes(X, {'numeric'}, {'2d', 'finite', 'real'}, ...
    'removeScalarOutliersFromPointCloud', 'X');
numPoints = size(X,1); % dim = size(X,2);

validateattributes(Uin, {'numeric'}, {'vector', 'finite', 'real', ...
    'numel', numPoints}, 'removeScalarOutliersFromPointCloud', 'Uin');
if (size(Uin, 2) ~= 1), Uin = Uin.'; end

validateattributes(outlierThreshold, {'numeric'}, {'vector', ...
    'finite', 'real', 'numel', 2}, ...
    'removeScalarOutliersFromPointCloud', 'outlierThreshold');
assert(outlierThreshold(2) > outlierThreshold(1), ...
    ['Outlier threshold must have a second element ' ...
    'that is greater than its first element']);

validateattributes(outlierNNSize, {'numeric'}, ...
    {'positive', 'integer', 'scalar', 'finite', 'real'}, ...
    'removeScalarOutliersFromPointCloud', 'outlierNNSize');

%--------------------------------------------------------------------------
% REMOVE OUTLIERS
%--------------------------------------------------------------------------

U = Uin;

rmIDx = find( (U < outlierThreshold(1)) | ...
    (outlierThreshold(2) < U) );

if ~any(rmIDx), return; end

[nnIDx, nnDists] = knnsearch(X, X, 'K', numPoints);
nnIDx(:, 1) = []; nnDists(:, 1) = [];
assert(all(nnDists(:) > 0), 'Input point set contains duplicates');

for i = 1:numel(rmIDx)

    curID = rmIDx(i); % The current outlier

    % Determine the averaging neighborhood
    curNNIDx = ~ismember(nnIDx(curID, :), rmIDx);
    assert(sum(curNNIDx) >= outlierNNSize, ...
        'Too few neighbors for outlier removal');
    curNNIDx = find(curNNIDx, outlierNNSize, 'first');
    curNNDists = nnDists(curID, curNNIDx);
    curNNIDx = nnIDx(curID, curNNIDx);

    % Averaging weights are proportional to inverse distance to the
    % outlier
    rmWeights = (1./curNNDists) ./ sum(1./curNNDists);

    U(curID) = sum(rmWeights .* U(curNNIDx) .');

end

end