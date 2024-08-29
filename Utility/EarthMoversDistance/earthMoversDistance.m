function [emd, flowMatrix] = earthMoversDistance(F1, W1, F2, W2, distMatrix)
%EARTHMOVERSDISTANCE Computes the Earth Mover's Distance (EMD) between two
%signatures defined on point sets in R^dim
%
%   INPUT PARAMETERS:
%
%       - F1:           #N1 x dim set of feature vector coordinates
%       - W1:           #N1 x 1 set of weights
%       - F2:           #N2 x dim set of feature vector coordinates
%       - W2:           #N2 x 1 set of weights
%       - distMatrix:   #N1 x #N2 set of costs for mapping between feature
%                       points. For classic EMD, this should be the
%                       Euclidean distance.
%
%   OUTPUT PARAMETERS:
%
%       - emd:          The Earth Mover's Distance between the signatures
%       - flowMatrix:   The flow matrix associated to 'emd'. Intuitively,
%                       flowMatrix(i,j) represents the amount of weight at
%                       F1(i,:) that is matched to the weight at F2(j,:)
%
%   by Dillon Cislo 2024/05/23

%--------------------------------------------------------------------------
% INPUT PROCESSING
%--------------------------------------------------------------------------
if (nargin < 5), distMatrix = []; end

validateattributes(F1, {'numeric'}, {'2d', 'finite', 'real'}, ...
    'earthMoversDistance', 'F1');
[N1, dim] = size(F1);

validateattributes(W1, {'numeric'}, {'vector', 'nonnegative', 'finite', ...
    'real', 'numel', N1}, 'earthMoversDistance', 'W1');
if (size(W1, 2) ~= 1), W1 = W1.'; end

validateattributes(F2, {'numeric'}, {'2d', 'finite', 'real', ...
    'ncols', dim}, 'earthMoversDistance', 'F2');
N2 = size(F2, 1);

validateattributes(W2, {'numeric'}, {'vector', 'nonnegative', 'finite', ...
    'real', 'numel', N2}, 'earthMoversDistance', 'W2');
if (size(W2, 2) ~= 1), W2 = W2.'; end

if isempty(distMatrix)

    distMatrix = pdist2(F1, F2, 'euclidean');

else

    validateattributes(distMatrix, {'numeric'}, {'2d', 'finite', ...
        'real', 'nonnegative', 'nrows', N1, 'ncols', N2}, ...
        'earthMoversDistance', 'distMatrix');

end

%--------------------------------------------------------------------------
% COMPUTE EARTH MOVER'S DISTANCE
%--------------------------------------------------------------------------

% Build inequality constraints
gridIDx = reshape(1:(N1*N2), [N1, N2]);
ICols = repmat((1:N1).', [1 N2]);
IRows = repmat((1:N2), [N1, 1])+N1;

A = sparse([ICols(:); IRows(:)], [gridIDx(:); gridIDx(:)], ...
    1, N1+N2, N1*N2);
b = sparse([W1; W2]);

clear gridIDx ICols IRows

% Build equality constraint
Aeq = ones(1, N1*N2);
beq = min(sum(W1), sum(W2));

% Specifiy nonnegativity constraint
lb = zeros(N1*N1, 1);

% Solve the linear problem
options = optimoptions('linprog', 'Display', 'off');
[flowMatrix, emd] = linprog(distMatrix(:), A, b, Aeq, beq, lb, [], options);
emd = emd ./ sum(flowMatrix);
flowMatrix = reshape(flowMatrix, [N1, N2]);

end



