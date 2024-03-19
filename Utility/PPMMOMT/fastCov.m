function C = fastCov(X, w)
% Fast calculation of the (unbiased) weighted covariance matrix. 
%
%   INPUT PARAMETERS:
%
%       - X:    #N x #D data matrix. Each row corresponds to a data point
%       - w:    #N x 1 data point weight vector. Weights should be
%               normalized so that sum(w) == N 
%
%   OUTPUT PARAMETERS:
%
%       - C:    #D x #D sample covariance matrix
%
%   by Dillon Cislo 01/27/2023

% Find the weighted mean of the input data
XW = repmat(w, 1, size(X,2)) .* X;
COM = mean(XW,1);

% Center the data relative to the weighted mean
X0 = X - repmat(COM, size(X,1), 1);

% Calculate the (unbiased) weighted covariance matrix
C = repmat(sqrt(w), 1, size(X,2)) .* X0;
C = (C.' * C) ./ (size(X,1)-1);

end