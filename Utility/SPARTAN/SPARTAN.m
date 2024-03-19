function [XSS, XSSIDx, Phi] = SPARTAN(X, numSamples, varargin)
%SPARTAN An implementation of the SPARTAN algorithm from "An optimal
%transport approach for selecting a representative subsample with
%application in efficient kernel density estimation" by Zhang et al.
%(2022). This algorithm performs a density aware subsampling of an
%D-dimensional point cloud
%
%   INPUT PARAMETERS:
%
%       - X:    #N x #D array of points to be subsampled
%
%       - numSamples:	The desired number of output samples
%
%   OPTIONAL INPUT PARAMETERS:
%
%       - ('OMTIterations', omtIter = 500): The number of iterations to run
%       the OMT process mapping the input distribution into the uniform
%       distribution on U[0,1]^D
%
%       - ('OMTWassersteinThresh', omtWasserThresh = 0): OMT will terminate
%       if the change in the Wasserstein distance of the iterates falls
%       below this threshold. WARNING: Not a great indicator of map quality
%       for large point set. Probably should just leave this equal to zero.
%
%       - ('QRPType', qrpType = 'sobol'): The quasi-random point generation
%       process used to produce a space-filling set of design points over
%       U[0,1]^D.
%
%       - ('QRPLeap', qrpLeap = 0): The leap parameter used for
%       quasi-random design point generation. Change to produce different
%       design points.
%
%       - ('QRPSkip', qrpSkip = 0): The skip parameter used for
%       quasi-random design point generation. Change to produce different
%       design points.
%
%       - ('QRPScramble', qrpScramble = 0): Whether or not to scramble the
%       quasi-random design points. Another knob to tune to adjust points.
%
%       - ('Verbose', verbose = false): Whether or not to print verbose
%       progress updates.
%
%   OUTPUT PARAMETERS:
%
%       - XSS:      #numSamples x #D array of subsampled points
%
%       - XSSIDx:   #numSamples x 1 column vector of point IDs in the
%                   original input cloud, i.e. XSS = X(XSSIDx, :)
%
%       - Phi:      #N x #D array of transformed input point coordinates
%                   mapped to the D-dimensional unit hypercube by OMT
%
%   by Dillon Cislo 01/28/2023

% Input Processing --------------------------------------------------------
validateattributes(X, {'numeric'}, {'2d', 'finite', 'real'});
validateattributes(numSamples, {'numeric'}, ...
    {'scalar', 'positive', 'finite', 'integer', 'real'});

numPoints= size(X,1); dim = size(X,2);
assert(numSamples < numPoints, ['Number of output samples must ' ...
    'be less than the number of input points']);

% Default options
omtIter = 500;
omtWasserThresh = 0;
qrpType = 'sobol';
qrpLeap = 0;
qrpSkip = 0;
qrpScramble = false;
verbose = false;

for i = 1:length(varargin)
   
    if isa(varargin{i}, 'double'), continue; end
    if isa(varargin{i}, 'single'), continue; end
    if isa(varargin{i}, 'logical'), continue; end
    
    if strcmpi(varargin{i}, 'OMTIterations')
        omtIter = varargin{i+1};
        validateattributes(omtIter, {'numeric'}, ...
            {'scalar', 'integer', 'positive', 'finite', 'real'});
    end    
    if strcmpi(varargin{i}, 'OMTWassersteinThresh')
        omtWasserThresh = varargin{i+1};
        validateattributes(omtWasserThresh, {'numeric'}, ...
            {'scalar', 'finite', 'real'});
    end    
    if strcmpi(varargin{i}, 'QRPType')
        qrpType = varargin{i+1};
        validateattributes(qrpType, {'char'}, {'vector'});
        assert(ismember(qrpType, {'sobol', 'halton'}), ...
            'Invalid quasi-random point generation type');
    end    
    if strcmpi(varargin{i}, 'QRPLeap')
        qrpLeap = varargin{i+1};
        validateattributes(qrpLeap, {'numeric'}, ...
            {'scalar', 'integer', 'finite', 'nonnegative', 'real'});
    end    
    if strcmpi(varargin{i}, 'QRPSkip')
        qrpSkip = varargin{i+1};
        validateattributes(qrpSkip, {'numeric'}, ...
            {'scalar', 'integer', 'finite', 'nonnegative', 'real'});
    end
    if strcmpi(varargin{i}, 'QRPScramble')
        qrpScramble = varargin{i+1};
        validateattributes(qrpScramble, {'logical'}, {'scalar'});
    end
    if strcmpi(varargin{i}, 'Verbose')
        verbose = varargin{i+1};
        validateattributes(verbose, {'logical'}, {'scalar'});
    end
end

% Run SPARTAN -------------------------------------------------------------

% Generate a synthetic random sample from U[0,1]^dim
if verbose, fprintf('Generating synthetic random sample... '); end
U = rand(numPoints, dim);
if verbose, fprintf('Done\n'); end

% Calculate the empirical optimal transport map from the input sample to
% the synthetic sample
if verbose, fprintf('Running PPMMOMT... '); end
WX = []; WU = [];
Phi = PPMMOMT(X, U, WX, WU, omtIter, omtWasserThresh);

for i = 1:dim
    [h, p, kd] = kstest2(Phi(:,i), U(:,i));
    assert(~h, 'PPMMOMT failed: p = %0.5e, ksstat = %0.5e\n', p, kd);
end
    
    
if verbose, fprintf('Done\n'); end

% Generate a space-filling set of design points on U[0,1]^dim
if verbose, fprintf('Generating design points... '); end

if strcmpi(qrpType, 'sobol')
    
    designSet = sobolset(dim, 'Skip', qrpSkip, 'Leap', qrpLeap);
    if qrpScramble
        designSet = scramble(designSet, 'MatousekAffineOwen');
    end
    
elseif strcmpi(qrpType, 'halton')
    
    designSet = haltonset(dim, 'Skip', qrpSkip, 'Leap', qrpLeap);
    if qrpScramble
        designSet = scramble(designSet, 'RR2');
    end
    
end

USS = net(designSet, numSamples);
if verbose, fprintf('Done\n'); end

% Point match transported data points to design points
if verbose, fprintf('Point-matching data points... '); end

XSSIDx = unique_knnsearch(Phi, USS);
assert(isequal(sort(XSSIDx), unique(XSSIDx)), ...
    'Invalid design point matching output');

XSS = X(XSSIDx, :);

if verbose, fprintf('Done\n'); end

end

function idx = unique_knnsearch(X, Y)
% A recursive function that will return the index of the nearest neighbor
% in X (target set) for each point in Y (source set). This method DOES NOT
% solve the more difficult problem of assigning a unique point that
% minimizes the total sum of distances between all pairs - it simply
% guarantees that there is a unique 1-to-1 matching.

if isempty(Y)
    
    % Base case
    idx = [];
    
else
    
    assert(~isempty(X), 'Empty target set');
    idx = knnsearch(X, Y);
    
    [matchedXIDx, matchedYIDx] = unique(idx);
    duplIDx = setdiff( (1:size(Y,1)).', matchedYIDx );
    if isempty(duplIDx), return; end
    
    % Generate a reduced target list with already matched points removed
    tempX = X; tempX(matchedXIDx, :) = [];
    tempXIDx = (1:size(X,1)).'; tempXIDx(matchedXIDx) = [];
    
    % Generate a reduced source list with already matched points removed
    tempY = Y; tempY(matchedYIDx, :) = [];
    % tempYIDx = (1:size(Y,1)).'; tempYIDx(matchedYIDx) = [];
    
    % For each remaining unmatched point, this returns the associated
    % unique ID of a point in the reduced target list
    repIDx = unique_knnsearch(tempX, tempY);
    idx(duplIDx) = tempXIDx(repIDx);
    
end

end

