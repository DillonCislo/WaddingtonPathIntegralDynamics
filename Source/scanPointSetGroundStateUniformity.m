function [allScalarUniErr, allGroundStateEigs, allU0] = ...
    scanPointSetGroundStateUniformity(X, allTimeSteps, varargin)
%SCANPOINTSETGROUNDSTATEUNIFORMITY Assess the uniformity of the ground
%state of the transition matrix constructed from a point set
%pseudopotential across a user supplied set of time steps. NOTE: You are
%allowed to set the various diffusion coefficients for the sake of
%generalizability, but we STRONGLY recommend you just leave them all equal
%to one.
%
%   INPUT PARAMETERS:
%
%       - X:    #N x dim set of input points
%
%       - allTimeSteps:	1 x #T vector of candidate short time steps over
%                       which the transition matrix approximates the
%                       drift-diffusion dynamics
%
%   OPTIONAL INPUT PARAMETERS (Name, Value)-Pairs:
%
%       - ('DiffusionCoefficient', D = 1): The diffusion coefficient for
%       the dynamical drift-diffusion process
%
%       - ('PointDiffusionCoefficient', D0 = 1): The diffusion coefficient
%       for the drift-diffusion process assumed to generate the input point
%       set
%
%       - ('Verbose', verbose = false): Whether or not to produce verbose
%       progress output
%
%   OUTPUT PARAMETERS:
%
%       - allScalarUniErr:      1 x #T vector of the ratio of the standard
%                               deviation of the transition matrix ground
%                               state eigenvector to its mean as a function
%                               of the user supplied time steps. Smaller
%                               values indicate greater uniformity.
%
%       - allGroundStateEigs:   #N x #T matrix of transition matrix ground
%                               state eigenvectors
%
%       - allU0:                #N x #T matrix of inferred point set
%                               pseudopotentials computed as a function of
%                               the user supplied time steps
%
%   by Dillon Cislo 2024/03/21

%--------------------------------------------------------------------------
% INPUT PROCESSING
%--------------------------------------------------------------------------
validateattributes(X, {'numeric'}, {'2d', 'finite', 'real'});
numPoints = size(X,1); % dim = size(X,2);

validateattributes(allTimeSteps, {'numeric'}, ...
    {'vector', 'positive', 'finite', 'real'});
if (size(allTimeSteps,1) ~= 1), allTimeSteps = allTimeSteps.'; end

% OPTIONAL INPUT PROCESSING -----------------------------------------------

D = 1;
D0 = 1;
verbose = false;

for i = 1:length(varargin)
    
    if isa(varargin{i}, 'double'), continue; end
    if isa(varargin{i}, 'logical'), continue; end
    
    if strcmpi(varargin{i}, 'DiffusionCoefficient')
        D = varargin{i+1};
        validateattributes(D, {'numeric'}, ...
            {'scalar', 'positive', 'finite', 'real'});
    end
    
    if strcmpi(varargin{i}, 'PointDiffusionCoefficient')
        D0 = varargin{i+1};
        validateattributes(D0, {'numeric'}, ...
            {'scalar', 'positive', 'finite', 'real'});
    end
    
    if strcmpi(varargin{i}, 'Verbose')
        verbose = varargin{i+1};
        validateattributes(verbose, {'logical'}, {'scalar'});
    end
    
end

%--------------------------------------------------------------------------
% SCAN GROUND STATE UNIFORMITY ACROSS TIME STEPS
%--------------------------------------------------------------------------

if verbose
    disp(['Determining uniformity of ground state as ' ...
        'a function of D * dt:']);
end

allU0 = zeros(numPoints, numel(allTimeSteps));
allGroundStateEigs = zeros(numPoints, numel(allTimeSteps));
allScalarUniErr = zeros(1, numel(allTimeSteps));

for n = 1:numel(allTimeSteps)
    
    if verbose
      fprintf('Now processing length scale %d/%d\n', ...
          n, numel(allTimeSteps));
    end
    
    % Estimate Density From Point Cloud -----------------------------------
    
    if verbose
       disp('Performing kernel density estimation: ');
    end
    
    curDensity = gaussianKDE(X, X, [], ...
        sqrt(2 * D * allTimeSteps(n)), verbose);
    
    U0 = -D0 * log(curDensity);
    allU0(:, n) = U0;
    
    % Construct Fokker-Planck Transition Matrix ---------------------------
    
    if verbose, fprintf('Constructing transition matrix... '); end
    
    T = computeTransitionMatrix(X, U0, allTimeSteps(n), ...
        'DiffusionCoefficient', D, 'PointDiffusionCoefficient', D0);
    
    if verbose, fprintf('Done\n'); end
    
    % Compute Uniformity of Ground State ----------------------------------
    
    if verbose, fprintf('Calculating groundstate eigenvector... '); end
    
    
    [eigV, ~] = eigs(T, 5, 'largestabs');
    allGroundStateEigs(:, n) = eigV(:,1);
    allScalarUniErr(n) = abs( std(eigV(:,1)) / mean(eigV(:,1)) );
    
    if verbose, fprintf('Done\n'); end

end

end

