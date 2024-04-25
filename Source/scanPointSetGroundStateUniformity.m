function [allScalarUniErr, allGroundStateEigs, allU0, allIsReducible] = ...
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
%       - X:            #N x dim set of input points -OR- a two element
%                       vector with X(1) == #N and X(2) = dim
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
%       - ('UseGPU', useGPU = false): Try to compute the groundstate
%       eigenvector on the GPU. This is just repeated matrix multiplication
%       from a random starting vector.
%
%       - ('MaxIterations', maxIter = 300 (no GPU) or maxIter = 1000
%       (GPU)): The number of iterations to use when computing the ground
%       state eigenvector (either with MATLAB's 'eigs' or through
%       GPU-accelerated iteration)
%
%       - ('Tolerance', tol = 1e-14): The covergence tolerance for the
%       ground state eigenvector comptuation
%
%       - ('EnsembleSize', ensembleSize = 1): For the simple
%       GPU-accelerated transition operator iteration method, we apply the
%       iteration to an ensemble of a user specified size and report
%       ensemble averages
%
%       - ('DistanceMatrix', distMatrix = []): #N x #N pairwise distance
%       matrix, i.e. distMatrix(i,j) is the distance between cell i and
%       cell j
%
%       - ('FixedPotential', U0 = []): #N x 1 scalar potential defined on
%       the input points. If this field is supplied, the potential is held
%       fixed over all time steps. If not, the potential is recalculated at
%       each iteration, using the currrent time step as a bandwidth for
%       density estimation
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
%       - allIsReducible:       1 x #T logical vector indicating whether or
%                               not the corresponding transition matrix is
%                               reducible. By the Perron-Frobenius theorem,
%                               irreducible Markov chains have unique
%                               stationary distributions.
%
%   by Dillon Cislo 2024/03/21

%--------------------------------------------------------------------------
% INPUT PROCESSING
%--------------------------------------------------------------------------
if (numel(X) == 2)
    numPoints = X(1); dim = X(2);
    validateattributes(numPoints, {'numeric'}, {'integer', ...
        'scalar', 'positive', 'finite', 'real'});
    validateattributes(dim, {'numeric'}, {'integer', ...
        'scalar', 'positive', 'finite', 'real'});
else
    validateattributes(X, {'numeric'}, {'2d', 'finite', 'real'});
    numPoints = size(X,1); % dim = size(X,2);
end

validateattributes(allTimeSteps, {'numeric'}, ...
    {'vector', 'positive', 'finite', 'real'});
if (size(allTimeSteps,1) ~= 1), allTimeSteps = allTimeSteps.'; end

% OPTIONAL INPUT PROCESSING -----------------------------------------------

D = 1;
D0 = 1;
useGPU = false;
maxIter = 300;
maxIterSet = false;
tol = 1e-14;
ensembleSize = 1;
distMatrix = [];
verbose = false;
fixU0 = [];

supportedOptions = {'DiffusionCoefficient', ...
    'PointDiffusionCoefficient', 'UseGPU', 'MaxIterations', ...
    'Tolerance', 'EnsembleSize', 'Verbose', 'DistanceMatrix', ...
    'FixedPotential'};
checkSupportedOptions(supportedOptions, varargin);

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

    if strcmpi(varargin{i}, 'UseGPU')
        useGPU = varargin{i+1};
        validateattributes(useGPU, {'logical'}, {'scalar'});
    end

    if strcmpi(varargin{i}, 'MaxIterations')
        maxIter = varargin{i+1};
        validateattributes(maxIter, {'numeric'}, ...
            {'scalar', 'integer', 'finite', 'real', '>', 1});
        maxIterSet = true;
    end

    if strcmpi(varargin{i}, 'Tolerance')
        tol = varargin{i+1};
        validateattributes(tol, {'numeric'}, ...
            {'scalar', 'positive', 'finite', 'real'});
    end

    if strcmpi(varargin{i}, 'EnsembleSize')
        ensembleSize = varargin{i+1};
        validateattributes(ensembleSize, {'numeric'}, ...
            {'scalar', 'integer', 'finite', 'positive', 'real'});
    end

    if strcmpi(varargin{i}, 'DistanceMatrix')
        distMatrix = varargin{i+1};
        if ~isempty(distMatrix)
            validateattributes(distMatrix, {'numeric'}, {'2d', ...
                'finite', 'real', 'nonnegative', 'square'});
            assert(size(distMatrix,1) == numPoints, ...
                'Distance matrix is improperly sized');
        end
    end

    if strcmpi(varargin{i}, 'FixedPotential')
        fixU0 = varargin{i+1};
        if ~isempty(fixU0)
            validateattributes(fixU0, {'numeric'}, {'vector', ...
                'finite', 'real', 'numel', numPoints});
            if (size(fixU0, 2) ~= 1), fixU0 = fixU0.'; end
        end
    end

    if strcmpi(varargin{i}, 'Verbose')
        verbose = varargin{i+1};
        validateattributes(verbose, {'logical'}, {'scalar'});
    end
    
end

if useGPU
    try
        gpuDevice;
        if ~maxIterSet, maxIter = 1000; end
    catch
        warning('Failed to find supported GPU device');
        useGPU = false;
    end
end

assert(~((numel(X) == 2) && isempty(distMatrix)), ['You have to supply ' ...
    'either a complete input point set or a distance matrix']);

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
allIsReducible = false(1, numel(allTimeSteps));

zeroTol = numPoints * eps(1);

for n = 1:numel(allTimeSteps)
    
    if verbose
      fprintf('Now processing length scale %d/%d = %0.3e\n', ...
          n, numel(allTimeSteps), allTimeSteps(n));
    end
    
    % Estimate Density From Point Cloud -----------------------------------
    
    if isempty(fixU0)

        if verbose
            disp('Performing kernel density estimation: ');
        end

        curDensity = gaussianKDE(X, X, [], ...
            sqrt(2 * D * allTimeSteps(n)), verbose, [], ...
            distMatrix, useGPU);

        U0 = -D0 * log(curDensity);

    else

        U0 = fixU0;

    end

    allU0(:, n) = U0;
    
    % Construct Fokker-Planck Transition Matrix ---------------------------
    
    if verbose, fprintf('Constructing transition matrix... '); end

    if (numel(X) == 2)
        T = computeTransitionMatrix([], U0, allTimeSteps(n), ...
            'DiffusionCoefficient', D, 'PointDiffusionCoefficient', D0, ...
            'ClipThreshold', 1e-14, 'StrictNormalization', true, ...
            'UseGPU', useGPU, 'DistanceMatrix', distMatrix);
    else
        T = computeTransitionMatrix(X, U0, allTimeSteps(n), ...
            'DiffusionCoefficient', D, 'PointDiffusionCoefficient', D0, ...
            'ClipThreshold', 1e-14, 'StrictNormalization', true, ...
            'UseGPU', useGPU);
    end
    
    if verbose, fprintf('Done\n'); end

    % Check if the transition matrix is irreducible
    G = T;
    G(G(:) < zeroTol) = 0;
    G = digraph(G);
    graphDistMatrix = distances(G, 'Method', 'unweighted');
    allIsReducible(n) = any(isinf(graphDistMatrix(:)));

    if verbose
        if allIsReducible(n)
            disp('Transition matrix is REDUCIBLE');
        else
            disp('Transition matrix is IRREDUCIBLE');
        end
    end
    
    % Compute Uniformity of Ground State ----------------------------------
    
    if allIsReducible(n)

        allScalarUniErr(n) = NaN;
        allGroundStateEigs(:, n) = nan(numPoints, 1);
        if verbose
            disp(['A reducible transition matrix has no well ' ...
                'defined stationary state. Skipping computation.']);
        end

        continue;
        
    end

    if verbose, fprintf('Calculating groundstate eigenvector... '); end
    
    if useGPU

        gpuT = gpuArray(T);
        gpuP = gpuArray(rand(numPoints, ensembleSize)+0.1);
        gpuP = gpuP ./ sum(gpuP, 1);
        gpuP = gpuT * gpuP;

        iter = 1;
        allConverged = false;

        while ~allConverged

            gpuPrevP = gpuP;
            gpuP = gpuT * gpuP;

            relDiff = abs(gpuP-gpuPrevP) ./ abs(gpuPrevP);
            relDiff(isinf(relDiff)) = NaN;
            relDiff = max(relDiff, [], 1);

            allConverged = all(relDiff < tol);

            iter = iter + 1;

            if (iter >= maxIter)
                warning(['\nFailed to converge after %d iterations. ' ...
                    'Max relative difference = %0.5e'], ...
                    iter, max(relDiff));
                break;
            end

        end

        eigV = gather(gpuP);
        scalarUniErr = abs(std(eigV, 0, 1) ./ mean(eigV, 1)) ;

        allScalarUniErr(n) = mean(scalarUniErr);
        allGroundStateEigs(:, n) = ...
            eigV(:, knnsearch(scalarUniErr.', mean(scalarUniErr)));
        
    else

        [eigV, ~] = eigs(T, 1, 'largestabs', ...
            'Tolerance', tol, 'MaxIterations', maxIter);

        allGroundStateEigs(:, n) = eigV(:,1);
        allScalarUniErr(n) = abs( std(eigV(:,1)) / mean(eigV(:,1)) );

    end

    if verbose, fprintf('Done\n'); end

end

end

