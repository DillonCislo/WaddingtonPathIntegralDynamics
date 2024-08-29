function [t, c0, c1, v] = fitLinearInterpParam(u, v0, v1, varargin)
%FITLINEARINTERPPARAM Fits the parameters of a linear interpolation between
%two vectors for t \in [0, 1]
%
%   mininimize ||u-v||^2 where v = t (v1 + c1) + (1-t) (v0 + c0)
%
%   INPUT PARAMETERS:
%
%       - u:    #Nx1 target vector
%       - v0:   #Nx1 initial state vector
%       - v1:   #Nx1 final state vector
%
%   OUTPUT PARAMETERS:
%
%       - t:    Linear interpolation parameter
%       - c0:   Scalar shift for initial state vector
%       - c1:   Scalar shift for final state vector
%       - v:    Optimized interpolated vector
%
%   by Dillon Cislo 2023/09/12

%--------------------------------------------------------------------------
% INPUT PROCESSING
%--------------------------------------------------------------------------

validateattributes(u, {'numeric'}, {'vector', 'finite', 'real'});
validateattributes(v0, {'numeric'}, {'vector', 'finite', 'real'});
validateattributes(v1, {'numeric'}, {'vector', 'finite', 'real'});

assert((numel(u) == numel(v0)) && (numel(u) == numel(v1)), ...
    'All input vectors must have the same number of elements');

if (size(u,2) ~= 1), u = u.'; end
if (size(v0,2) ~= 1), v0 = v0.'; end
if (size(v1,2) ~= 1), v1 = v1.'; end

% Optional Input Processing -----------------------------------------------

dispStyle = 'none';
maxIter = 1000;
noShift = false;
clipUnit = true;

for i = 1:length(varargin)

   if isa(varargin{i}, 'double'), continue; end
   if isa(varargin{i}, 'logical'), continue; end

   if strcmpi(varargin{i}, 'DisplayStyle')
       dispStyle = varargin{i+1};
       validateattributes(dispStyle, {'char'}, {'vector'});
   end

   if strcmpi(varargin{i}, 'MaxIterations')
       maxIter = varargin{i+1};
       validateattributes(maxIter, {'numeric'}, ...
           {'scalar', 'positive', 'finite', 'real'} );
   end

   if strcmpi(varargin{i}, 'NoShift')
       noShift = varargin{i+1};
       validateattributes(noShift, {'logical'}, {'scalar'});
   end

   if strcmpi(varargin{i}, 'ClipToUnitInterval')
       clipUnit = varargin{i+1};
       validateattributes(clipUnit, {'logical'}, {'scalar'});
   end

end

%--------------------------------------------------------------------------
% MINIMIZATION PROCESSING
%--------------------------------------------------------------------------

if noShift

    t = dot((u-v0), (v1-v0)) ./ dot((v1-v0), (v1-v0));
    if clipUnit, t = min(max(t, 0), 1); end
    c0 = 0; c1 = 0;

else

    % Energy function
    fun = @interpEnergy;

    % Initial guess
    x0 = [0.5, 0, 0];

    if clipUnit

        % Upper and lower bounds
        lb = [0, -Inf, -Inf];
        ub = [1, Inf, Inf];

        % Inequality constraints
        A = [];
        b = [];

        % Equality constraints
        Aeq = [];
        beq = [];

        % Nonlinear constraints
        nonlcon = [];

        options = optimoptions( 'fmincon', ...
            'Algorithm', 'interior-point', ...
            'ConstraintTolerance', 1e-6, ...
            'SpecifyObjectiveGradient', true, ...
            'CheckGradients', false, ...
            'MaxIterations', maxIter, ...
            'HessianApproximation', 'bfgs', ...
            'Display', dispStyle );

        x = fmincon( fun, x0, A, b, Aeq, beq, lb, ub, nonlcon, options );

    else

        options = optimoptions( 'fminunc', ...
            'Algorithm', 'quasi-newton', ...
            'SpecifyObjectiveGradient', true, ...
            'CheckGradients', false, ...
            'MaxIterations', maxIter, ...
            'HessianApproximation', 'bfgs', ...
            'Display', dispStyle );

        x = fminunc( fun, x0, options );

    end

    % Format output variables
    t = x(1); c0 = x(2); c1 = x(3);

end

% Compute interpolated vector
v = t * (v1 + c1) + (1-t) * (v0 + c0);

%--------------------------------------------------------------------------
% ENERGY FUNCTION
%--------------------------------------------------------------------------

    function [E, EG] = interpEnergy ( x )
        
        T = x(1); C0 = x(2); C1 = x(3);
        V = T * (v1 + C1) + (1-T) * (v0 + C0);
        
        d = u - V;
        E = dot(d, d);
        
        if (nargout > 1)
            
            EG = zeros(1,3);
            EG(1) = -2 * dot(d, (v1+C1)-(v0+C0));
            EG(2) = -2 * (1-T) * sum(d);
            EG(3) = -2 * T * sum(d);
            
        end

    end

end

