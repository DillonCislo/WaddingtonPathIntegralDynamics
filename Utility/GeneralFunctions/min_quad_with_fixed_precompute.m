function F = min_quad_with_fixed_precompute(A, known, Aeq)
%MIN_QUAD_WITH_FIXED_PRECOMPUTE perform any necessary precomputation of
%system including factorization and preparation of right-hand side. This is
%a direct rip of the 'precompute' function within the 'min_quad_with_fixed'
%function included in 'gptoolbox'.
%
% F = min_quad_with_fixed_precompute(A,known,Aeq)
%
% Inputs:
%   A  n by n matrix of quadratic coefficients
%   known  #known list of indices to known rows in Z
%   Optional:
%     Aeq  m by n list of linear equality constraint coefficients
% Outputs:
%   F  struct containing all information necessary to solve a prefactored
%   system touching only B, Y, and optionally Beq

% number of rows
n = size(A,1);
% cache problem size
F = struct();
F.n = n;

if isempty(Aeq)
    Aeq = zeros(0,n);
end

assert(size(A,1) == n, ...
    'Rows of system matrix (%d) != problem size (%d)',size(A,1),n);
assert(size(A,2) == n, ...
    'Columns of system matrix (%d) != problem size (%d)',size(A,2),n);
assert(isempty(known) || min(size(known))==1, ...
    'known indices (size: %d %d) not a 1D list',size(known));
assert(isempty(known) || min(known) >= 1, ...
    'known indices (%d) < 1',min(known));
assert(isempty(known) || max(known) <= n, ...
    'known indices (%d) > problem size (%d)',max(known),n);
assert(n == size(Aeq,2), ...
    'Columns of linear constraints (%d) != problem size (%d)',size(Aeq,2),n);

% cache known
F.known = known;
% get list of unknown variables including lagrange multipliers
F.unknown = find(~sparse(1,known,true,1,n));

Auu = A(F.unknown,F.unknown);
% note that columns are in *original* order
F.Ak = A(F.known,:);

% determine if A(unknown,unknown) is symmetric and/or postive definite
sym_measure = max(max(abs(Auu - Auu')))/max(max(abs(Auu)));
%sym_measure = normest(Auu-Auu')./normest(Auu);
if sym_measure > eps
    % not very symmetric
    F.Auu_sym = false;
elseif sym_measure > 0
    % nearly symmetric but not perfectly
    F.Auu_sym = true;
else

    % Either Auu is empty or sym_measure should be perfect
    assert(isempty(sym_measure) || sym_measure == 0 || max(max(abs(Auu))) == 0,'not symmetric');
    % Perfectly symmetric
    F.Auu_sym = true;
end

% check if there are blank constraints
F.blank_eq = ~any(Aeq(:,F.unknown),2);
if any(F.blank_eq)
    warning('min_quad_with_fixed:blank_eq', [ ...
        'Removing blank constraints. ' ...
        'You ought to verify that known values satisfy contsraints']);
    Aeq = Aeq(~F.blank_eq,:);
end
% number of linear equality constraints
neq = size(Aeq,1);
%assert(neq <= n,'Number of constraints (%d) > problem size (%d)',neq,n);

% Determine if positive definite (also compute cholesky decomposition if it
% is as a side effect)
F.Auu_pd = false;
if F.Auu_sym && neq == 0
    % F.S'*Auu*F.S = F.L*F.L'
    if issparse(Auu)
        [F.L,p,F.S] = chol(Auu,'lower');
    else
        [F.L,p] = chol(Auu,'lower');
        F.S = eye(size(F.L));
    end
    F.Auu_pd = p==0;
end

% keep track of whether original A was sparse
A_sparse = issparse(A);

% Determine number of linearly independent constraints
if neq > 1 && ~(isfield(F,'force_Aeq_li') && ~isempty(F.force_Aeq_li)&& F.force_Aeq_li)
    %tic;
    % Null space substitution with QR
    [AeqTQ,AeqTR,AeqTE] = qr(Aeq(:,F.unknown)');
    nc = find(any(AeqTR,2),1,'last');
    if isempty(nc)
        nc = 0;
    end
    %fprintf('QR: %g secs\n',toc);
    assert(nc<=neq);
    F.Aeq_li = nc == neq;
else
    F.Aeq_li = true;
end
if neq > 0 && isfield(F,'force_Aeq_li') && ~isempty(F.force_Aeq_li)
    F.Aeq_li = F.force_Aeq_li;
end

% Use raw Lagrange Multiplier method only if rows of Aeq are Linearly
% Independent
if F.Aeq_li
    % get list of lagrange multiplier indices
    F.lagrange = n+(1:neq);
    if neq > 0
        if issparse(A) && ~issparse(Aeq)
            warning('min_quad_with_fixed:sparse_system_dense_constraints', ...
                'System is sparse but constraints are not, solve will be dense');
        end
        if issparse(Aeq) && ~issparse(A)
            warning('min_quad_with_fixed:dense_system_sparse_constraints', ...
                'Constraints are sparse but system is not, solve will be dense');
        end
        Z = sparse(neq,neq);
        % append lagrange multiplier quadratic terms
        A = [A Aeq';Aeq Z];
        %assert(~issparse(Aeq) || A_sparse == issparse(A));
    end
    % precompute RHS builders
    F.preY = A([F.unknown F.lagrange],known) + ...
        A(known,[F.unknown F.lagrange])';

    % LDL has a different solve prototype
    F.ldl = false;
    % create factorization
    if F.Auu_sym
        if neq == 0 && F.Auu_pd
            % we already have F.L
            F.U = F.L';
            F.P = F.S';
            F.Q = F.S;
        else
            % LDL is faster than LU for moderate #constraints < #unknowns
            NA = A([F.unknown F.lagrange],[F.unknown F.lagrange]);
            if issparse(NA)
                [F.L,F.D,F.P,F.S] = ldl(NA);
            else
                [F.L,F.D,F.P] = ldl(NA);
                F.S = eye(size(NA));
            end
            F.ldl = true;
        end
    else
        NA = A([F.unknown F.lagrange],[F.unknown F.lagrange]);
        % LU factorization of NA
        if issparse(NA)
            [F.L,F.U,F.P,F.Q] = lu(NA);
        else
            [F.L,F.U] = lu(NA);
            F.P = 1;
            F.Q = 1;
        end
    end
else
    % We alread have CTQ,CTR,CTE
    %tic;
    % Aeq' * AeqTE = AeqTQ * AeqTR
    % AeqTE' * Aeq = AeqTR' * AeqTQ'
    % Aeq x = Beq
    % Aeq (Q2 lambda + lambda_0) = Beq
    % we know Aeq Q2 = 0 --> Aeq Q2 lambda = 0
    % Aeq lambda_0 = Beq
    % AeqTE' * Aeq lambda_0 = AeqTE' * Beq
    % AeqTR' * AeqTQ' lambda_0 = AeqTE' * Beq
    % AeqTQ' lambda_0 = AeqTR' \ (AeqTE' * Beq)
    % lambda_0 = AeqTQ * (AeqTR' \ (AeqTE' * Beq))
    % lambda_0 = Aeq \ Beq;
    % lambda_0 = AeqTQ * (AeqTR' \ (AeqTE' * Beq));
    AeqTQ1 = AeqTQ(:,1:nc);
    AeqTR1 = AeqTR(1:nc,:);
    %lambda_0 = [AeqTQ1 * (AeqTR1' \ (AeqTE' * Beq))];
    %fprintf('lambda_0: %g secs\n',toc);
    %tic;
    % Substitute x = Q2 lambda + lambda_0
    % min 0.5 x' A x - x' b
    %   results in A x = b
    % min 0.5 (Q2 lambda + lambda_0)' A (Q2 lambda + lambda_0) - (Q2 lambda + lambda_0)' b
    % min 0.5 lambda' Q2' A Q2 lambda + lambda Q2' A lambda_0 - lambda Q2' b
    %  results in Q2' A Q2 lambda = - Q2' A lambda_0 + Q2' b
    AeqTQ2 = AeqTQ(:,(nc+1):end);
    QRAuu =  AeqTQ2' * Auu * AeqTQ2;
    %QRb = -AeqTQ2' * Auu * lambda_0 + AeqTQ2' * b;
    % precompute RHS builders
    F.preY = A(F.unknown,known) + A(known,F.unknown)';
    %fprintf('Proj: %g secs\n',toc);
    %tic;
    % QRA seems to be PSD
    if issparse(QRAuu)
        [F.L,p,F.S] = chol(QRAuu,'lower');
    else
        [F.L,p] = chol(QRAuu,'lower');
        F.S = eye(size(F.L));
    end
    F.U = F.L';
    F.P = F.S';
    F.Q = F.S;
    %fprintf('Chol: %g secs\n',toc);
    % Perhaps if Auu is not PD then we need to use LDL...
    assert(p==0);
    % WHICH OF THESE ARE REALLY NECESSARY?
    F.Aeq = Aeq;
    F.AeqTQ2 = AeqTQ2;
    F.AeqTQ1 = AeqTQ1;
    F.AeqTR1 = AeqTR1;
    F.AeqTE = AeqTE;
    F.Auu = Auu;
end
F.precomputed = true;

end
