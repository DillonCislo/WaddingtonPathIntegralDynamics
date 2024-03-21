function [ dupl, C ] = find_duplicate_rows( A )
%FIND_DUPLICATE_ROWS Finds duplicate rows in a matrix
%
%   INPUT PARAMETERS:
%
%       - A:    MxN matrix
%
%   OUTPUT PARAMETERS:
%
%       - dupl: struct with fields "val" containing the value
%               of the duplicated row and "idx" containing the
%               row IDs in A corresponding to that value
%
%       - C:    uMxN matrix consisting of the unique rows of A

[C, ia, ~] = unique( A, 'rows' );

if size(A,1) == size(C,1)
    % disp('There are no duplicate rows!');
    dupl = {};
    return;
end

rep_idx = setdiff(1:size(A,1), ia);
rep_val = unique( A(rep_idx,:), 'rows');

dupl_val = cell( size(rep_val,1), 1 );
dupl_idx = cell( size(rep_val,1), 1 );

for i = 1:size(rep_val,1)
    
    dupl_val{i} = rep_val(i,:);
    
    diffA = A - repmat( rep_val(i,:), size(A,1), 1 );
    dupl_idx{i} = find( sqrt(sum(diffA .* conj(diffA), 2)) < eps );
    
end

dupl = struct( 'val', dupl_val, 'idx', dupl_idx );

end

