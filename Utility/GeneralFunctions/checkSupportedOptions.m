function checkSupportedOptions(supportedOptions, suppliedOptions)
%CHECKSUPPORTEDOPTIONS Checks that all (name, value)-pair options in a user
%supplied list are a valid subset of the user supplied options
%
%   INPUT PARAMETERS:
%
%       - supportedOptions:     Cell vector of character vectors
%                               listing all supported options for the
%                               current function
%
%       - suppliedOptions:      Cell vector of user supplied options as a
%                               set of (name, value)-pairs. Values can have
%                               arbitrary type (including character
%                               vectors)
%
%   by Dillon Cislo 2024/03/28

if isempty(suppliedOptions), return; end
validateattributes(supportedOptions, {'cell'}, {'vector'});
validateattributes(suppliedOptions, {'cell'}, {'vector'});

% Determine which elements of the supplied options are character vectors
isCharVec = cellfun(@(x) ischar(x) && isvector(x), ...
    suppliedOptions, 'UniformOutput', true);

% Determine which of the supplied character vectors are supported options
isOption = false(1, numel(suppliedOptions));
isOption(isCharVec) = ismember(lower(suppliedOptions(isCharVec)), ...
    lower(supportedOptions));

% Check the options were supplied as (name, value)-pairs
assert((mod(numel(suppliedOptions), 2) == 0), ...
    'Options do not appear to be supplied as (name, value)-pairs');

% Any character vector that is not a valid option name must be the value
% corresponding to a valid option name
goodParam = isOption | (~isCharVec);
if all(goodParam), return; end

badParamIDx = setdiff(find(~goodParam), 1);
goodParam(badParamIDx) = isOption(badParamIDx-1);

if ~all(goodParam)

    errString = cellfun(@(x) ['''' x ''', '], ...
        suppliedOptions(~goodParam), 'UniformOutput', false);
    errString = ['Unsupported options supplied: ' errString{:}];
    errString = errString(1:(end-2));

    error(errString);

end

end

