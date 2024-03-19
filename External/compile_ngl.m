function compile_ngl(verbose)
% NOTE: You must have 'CMake' installed in order to compile NGL!
% by Dillon Cislo 

% Basic Input Processing --------------------------------------------------
if (nargin < 1), verbose = false; end

% Navigate to the NGL directory
[nglDir, ~, ~] = fileparts(mfilename("fullpath"));
nglDir = fullfile(nglDir, 'ngl-beta');
cd(nglDir);

% Check if the directory is on the path
nglOnPath = contains(path, nglDir);
if nglOnPath, rmpath(genpath(nglDir)); end

% Compile the ANN Library -------------------------------------------------

annDir = fullfile(nglDir, 'ann_1.1.2');
cd(annDir);

if ispc
    
    warning(['You have to compile the ANN yourself on Windows. ' ...
        'Please navigate ngl-beta/ann_1.1.2/doc and read the ' ...
        'documentation.']);
    
else
    
    [status, cmdout] = system('make realclean');
    if (status == 0)
        if verbose
            disp(['''make realclean'' executed successfully. ' ...
                'Command output:']);
            disp(cmdout);
        end
    else
        if verbose
            errorMessage = sprintf(['''make realclean'' execution' ...
                'failed! Command output:\n%s'], cmdout);
        else
            errorMessage = '''make realclean'' execution failed!';
        end
        disp(' ');
        error(errorMessage);
        
    end
    
    if isunix
        
        [status, cmdout] = system('make linux-g++');
        if (status == 0)
            if verbose
                disp(' ');
                disp(['''make linux-g++'' executed successfully. ' ...
                    'Command output:']);
                disp(cmdout);
            end
        else
            if verbose
                errorMessage = sprintf(['''make linux-g++'' execution ' ...
                    'failed! Command output:\n%s'], cmdout);
            else
                errorMessage = '''make linux-g++'' execution failed!';
            end
            disp(' ');
            error(errorMessage);
        end
        
    elseif ismac
        
        [status, cmdout] = system('make mac-g++');
        if (status == 0)
            if verbose
                disp(' ');
                disp(['''make mac-g++'' executed successfully. ' ...
                    'Command output:']);
                disp(cmdout);
            end
        else
            if verbose
                errorMessage = sprintf(['''make mac-g++'' execution ' ...
                    'failed! Command output:\n%s'], cmdout);
            else
                errorMessage = '''make mac-g++'' execution failed!';
            end
            disp(' ');
            error(errorMessage);
        end
        
    else
        
        error('You don''t appear to have a 64-bit OS');
        
    end
    
end


% Compile the NGL Library -------------------------------------------------

if ispc   
    warning(['NGL compilation has NOT been tested on Windows. ' ...
        'Good luck!']);   
end

buildDir = fullfile(nglDir, 'build');
if exist(buildDir, 'dir'), rmdir(buildDir, 's'); end
mkdir(buildDir);
cd(buildDir);

[status, cmdout] = system('cmake ../');
if (status == 0)
    if verbose
        disp(' ');
        disp('''cmake'' executed successfully. Command output:');
        disp(cmdout);
    end
else
    if verbose
        errorMessage = sprintf(['''cmake'' execution failed! ' ...
            'Command output:\n%s'], cmdout);
    else
        errorMessage = '''cmake'' execution failed!';
    end
    disp(' ');
    error(errorMessage);
end

[status, cmdout] = system('make');
if (status == 0)
    if verbose
        disp(' ');
        disp('''make'' executed successfully. Command output:');
        disp(cmdout);
    end
else
    if verbose
        errorMessage = sprintf(['''make'' execution failed! ' ...
            'Command output:\n%s'], cmdout);
    else
        errorMessage = '''make'' execution failed!';
    end
    disp(' ');
    error(errorMessage);
end

[status, cmdout] = system('make install');
if (status == 0)
    if verbose
        disp(' ');
        disp('''make install'' executed successfully. Command output:');
        disp(cmdout);
    end
else
    if verbose
        errorMessage = sprintf(['''make install'' execution failed! ' ...
            'Command output:\n%s'], cmdout);
    else
        errorMessage = '''make install'' execution failed!';
    end
    disp(' ');
    error(errorMessage);
end
 
if nglOnPath, addpath(genpath(nglDir)); end
    
end

