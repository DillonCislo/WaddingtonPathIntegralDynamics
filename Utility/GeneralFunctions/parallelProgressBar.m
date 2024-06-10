function parallelProgressBar(curIter, maxIter)
%PARALLELPROGRESSBAR A progress bar for parfor loops. Unfortunately, this
%function needs a little bit of set up outside/within the loop in order to
%work properly. A working example is:
%
%   DQ = parallel.pool.DataQueue;
%   afterEach(DQ, @parallelProgressBar)
%
%   N = 200;
%   parallelProgressBar(0, N);
%
%   v = nan(N,1);
%   parfor i = 1:N
%       v(i) = max(abs(eig(rand(200))));
%       send(DQ, []);
%   end
%
%   clear DQ parallelProgressBar % Clears persistent variables
%
%   By Dillon Cislo 2023/12/06

persistent parCurIter parMaxIter

if (nargin == 2)
    
    parCurIter = curIter;
    parMaxIter = maxIter;
    
    try 
        progressbar(parCurIter, parMaxIter)
    catch
        fprintf('2 arg: %d/%d\n', parCurIter, parMaxIter);
    end 
    
else
    
    try 
        progressbar(parCurIter, parMaxIter)
    catch
        fprintf('0 arg: %d/%d\n', parCurIter, parMaxIter);
    end 
    
    parCurIter = parCurIter + 1;
    
end

end