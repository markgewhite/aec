% ************************************************************************
% Function: mustBeValidBasis
%
% Custom validator function (code from Matlab)
% Check if functional data object matches initialized parameters
%
% ************************************************************************

function mustBeValidBasis( obj, fd )
    
    if not(getbasis( fd ) == obj.basisFd)
        eid = 'Basis:inValidBasis';
        msg = 'The functional basis does not match the initialized basis';
        throwAsCaller(MException(eid,msg))
    end

end