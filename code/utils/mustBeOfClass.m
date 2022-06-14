% ************************************************************************
% Function: mustBeOfClass
%
% Custom validator function (code from Matlab)
% Tests for a specific class name
%
% ************************************************************************

function mustBeOfClass( input, className )

    cname = class( input );
    if ~strcmp( cname, className )
        eid = 'Class:notCorrectClass';
        msg = ['Input must be of class ', cname];
        throwAsCaller( MException(eid, msg) )
    end

end