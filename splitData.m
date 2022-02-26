% ************************************************************************
% Function: splitData
% ************************************************************************

function XS = splitData( X, indices )

if iscell( X )
    XS = X( indices );
else
    XS = X( :, indices, : );
end

end
