function dlX = flattenDLArray( dlX )
    % Flatten a dl array (usually for input to dlnetwork
    arguments
        dlX     dlarray
    end

    CDimIdx = finddim( dlX, 'C' );
    SDimIdx = finddim( dlX, 'S' );
    BDimIdx = finddim( dlX, 'B' );
    
    dlX = reshape( dlX, ...
                   size(dlX,SDimIdx)*size(dlX,CDimIdx), ...
                   size(dlX,BDimIdx) );
    
    dlX = dlarray( dlX, 'CB' );

end