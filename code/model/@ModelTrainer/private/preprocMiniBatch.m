function [ X, XN, P, Y ] = preprocMiniBatch( XCell, XNCell, PCell, YCell, ICell, ...
                                          padValue, padLoc )
    % Preprocess a sequence batch for training

    X = padData( XCell, 0, padValue, Longest = true, Location = padLoc  );

    % extract the X normalized array
    if ~isempty( XNCell )
        XN = cat( 2, XNCell{:} );   
    else
        XN = [];
    end

    % extract the P distribution array
    if ~isempty( PCell )

        P = cat( 1, PCell{:} );
        % check if P is square - if so, it is being used
        pSize = size( P );
        if pSize(1)==pSize(2) && pSize(1)>1
            % extract the I indexing array
            if ~isempty( ICell )
                I = cat( 2, ICell{:} );
                % square the P array by filtering on columns using index
                P = P(:,I);
                % ensure p values sum to one
                P = max(P ./ sum(P(:)), realmin);
            end
        end

    else
        P = [];
    end

    % extract the Y label array
    if ~isempty( YCell )
        Y = cat( 2, YCell{:} );
    else
        Y = [];
    end

end