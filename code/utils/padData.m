function XP = padData( X, padLen, padValue, args )
    % Pad time series to a specified length
    arguments
        X               cell             % arrays of time series
        padLen          double           % padding length (ignored if longest)
        padValue        double           % padding value (ignored if same)
        args.Longest    logical = false  % whether to pad to the longest
        args.Same       logical = false  % whether to pad with same value
        args.Location     char {mustBeMember( args.Location, ...
            {'Left', 'Right', 'Both', 'Symmetric'} )} = 'Left' % padding location
        args.Anchoring    char {mustBeMember( args.Anchoring, ...
            {'None', 'Left', 'Right', 'Both'} )} = 'None' % anchoring location
    end
    
    len = cellfun( @length, X );
    allSameLength = (std(len)==0);
    if args.Longest
        padLen = max( len );
    end
       
    nObs = length( X );
    nDim = size( X{1}, 2 );
    
    XP = single(zeros( padLen, nObs, nDim ));
    
    for i = 1:nObs
    
        if allSameLength
            XP( :, i, : ) = X{i};

        else
            trialLen = min( [ size(X{i}, 1), padLen] );
        
            if args.Same
                xStart = X{i}(1,:);
                xEnd = X{i}(end,:);
            else
                xStart = repelem( padValue, 1, nDim );
                xEnd = repelem( padValue, 1, nDim );
            end
        
            switch args.Location
    
                case 'Left' 
                    % insert padding at the beginning
                    padLeft = ones( padLen-trialLen, 1 )*xStart;
                    padRight = [];
        
                case 'Right'
                    % insert padding at the end
                    padLeft = [];
                    padRight = ones( padLen-trialLen, 1 )*xEnd;
        
                case 'Both'
                    % insert padding at both ends, roughly evenly
                    startLen = fix( (padLen-trialLen)/2 );
                    padLeft = ones( startLen, 1 )*xStart;
                    padRight = ones( padLen-trialLen-startLen, 1 )*xEnd;
        
                case 'Symmetric'
                    % insert padding at both ends as mirror image of opposite end
                    startLen = fix( (padLen-trialLen)/2 );
                    padLeft = X{i}( end-startLen+1:end, : );
                    padRight = X{i}( 1:startLen, : );
    
            end
    
            XP( :, i, : ) = [ padLeft; ...
                                X{i}(end - trialLen+1:end, :); ...
                                  padRight ];

        end

    end

    % for anchoring set the same value at limits across the dataset
    switch args.Anchoring

        case 'Left'
            XP( 1, :, : ) = repmat( mean( XP(1, :, :), 2 ), nObs, 1 );

        case 'Right'
            XP( end, :, : ) = repmat( mean( XP(end, :, :), 2 ), nObs, 1 );

        case 'Both'
            XP( 1, :, : ) = repmat( mean( XP(1, :, :), 2 ), nObs, 1 );
            XP( end, :, : ) = repmat( mean( XP(end, :, :), 2 ), nObs, 1 );

    end
    
end

