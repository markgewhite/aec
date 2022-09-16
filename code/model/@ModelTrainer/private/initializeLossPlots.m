function [fig, lossLines] = initializeLossPlots( lossFcnTbl )
    % Setup plots for tracking progress
   
    lossFcnTbl = lossFcnTbl( lossFcnTbl.DoCalcLoss, : );
    nAxes = size( lossFcnTbl, 1 );
    [ rows, cols ] = sqdim( nAxes );
    
    % setup figure for plotting loss functions
    fig = figure(3);
    clf;
    nLines = sum( lossFcnTbl.NumLosses );
    lossLines = gobjects( nLines, 1 );
    colours = lines( nLines );
    c = 0;
    for i = 1:nAxes

        thisName = lossFcnTbl.Names(i);
        axis = subplot( rows, cols, i );
        
        for k = 1:lossFcnTbl.NumLosses(i)
            c = c+1;
            lossLines(c) = animatedline( axis, 'Color', colours(c,:) );
        end
        
        title( axis, thisName );
        xlabel( axis, 'Iteration' );

        switch lossFcnTbl.Types(i)
            case 'Reconstruction'
                ylim( axis, [0 0.25] );
            case 'Regularization'
                ylim( axis, [0 1.5] );
            case {'Auxiliary', 'Comparator'}
                ylim( axis, [0 1.5] );
        end

    end

end