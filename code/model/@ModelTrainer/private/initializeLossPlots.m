function [fig, lossLines] = initializeLossPlots( lossFcnTbl, show )
    % Setup plots for tracking progress
    arguments
        lossFcnTbl      table
        show            logical = true
    end
   
    lossFcnTbl = lossFcnTbl( lossFcnTbl.DoCalcLoss, : );
    nAxes = size( lossFcnTbl, 1 );
    [ rows, cols ] = sqdim( nAxes );
    
    % setup figure for plotting loss functions
    fig = getFigure( 3 );   
    if show
        fig.Visible = 'on';
    else
        fig.Visible = 'off';
    end
    nLines = sum( lossFcnTbl.NumLosses );

    layout = tiledlayout( fig, rows, cols, TileSpacing = 'Compact' );
    lossLines = gobjects( nLines, 1 );
    colours = lines( nLines );
    c = 0;
    for i = 1:nAxes

        thisName = lossFcnTbl.Names(i);
        axis = nexttile( layout );
        
        for k = 1:lossFcnTbl.NumLosses(i)
            c = c+1;
            lossLines(c) = animatedline( axis, 'Color', colours(c,:) );
        end
        
        title( axis, thisName );
        xlabel( axis, 'Iteration' );
        ylim( axis, lossFcnTbl.YLim{i} );

    end

end