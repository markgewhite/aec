function genPaperDataPlots( thisData, title, names )
    % Compile dataset plots for the paper
    arguments
        thisData        cell
        title           string
        names           strings
    end

    n = length( thisData );
    if n ~= length( names )
        error('The number of datasets and names do not match.');
    end
    
    for i = 1:n

        fig = plotDataset( thisData{i} );
        filename = strcat( title, "-", names(i), "-Data.pdf" );

        fig = formatIEEEFig( fig, ...
                             width = "Bespoke", ...
                             inches = 1.5, ...
                             size = "Medium", ...
                             keepAxisLabels = false, ...
                             keepAxisTicks = true, ...
                             keepTitle = true, ...
                             filename = filename );

    end

end