function genPaperDataPlots( path, title, names )
    % Compile dataset plots for the paper
    arguments
        path            string
        title           string
        names           string
    end

    nDatasets = length( names );
    for i = 1:nDatasets

        figFilename = strcat( names(i), "-InvestigationData" );
        fig = openfig( fullfile(path, "figs", figFilename) );

        newFilename = strcat( title, "-", names(i), "-Data Plot.pdf" );

        formatElsevierFig( fig, ...
                       sizeType = "Minimal", ...
                       keepLegend = false, ...
                       keepXAxisLabels = false, ...
                       keepYAxisLabels = false, ...
                       filename = newFilename );

    end

end