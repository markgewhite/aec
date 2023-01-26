function genPaperCompPlots( path, title, names, nDims, nReports, nModels ) 
    % Compile component plots for the paper, re-formatting saved figures
    arguments
        path        string
        title       string
        names       string
        nDims       double {mustBeInteger, mustBePositive}
        nReports    double {mustBeInteger, mustBePositive}
        nModels     double {mustBeInteger, mustBePositive}
    end

    for i = 1:nReports
        for d = 1:nDims
            for j = 1:nModels

                if nDims==1 && nModels==1
                    if nDims==1
                        evalCode = strcat(num2str(j));
                    else
                        evalCode = strcat(num2str(d));
                    end
                else
                    evalCode = strcat(num2str(j), ",", num2str(d));
                end

                figpath = strcat( path, "/", names(i), "/comp/figs/" );
            
                figfilename = strcat( names(i), "(", evalCode, ").fig" );
                fig = openfig( fullfile(figpath, figfilename) );
                filename = strcat( title, "-", names(i), "-Comp(", ...
                    num2str(j), ",", num2str(d), ").pdf" );

                nComp = fig.Children.GridSize(2);
    
                fig = formatElsevierFig( fig, ...
                                     sizeType = "Custom", ...
                                     width = nComp*3.5+2, ...
                                     keepXAxisLabels = false, ...
                                     keepYAxisLabels = false, ...
                                     keepLegend = false, ...
                                     filename = filename );
    
    
            end
        end
    end

end