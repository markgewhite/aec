function results = investigationResults( names, path, ...
                                         parameters, values, setup, ...
                                         resume, catchErrors, memorySaving )

    thisInvestigation = Investigation( names, path, ...
                             parameters, values, setup, ...
                             resume, catchErrors, memorySaving );

    thisInvestigation.saveDataPlot;
    results = thisInvestigation.saveReport;

    thisInvestigation.conserveMemory( memorySaving );
    thisInvestigation.save;

end