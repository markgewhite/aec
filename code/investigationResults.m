function results = investigationResults( names, path, ...
                                         parameters, values, setup, ...
                                         memorySaving, resume )

    thisInvestigation = Investigation( names, path, ...
                             parameters, values, setup, resume );

    results = thisInvestigation.getResults;

    thisDataset = thisInvestigation.getDatasets;

    fig = thisDataset.plot;

    thisInvestigation.conserveMemory( memorySaving );

    thisInvestigation.save;

end