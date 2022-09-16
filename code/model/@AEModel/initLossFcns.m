function self = initLossFcns( self, setup )
    % Initialize the loss function objects
    arguments
        self        AEModel
        setup       struct
    end

    fldNames = fields( setup );
    nLossFcns = length( fldNames );
    names = string( nLossFcns );
    for i = 1:nLossFcns
        
        names(i) = setup.(fldNames{i}).name;
        try
            argsCell = namedargs2cell( setup.(fldNames{i}).args );
        catch
            argsCell = {};
        end

        % initialize the loss function object
        self.LossFcns.(names(i)) = setup.(fldNames{i}).class( names(i), argsCell{:} );

        % add details associated with the loss function networks
        % but without initializing them
        self = self.addLossFcnNetworks;

    end

    % formally record the names
    self.LossFcnNames = string(names);

    % add details associated with the loss function networks
    % but without initializing them
    self = self.addLossFcnNetworks;

    % store the loss functions' details 
    % and relevant details for easier access when training
    self = self.setLossInfoTbl;
    self.NumLoss = sum( self.LossFcnTbl.NumLosses.*self.LossFcnTbl.DoCalcLoss );

    self.LossFcnTbl.Types = categorical( self.LossFcnTbl.Types );
    self.LossFcnTbl.Inputs = categorical( self.LossFcnTbl.Inputs );

    % set loss function scaling factors if required
    self = self.setLossScalingFactor;

    % check a reconstruction loss is present
    if ~any( self.LossFcnTbl.Types=='Reconstruction' )
        eid = 'aeModel:NoReconstructionLoss';
        msg = 'No reconstruction loss object has been specified.';
        throwAsCaller( MException(eid,msg) );
    end 

    % check there is no more than one auxiliary network, if at all
    auxFcns = self.LossFcnTbl.Types=='Auxiliary';
    if sum( auxFcns ) > 1
        eid = 'FullAEModel:MultipleAuxiliaryFunction';
        msg = 'There is more than one auxiliary loss function.';
        throwAsCaller( MException(eid,msg) );
    end

    % identify the comparator network, if present
    comparatorFcns = self.LossFcnTbl.Types=='Comparator';
    if sum( comparatorFcns ) > 1
        eid = 'FullAEModel:MultipleComparatorFunction';
        msg = 'There is more than one comparator loss function.';
        throwAsCaller( MException(eid,msg) );
    end

end