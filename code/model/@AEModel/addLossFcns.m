function self = addLossFcns( self, args )
    % Add one or more loss function objects to the model
    arguments
        self            AEModel
        args.weights    double {mustBeNumeric,mustBeVector} = 1
    end

    nFcns = length( self.LossFcns );

    % check the weights
    if args.weights==1
        % default is to assign a weight of 1 to all functions
        w = ones( nFcns, 1 );
    elseif length( args.weights ) ~= nFcns
        % weights don't correspond to the functions
        eid = 'aeModel:WeightsMismatch';
        msg = 'Number of assigned weights does not match number of functions';
        throwAsCaller( MException(eid,msg) );
    else
        w = args.weights;
    end
    self.LossFcnWeights = [ self.LossFcnWeights w ];

    % update the names list
    self.LossFcnNames = [ self.LossFcnNames getFcnNames(self.LossFcns) ];


    % add details associated with the loss function networks
    % but without initializing them
    self = self.addLossFcnNetworks;

    % store the loss functions' details 
    % and relevant details for easier access when training
    self = self.setLossInfoTbl;
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