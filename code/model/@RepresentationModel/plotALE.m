function plotALE( self, args )
    % Update the Accumulated Local Effects plot
    arguments
        self           RepresentationModel
        args.quantiles      double = []
        args.pts            double = []
        args.type           char ...
            {mustBeMember( args.type, {'Network', 'Model'})} = 'Network'
        args.axis           = []
    end

    switch args.type
        case 'Network'
            A = self.AuxNetworkALE;
            axis = self.Axes.AuxNetwork;
            name = 'Auxiliary Network';
        case 'Model'
            A = self.AuxModelALE;
            axis = self.Axes.AuxModel;
            name = 'Auxiliary Model';
    end

    if ~isempty(args.pts)
        A = args.pts;
    end

    if isempty(args.quantiles)
        Q = self.ALEQuantiles;
    else
        Q = args.quantiles;
    end

    if ~isempty(args.axis)
        axis = args.axis;
    end

    nCodes = size( A, 1 );
    hold( axis, 'off');
    for i = 1:nCodes

        plot( axis, Q, A(i,:), 'LineWidth', 1 );
        hold( axis, 'on' );

    end
    
    hold( axis, 'off');
    
    title( axis, name );
    xlim( axis, [0,1] );
    xlabel( axis, 'Quantiles' );
    ylabel( axis, 'ALE' );   
    axis.YAxis.TickLabelFormat = '%.2f';
    
    finalisePlot( axis, square = true );
    
end

