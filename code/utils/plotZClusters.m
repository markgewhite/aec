function plotZClusters( thisModel, Z, args )
    % Plot the latent codes on 2D plane
    arguments
        thisModel           {mustBeA( thisModel, ...
                                { 'FullRepresentationModel', ...
                                  'CompactRepresentationModel' })}
        Z                   {mustBeA( Z, { 'dlarray', 'double' })}
        args.Y              = []
        args.type           char ...
            {mustBeMember(args.type, ...
                {'Canonical', 'TSNE'} )} = 'TSNE'
        args.name           string = 'Latent Space'
        args.perplexity     double = 50
        args.compact        logical = false

    end

    if isa( Z, 'dlarray' )
        Z = double( extractdata( Z ) )';
    end
    % ensure Z is 2D
    Z = reshape( Z, size(Z,1), [] );

    if isempty( args.Y ) && strcmp( args.type, 'Canonical' )
        eid = 'aeModel:LabelsMissing';
        msg = 'Canonical discriminant analysis needs labels.';
        throwAsCaller( MException(eid,msg) );
    end

    if ~isempty( args.Y )
        Y = args.Y;
        if isa( args.Y, 'dlarray' )
            Y = double( extractdata( Y ) );
        end
        classes = unique( Y );
    else
        Y = ones( size( Z,1 ), 1 );
        classes = 1;
    end           

    switch args.type
        case 'Canonical'
            % canonical discriminant analysis                  
            ZCanInfo = cda( Z, Y );
            ZT = ZCanInfo.scores;
            if size( ZT, 2 )==1
                % only one canonical dimension
                ZT = [ ZT ZT ];  
            end

        case 'TSNE'
            % t-distribution stochastic neighbour embedding
            perplexity = min( size(Z,1), args.perplexity );
            ZT = tsne( Z, ...
                       Perplexity = perplexity, ...
                       Standardize = true ); 
            
    end

    axis = thisModel.Axes.ZClustering;
    cla( axis, 'reset' );
    hold( axis, 'on' );
    
    if args.compact
        dotSize = 5;
    else
        dotSize = 10;
    end
    
    % plot true classes (large dots)
    colours = lines( length(classes) );
    gscatter( axis, ZT(:,1), ZT(:,2), Y, colours, '.', dotSize );
    
    hold( axis, 'off' );
    
    if ~args.compact
        legend( axis, 'Location', 'Best' );
    end

    title( axis, args.name );
    axis.TickDir = 'none';
    axis.XTickLabel = [];
    axis.YTickLabel = [];
    finalisePlot( axis, square = true );


end