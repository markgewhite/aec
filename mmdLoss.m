% ************************************************************************
% Function: mmdLoss
%
% Compute the Maximum Mean Discrepancy between two samples (distributions)
%
% Parameters:
%           

% Outputs:
%           setup : initialised setup structure
%
% ************************************************************************

function stat = mmdLoss( dlZQ, dlZP, setup )

ZQ = double(extractdata( dlZQ ) )';
ZP = double(extractdata( dlZP ) )';

sigma2_p = setup.scale^2;

[ nObs, zDim ] = size( ZQ );
% half_size = (n*n - n)/2;

%norms_pz = sum( sample_pz.^2, 2 );
%dotprods_pz = sample_pz.sample_pz';
%distances_pz = norms_pz + norms_pz' - 2*dotprods_pz;
distPPZ = pdist2( ZP, ZP ).^2;

%norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=1, keep_dims=True)
%dotprods_qz = tf.matmul(sample_qz, sample_qz, transpose_b=True)
%distances_qz = norms_qz + tf.transpose(norms_qz) - 2. * dotprods_qz
distQQZ = pdist2( ZQ, ZQ ).^2;

%dotprods = tf.matmul(sample_qz, sample_pz, transpose_b=True)
%distances = norms_qz + tf.transpose(norms_pz) - 2. * dotprods
distPQZ = pdist2( ZQ, ZP ).^2;

switch setup.kernel
    case 'RBF'
    % Median heuristic for the sigma^2 of Gaussian kernel
    %        sigma2_k = tf.nn.top_k(
    %            tf.reshape(distances, [-1]), half_size).values[half_size - 1]
    %        sigma2_k += tf.nn.top_k(
    %            tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
    %        # Maximal heuristic for the sigma^2 of Gaussian kernel
    %        # sigma2_k = tf.nn.top_k(tf.reshape(distances_qz, [-1]), 1).values[0]
    %        # sigma2_k += tf.nn.top_k(tf.reshape(distances, [-1]), 1).values[0]
    %        # sigma2_k = opts['latent_space_dim'] * sigma2_p
    %        if opts['verbose']:
    %            sigma2_k = tf.Print(sigma2_k, [sigma2_k], 'Kernel width:')
    %        res1 = tf.exp( - distances_qz / 2. / sigma2_k)
    %        res1 += tf.exp( - distances_pz / 2. / sigma2_k)
    %        res1 = tf.multiply(res1, 1. - tf.eye(n))
    %        res1 = tf.reduce_sum(res1) / (nf * nf - nf)
    %        res2 = tf.exp( - distances / 2. / sigma2_k)
    %        res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
    %        stat = res1 - res2
    case 'IMQ'
        % k(x, y) = C / (C + ||x - y||^2)
        % C = tf.nn.top_k(tf.reshape(distances, [-1]), half_size).values[half_size - 1]
        % C += tf.nn.top_k(tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
        switch setup.baseType
            case 'Normal'
                Cbase = 2*zDim*sigma2_p;
            case 'Sphere'
                Cbase = 2;
            case 'Uniform'
                Cbase = zDim;
            otherwise
                error('Unrecognised C base type')
        end
        stat = 0;
        scale = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0];
        mask = ones(nObs) - eye(nObs);
        for i = 1:length(scale)
            
            C = Cbase*scale(i);
            
            res1 = C./(C + distQQZ);
            res1 = res1 + C./(C + distPPZ);
            res1 = res1.*mask;
            res1 = sum( res1, 'all' )/(nObs*nObs-nObs);

            res2 = C./(C + distPQZ);
            res2 = sum( res2, 'all' )*2/(nObs*nObs);

            stat = stat + res1 - res2;

        end

end