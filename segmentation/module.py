from __future__ import division
import tensorflow as tf
from ops import conv3d,deconv3d,group_norm,dense1d,condition_instance_norm, lrelu, flip_gradient
import numpy as np
import random



# =============================================================================================================
# SYNTHESIS PART.

def latentEncodeNet(input_code, options, reuse=False, name="latentEncodeNet"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        hc1 = dense1d(input_code, options.gf_dim, name="ls_h1_dense1d")
        # h1 is self.df_dim, for c1 in encoder.
        hc2 = dense1d(hc1, options.gf_dim*2, name="ls_h2_dense1d")
        # h2 is self.df_dim * 2, for c2 in encoder.
        hc3 = dense1d(hc2, options.gf_dim*4, name="ls_h3_dense1d")
        # h3 is self.df_dim * 4, for c3 in encoder.
        hr11 = dense1d(hc3, options.gf_dim*4, name="ls_h4_dense1d")
        # h4 is self.df_dim * 4, for r11 in encoder.
        hr12 = dense1d(hr11, options.gf_dim*4, name="ls_h5_dense1d")
        # h5 is self.df_dim * 4, for r12 in encoder.
        hr21 = dense1d(hr12, options.gf_dim*4, name="ls_h6_dense1d")
        # h6 is self.df_dim * 4, for r21 in encoder.
        hr22 = dense1d(hr21, options.gf_dim*4, name="ls_h7_dense1d")
        # h7 is self.df_dim * 4, for r22 in encoder.
        hr31 = dense1d(hr22, options.gf_dim*4, name="ls_h8_dense1d")
        # h8 is self.df_dim * 4, for r31 in encoder.
        hr32 = dense1d(hr31, options.gf_dim*4, name="ls_h9_dense1d")
        # h9 is self.df_dim * 4, for r32 in encoder.
        hr41 = dense1d(hr32, options.gf_dim*4, name="ls_h10_dense1d")
        # h10 is self.df_dim * 4, for r41 in encoder.
        hr42 = dense1d(hr41, options.gf_dim*4, name="ls_h11_dense1d")
        # h11 is self.df_dim * 4, for r41 in encoder.
        hr51 = dense1d(hr42, options.gf_dim*4, name="ls_h12_dense1d")
        # h12 is self.df_dim * 4, for r41 in encoder.
        hr52 = dense1d(hr51, options.gf_dim*4, name="ls_h13_dense1d")
        # h13 is self.df_dim * 4, for r41 in encoder.
        hr61 = dense1d(hr52, options.gf_dim*4, name="ls_h14_dense1d")
        # h12 is self.df_dim * 4, for r41 in encoder.
        hr62 = dense1d(hr61, options.gf_dim*4, name="ls_h15_dense1d")
        # h13 is self.df_dim * 4, for r41 in encoder.
        return {'hc1':hc1, 'hc2':hc2, 'hc3':hc3,
                'hr11':hr11, 'hr12':hr12, 
                'hr21':hr21, 'hr22':hr22,
                'hr31':hr31, 'hr32':hr32, 
                'hr41':hr41, 'hr42':hr42,
                'hr51':hr51, 'hr52':hr52,
                'hr61':hr61, 'hr62':hr62}
     
        
def latentDecodeNet(input_code, options, reuse=False, name="latentDecodeNet"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        hd2 = dense1d(input_code, options.gf_dim, name="ls_h1_dense1d")
        # h1 is self.df_dim, for d2 in encoder.
        hd1 = dense1d(hd2, options.gf_dim*2, name="ls_h2_dense1d")
        # h2 is self.df_dim * 2, for d1 in encoder.
        hr122 = dense1d(hd1, options.gf_dim*4, name="ls_h3_dense1d")
        # h3 is self.df_dim * 4, for r92 in encoder.
        hr121 = dense1d(hr122, options.gf_dim*4, name="ls_h4_dense1d")
        # h4 is self.df_dim * 4, for r91 in encoder.
        hr112 = dense1d(hr121, options.gf_dim*4, name="ls_h5_dense1d")
        # h3 is self.df_dim * 4, for r92 in encoder.
        hr111 = dense1d(hr112, options.gf_dim*4, name="ls_h6_dense1d")
        # h4 is self.df_dim * 4, for r91 in encoder.
        hr102 = dense1d(hr111, options.gf_dim*4, name="ls_h7_dense1d")
        # h3 is self.df_dim * 4, for r92 in encoder.
        hr101 = dense1d(hr102, options.gf_dim*4, name="ls_h8_dense1d")
        # h4 is self.df_dim * 4, for r91 in encoder.
        hr92 = dense1d(hr101, options.gf_dim*4, name="ls_h9_dense1d")
        # h3 is self.df_dim * 4, for r92 in encoder.
        hr91 = dense1d(hr92, options.gf_dim*4, name="ls_h10_dense1d")
        # h4 is self.df_dim * 4, for r91 in encoder.
        hr82 = dense1d(hr91, options.gf_dim*4, name="ls_h11_dense1d")
        # h5 is self.df_dim * 4, for r82 in encoder.
        hr81 = dense1d(hr82, options.gf_dim*4, name="ls_h12_dense1d")
        # h6 is self.df_dim * 4, for r81 in encoder.
        hr72 = dense1d(hr81, options.gf_dim*4, name="ls_h13_dense1d")
        # h7 is self.df_dim * 4, for r72 in encoder.
        hr71 = dense1d(hr72, options.gf_dim*4, name="ls_h14_dense1d")
        # h8 is self.df_dim * 4, for r71 in encoder.
        return {'hd2':hd2, 'hd1':hd1,
                'hr121':hr121, 'hr122':hr122, 
                'hr111':hr111, 'hr112':hr112, 
                'hr101':hr101, 'hr102':hr102, 
                'hr91':hr91, 'hr92':hr92, 
                'hr81':hr81, 'hr82':hr82,
                'hr71':hr71, 'hr72':hr72}


def encoder_resnet(image, scale, offset, options, reuse=False, name="encoder"):
    
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, scale1, offset1, scale2, offset2, dim, ks=3, s=1, name='encoder_res'):
            pad = int((ks-1)/2)
            
            y = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')
            y = condition_instance_norm(conv3d(y, dim, ks, s, paddings='VALID', name=name+'_c1'), scale1, offset1, name+'_bn1')
            
            y = tf.pad(tf.nn.relu(y), [[0, 0], [pad, pad], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')
            y = condition_instance_norm(conv3d(y, dim, ks, s, paddings='VALID', name=name+'_c2'), scale2, offset2, name+'_bn2')
            return y + x

        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
        c1 = tf.nn.relu(condition_instance_norm(conv3d(c0, options.gf_dim, 7, 1, paddings='VALID', name='encoder_e1_c'), 
                                                scale['hc1'], offset['hc1'], 'encoder_e1_bn'))
        c2 = tf.nn.relu(condition_instance_norm(conv3d(c1, options.gf_dim*2, 3, 2, name='encoder_e2_c'), 
                                                scale['hc2'], offset['hc2'], 'encoder_e2_bn'))
        c3 = tf.nn.relu(condition_instance_norm(conv3d(c2, options.gf_dim*4, 3, 2, name='encoder_e3_c'), 
                                                scale['hc3'], offset['hc3'], 'encoder_e3_bn'))
        # define G network with 9 resnet blocks
        r1 = residule_block(c3, scale['hr11'], offset['hr11'], 
                                scale['hr12'], offset['hr12'], options.gf_dim*4, name='encoder_r1')
        r2 = residule_block(r1, scale['hr21'], offset['hr21'], 
                                scale['hr22'], offset['hr22'], options.gf_dim*4, name='encoder_r2')
        r3 = residule_block(r2, scale['hr31'], offset['hr31'], 
                                scale['hr32'], offset['hr32'], options.gf_dim*4, name='encoder_r3')
        r4 = residule_block(r3, scale['hr41'], offset['hr41'], 
                                scale['hr42'], offset['hr42'], options.gf_dim*4, name='encoder_r4')
        r5 = residule_block(r4, scale['hr51'], offset['hr51'], 
                                scale['hr52'], offset['hr52'], options.gf_dim*4, name='encoder_r5')
        r6 = residule_block(r5, scale['hr61'], offset['hr61'], 
                                scale['hr62'], offset['hr62'], options.gf_dim*4, name='encoder_r6')
        
        return r6


def decoder_resnet(r6, scale, offset, options, reuse=False, name="decoder"):
    
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, scale1, offset1, scale2, offset2, dim, ks=3, s=1, name='decoder_res'):
            pad = int((ks-1)/2)
            
            y = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')
            y = condition_instance_norm(conv3d(y, dim, ks, s, paddings='VALID', name=name+'_c1'), scale1, offset1, name+'_bn1')
            
            y = tf.pad(tf.nn.relu(y), [[0, 0], [pad, pad], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')
            y = condition_instance_norm(conv3d(y, dim, ks, s, paddings='VALID', name=name+'_c2'), scale2, offset2, name+'_bn2')
            return y + x

        r7 = residule_block(r6,   scale['hr71'],  offset['hr71'], 
                                  scale['hr72'],  offset['hr72'],  options.gf_dim*4, name='decoder_r6')
        r8 = residule_block(r7,   scale['hr81'],  offset['hr81'], 
                                  scale['hr82'],  offset['hr82'],  options.gf_dim*4, name='decoder_r5')
        r9 = residule_block(r8,   scale['hr91'],  offset['hr91'], 
                                  scale['hr92'],  offset['hr92'],  options.gf_dim*4, name='decoder_r4')
        r10 = residule_block(r9,  scale['hr101'], offset['hr101'], 
                                  scale['hr102'], offset['hr102'], options.gf_dim*4, name='decoder_r3')
        r11 = residule_block(r10, scale['hr111'], offset['hr111'], 
                                  scale['hr112'], offset['hr112'], options.gf_dim*4, name='decoder_r2')
        r12 = residule_block(r11, scale['hr121'], offset['hr121'], 
                                  scale['hr122'], offset['hr122'], options.gf_dim*4, name='decoder_r1')
        
        d1 = deconv3d(r12, options.gf_dim*2, 3, 2, name='decoder_d1_dc')
        d1 = tf.nn.relu(condition_instance_norm(d1, scale['hd1'], offset['hd1'], 'decoder_d1_bn'))
        
        d2 = deconv3d(d1, options.gf_dim, 3, 2, name='decoder_d2_dc')
        d2 = tf.nn.relu(condition_instance_norm(d2, scale['hd2'], offset['hd2'], 'decoder_d2_bn'))
        
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
        pred = tf.nn.tanh(conv3d(d2, options.image_dim, 7, 1, paddings='VALID', name='decoder_pred_c'))
        
        return pred


## Attention-based GCN.
def fusenet(ec1, ec2, ec3, ec4, indcode, input_code, options, reuse=False, name="fusenet"):
    
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        # ============================================================================================
        # reshape one-hot code.
        real_code_all = tf.reshape(input_code, (1, 1, 1, 1, options.n_domains, options.n_domains))
        real_code_all = tf.tile(real_code_all, (options.batch_size, 
                                                options.crop_size//4, 
                                                options.crop_size//4, 
                                                options.crop_size//4, 1, 1))
        
        # ============================================================================================
        # reshape encoded feature.
        ec_all = tf.stack((ec1, ec2, ec3, ec4), axis=5)
        
        ec_all_app = tf.concat([ec_all, real_code_all], 4)
        
        # ============================================================================================
        # reshape indicate code.
        ind_all_ = tf.expand_dims(indcode, 1)
        
        indcode_all = tf.tile(indcode, (1, 
                                        options.crop_size//4, 
                                        options.crop_size//4, 
                                        options.crop_size//4, 1))

        # ============================================================================================
        # message collection process.
        # for missing contrast j, remove mji when compute ri, based on indicate code of j.
        f_list = []
        
        weight_ji = onelayerMLP(ec_all_app[:,:,:,:,:,0], 
                                ec_all_app[:,:,:,:,:,0], 
                                indcode_all, options, reuse=reuse, name="onelayerMLP")
        
        for i in range(options.n_domains):
            
            for j in range(options.n_domains):
                
                # for fi: steal weight_wji of fj.
                weight_ji = onelayerMLP(ec_all_app[:,:,:,:,:,j], 
                                        ec_all_app[:,:,:,:,:,i], 
                                        indcode_all, options, reuse=True, name="onelayerMLP")
                
                if j == 0:
                    ftemp = tf.multiply(tf.multiply(weight_ji, ec_all[:,:,:,:,:,j]) + ec_all[:,:,:,:,:,i], 
                                        ind_all_[:,:,:,:,:,j])
                else:
                    ftemp = tf.add(ftemp, tf.multiply(tf.multiply(weight_ji, ec_all[:,:,:,:,:,j]) + ec_all[:,:,:,:,:,i], 
                                                      ind_all_[:,:,:,:,:,j]))
                                                      
            ftemp = tf.divide(ftemp, tf.reduce_sum(ind_all_, 5))
            f_list.append(ftemp)

        # ============================================================================================
        # modifying features.
        fused_encode = tf.reduce_sum(tf.multiply(tf.stack(f_list, axis=5), ind_all_), axis=5)
        fused_encode = tf.divide(fused_encode, tf.reduce_sum(ind_all_, axis=5))

        return fused_encode


def onelayerMLP(fj, fi, indinfo, options, reuse=False, name="onelayerMLP"):
    
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        
        h0 = tf.concat([fj, fi, indinfo], 4)
        # h0 is (B x H x W x self.df_dim*4*2)
        h1 = tf.nn.relu(conv3d(h0, options.gf_dim*6, ks=1, s=1, name='c_h1_conv'))
        # h1 is (B x H x W x self.df_dim*4)
        h2 = tf.nn.tanh(conv3d(h1, options.gf_dim*4, ks=1, s=1, name='c_h2_conv'))
        # h1 is (B x H x W x self.df_dim*4)
        
        return h2


def classifer(r6, options, reuse=False, name="classifer"):

    with tf.variable_scope(name):
        # image is 18 x 18 x 18 x self.df_dim*4
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        h0 = flip_gradient(r6, l=1.0, name='c_h0_fgrad')
        # h0 is (64 x 48 x self.df_dim*4)
        h1 = lrelu(conv3d(h0, options.df_dim*2, ks=1, s=1, name='c_h1_conv'))
        # h1 is (64 x 48 x self.df_dim*2)
        h2 = lrelu(conv3d(h1, options.df_dim, ks=1, s=1, name='c_h2_conv'))
        # h2 is (64 x 48 x self.df_dim)
        h3 = lrelu(conv3d(h2, options.df_dim/3, ks=1, s=1, name='c_h3_conv'))
        # h3 is (64 x 48 x self.df_dim/4)
        h4 = conv3d(h3, options.n_domains, ks=1, s=1, name='c_h4_conv')
        # h4 is (64 x 48 x 3)
        return h4

# =============================================================================================================





# =============================================================================================================
# SEGMENTATION PART.
def latentCoarseSegScaleNet(input_code, options, reuse=False, name="latentCoarseSegScaleNet"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        her1 = dense1d(input_code, options.sf_dim, name="ls_h1_dense1d")
        # h1 is self.df_dim, for e12 in segmentor.
        her22 = dense1d(her1, options.sf_dim*2, name="ls_h2_dense1d")
        # h2 is self.df_dim * 2, for e22 in segmentor.
        hes32 = dense1d(her22, options.sf_dim*4, name="ls_h3_dense1d")
        # h3 is self.df_dim * 4, for e32 in segmentor.
        her32 = dense1d(hes32, options.sf_dim*4, name="ls_h4_dense1d")
        # h4 is self.df_dim * 4, for e32 in segmentor.
        her44 = dense1d(her32, options.sf_dim*8, name="ls_h5_dense1d")
        # h5 is self.df_dim * 8, for e42 in segmentor.
        hdr3 = dense1d(her44, options.sf_dim*4, name="ls_h6_dense1d")
        # h6 is self.df_dim * 16, for e52 in segmentor.
        hdr2 = dense1d(hdr3, options.sf_dim*2, name="ls_h7_dense1d")
        # h7 is self.df_dim * 8, for d42 in segmentor.
        hdr1 = dense1d(hdr2, options.sf_dim, name="ls_h8_dense1d")
        # h8 is self.df_dim * 4, for d32 in segmentor.
        return {'her1': her1,  'hdr3':hdr3,
                'her22':her22, 'hdr2':hdr2, 
                'hes32':hes32, 'hdr1':hdr1,
                'her32':her32,
                'her44':her44}


def coarseSegmentor(image, latentscale, encode_fused, options, reuse=False, name="coarseSegmentor"):
    
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        
        def residule_block(x, dim, ks=3, s=1, name='csegmentor_res'):
            y = tf.nn.relu(group_norm(x, name=name+'_bn1'))
            y = conv3d(y, dim, ks, s, name=name+'_c1')
            
            y = tf.nn.relu(group_norm(y, name=name+'_bn2'))
            y = conv3d(y, dim, ks, s, name=name+'_c2')
            return y + x

        def residule_modblock(x, ls, dim, ks=3, s=1, name='csegmentor_res'):
            y = tf.nn.relu(group_norm(x, name=name+'_bn1'))
            y = conv3d(y, dim, ks, s, name=name+'_c1')
            
            y = tf.nn.relu(group_norm(y, name=name+'_bn2'))
            y = tf.multiply(conv3d(y, dim, ks, s, name=name+'_c2'), ls)
            return y + x

        # image is (256 x 256 x input_c_dim)
        e11 = conv3d(image, options.sf_dim, ks=3, s=1, name='cs_e11_conv')
        # e11 is (256 x 256 x self.sf_dim)
#        drop1 = tf.nn.dropout(e11, dropout_rate)
        # drop4 is (256 x 256 x self.sf_dim)
        er1 = residule_modblock(e11, latentscale['her1'], options.sf_dim, name='cs_er1')
        # er1 is (256 x 256 x self.sf_dim)
        e12 = conv3d(er1, options.sf_dim*2, ks=3, s=2, name='cs_e12_conv')
        # e12 is (128 x 128 x self.sf_dim*2)
        
        er21 = residule_block(e12, options.sf_dim*2, name='cs_er21')
        # er21 is (128 x 128 x self.sf_dim*2)
        er22 = residule_modblock(er21, latentscale['her22'], options.sf_dim*2, name='cs_er22')
        # er22 is (128 x 128 x self.sf_dim*2)
        e21 = conv3d(er22, options.sf_dim*4, ks=3, s=2, name='cs_e21_conv')
        # e21 is (64 x 64 x self.sf_dim*4)

        # features from image translation.
        es31 = conv3d(encode_fused, options.sf_dim*4, ks=3, s=1, name='cs_es31_conv')
        # es31 is (64 x 64 x self.gf_dim*4)
        es32 = tf.multiply(conv3d(es31, options.sf_dim*4, ks=3, s=1, name='cs_es32_conv'), latentscale['hes32'])
        # es32 is (64 x 64 x self.gf_dim*4)
        
        e30p1 = tf.concat([e21, es32], 4)
        # e31 is (64 x 64 x self.gf_dim*4)
        e30p2 = conv3d(e30p1, options.sf_dim*4, ks=3, s=1, name='cs_e30p2_conv')
        # es32 is (64 x 64 x self.gf_dim*4)
        
        er31 = residule_block(e30p2, options.sf_dim*4, name='cs_er31')
        # er31 is (64 x 64 x self.sf_dim*4)
        er32 = residule_modblock(er31, latentscale['her32'], options.sf_dim*4, name='cs_er32')
        # er32 is (64 x 64 x self.sf_dim*4)
        e31 = conv3d(er32, options.sf_dim*8, ks=3, s=2, name='cs_e31_conv')
        # e31 is (32 x 32 x self.sf_dim*8)
        
        er41 = residule_block(e31, options.sf_dim*8, name='cs_er41')
        # er41 is (32 x 32 x self.sf_dim*8)
        er42 = residule_block(er41, options.sf_dim*8, name='cs_er42')
        # er42 is (32 x 32 x self.sf_dim*8)
        er43 = residule_block(er42, options.sf_dim*8, name='cs_er43')
        # er43 is (32 x 32 x self.sf_dim*8)
        er44 = residule_modblock(er43, latentscale['her44'], options.sf_dim*8, name='cs_er44')
        # er44 is (32 x 32 x self.sf_dim*8)

        d31 = conv3d(er44, options.sf_dim*4, ks=1, s=1, name='cs_d31_conv')
        # d31 is (32 x 32 x self.sf_dim*8)
        d32 = deconv3d(d31, options.sf_dim*4, ks=2, s=2, name='cs_d32_dc')
        # d32 is (64 x 64 x self.sf_dim*4)
        merge3 = tf.add(er32, d32)
        # merge3 is (64 x 64 x self.gf_dim*4)
        dr3 = residule_modblock(merge3, latentscale['hdr3'], options.sf_dim*4, name='cs_dr3')
        # dr3 is (64 x 64 x self.sf_dim*4)
        
        d21 = conv3d(dr3, options.sf_dim*2, ks=1, s=1, name='cs_d21_conv')
        # d21 is (64 x 64 x self.sf_dim*4)
        d22 = deconv3d(d21, options.sf_dim*2, ks=2, s=2, name='cs_d22_dc')
        # d22 is (128 x 128 x self.sf_dim*2)
        merge2 = tf.add(er22, d22)
        # merge2 is (128 x 128 x self.gf_dim*2)
        dr2 = residule_modblock(merge2, latentscale['hdr2'], options.sf_dim*2, name='cs_dr2')
        # dr2 is (128 x 128 x self.sf_dim*2)
        
        d11 = conv3d(dr2, options.sf_dim, ks=1, s=1, name='cs_d11_conv')
        # d11 is (128 x 128 x self.sf_dim*2)
        d12 = deconv3d(d11, options.sf_dim, ks=2, s=2, name='cs_d12_dc')
        # d12 is (256 x 256 x self.sf_dim)
        merge1 = tf.add(er1, d12)
        # merge1 is (256 x 256 x self.gf_dim)
        dr1 = residule_modblock(merge1, latentscale['hdr1'], options.sf_dim, name='cs_dr1')
        # dr1 is (256 x 256 x self.sf_dim)

        final_output = conv3d(dr1, options.output_c_dim, ks=1, s=1, name='cs_fn_conv')

        return tf.nn.sigmoid(final_output)


def latentFineSegScaleNet(input_code, options, reuse=False, name="latentFineSegScaleNet"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        her1 = dense1d(input_code, options.sf_dim*2, name="ls_h1_dense1d")
        # h1 is self.df_dim, for e12 in segmentor.
        her22 = dense1d(her1, options.sf_dim*4, name="ls_h2_dense1d")
        # h2 is self.df_dim * 2, for e22 in segmentor.
        hes32 = dense1d(her22, options.sf_dim*8, name="ls_h3_dense1d")
        # h3 is self.df_dim * 4, for e32 in segmentor.
        her32 = dense1d(hes32, options.sf_dim*8, name="ls_h4_dense1d")
        # h4 is self.df_dim * 4, for e32 in segmentor.
        her44 = dense1d(her32, options.sf_dim*16, name="ls_h5_dense1d")
        # h5 is self.df_dim * 8, for e42 in segmentor.
        
        hp1dr3 = dense1d(her44, options.sf_dim*8, name="ls_h6_dense1d")
        # h6 is self.df_dim * 16, for e52 in segmentor.
        hp1dr2 = dense1d(hp1dr3, options.sf_dim*4, name="ls_h7_dense1d")
        # h7 is self.df_dim * 8, for d42 in segmentor.
        hp1dr1 = dense1d(hp1dr2, options.sf_dim*2, name="ls_h8_dense1d")
        # h8 is self.df_dim * 4, for d32 in segmentor.
        
        hp2dr3 = dense1d(her44, options.sf_dim*8, name="ls_h9_dense1d")
        # h6 is self.df_dim * 16, for e52 in segmentor.
        hp2dr2 = dense1d(hp2dr3, options.sf_dim*4, name="ls_h10_dense1d")
        # h7 is self.df_dim * 8, for d42 in segmentor.
        hp2dr1 = dense1d(hp2dr2, options.sf_dim*2, name="ls_h11_dense1d")
        # h8 is self.df_dim * 4, for d32 in segmentor.
        
        return {'her1': her1,  'hp1dr3':hp1dr3, 'hp2dr3':hp2dr3,
                'her22':her22, 'hp1dr2':hp1dr2, 'hp2dr2':hp2dr2,
                'hes32':hes32, 'hp1dr1':hp1dr1, 'hp2dr1':hp2dr1,
                'her32':her32,
                'her44':her44}
                

def fineSegmentor(image, latentscale, encode_fused, options, reuse=False, name="fineSegmentor"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        
        def residule_block(x, dim, ks=3, s=1, name='fsegmentor_res'):
            y = tf.nn.relu(group_norm(x, name=name+'_bn1'))
            y = conv3d(y, dim, ks, s, name=name+'_c1')
            
            y = tf.nn.relu(group_norm(y, name=name+'_bn2'))
            y = conv3d(y, dim, ks, s, name=name+'_c2')
            return y + x

        def residule_modblock(x, ls, dim, ks=3, s=1, name='fsegmentor_res'):
            y = tf.nn.relu(group_norm(x, name=name+'_bn1'))
            y = conv3d(y, dim, ks, s, name=name+'_c1')
            
            y = tf.nn.relu(group_norm(y, name=name+'_bn2'))
            y = tf.multiply(conv3d(y, dim, ks, s, name=name+'_c2'), ls)
            return y + x
        
        # image is (256 x 256 x input_c_dim)
        e11 = conv3d(image, options.sf_dim*2, ks=3, s=1, name='fs_e11_conv')
        # e11 is (256 x 256 x self.sf_dim*2)
#        drop1 = tf.nn.dropout(e11, dropout_rate)
        # drop4 is (256 x 256 x self.sf_dim*2)
        er1 = residule_modblock(e11, latentscale['her1'], options.sf_dim*2, name='fs_er1')
        # er1 is (256 x 256 x self.sf_dim*2)
        e12 = conv3d(er1, options.sf_dim*4, ks=3, s=2, name='fs_e12_conv')
        # e12 is (128 x 128 x self.sf_dim*4)
        
        er21 = residule_block(e12, options.sf_dim*4, name='fs_er21')
        # er21 is (128 x 128 x self.sf_dim*4)
        er22 = residule_modblock(er21, latentscale['her22'], options.sf_dim*4, name='fs_er22')
        # er22 is (128 x 128 x self.sf_dim*4)
        e21 = conv3d(er22, options.sf_dim*8, ks=3, s=2, name='fs_e21_conv')
        # e21 is (64 x 64 x self.sf_dim*8)
        
        # features from image translation.
        es31 = conv3d(encode_fused, options.sf_dim*8, ks=3, s=1, name='fs_es31_conv')
        # es31 is (64 x 64 x self.gf_dim*4)
        es32 = tf.multiply(conv3d(es31, options.sf_dim*8, ks=3, s=1, name='fs_es32_conv'), latentscale['hes32'])
        # es32 is (64 x 64 x self.gf_dim*4)
        
        e30p1 = tf.concat([e21, es32], 4)
        # e31 is (64 x 64 x self.gf_dim*4)
        e30p2 = conv3d(e30p1, options.sf_dim*8, ks=3, s=1, name='fs_e30p2_conv')
        # es32 is (64 x 64 x self.gf_dim*4)
        
        er31 = residule_block(e30p2, options.sf_dim*8, name='fs_er31')
        # er31 is (64 x 64 x self.sf_dim*8)
        er32 = residule_modblock(er31, latentscale['her32'], options.sf_dim*8, name='fs_er32')
        # er32 is (64 x 64 x self.sf_dim*8)
        e31 = conv3d(er32, options.sf_dim*16, ks=3, s=2, name='fs_e31_conv')
        # e31 is (32 x 32 x self.sf_dim*16)
        
        er41 = residule_block(e31, options.sf_dim*16, name='fs_er41')
        # er41 is (32 x 32 x self.sf_dim*16)
        er42 = residule_block(er41, options.sf_dim*16, name='fs_er42')
        # er42 is (32 x 32 x self.sf_dim*16)
        er43 = residule_block(er42, options.sf_dim*16, name='fs_er43')
        # er43 is (32 x 32 x self.sf_dim*16)
        er44 = residule_modblock(er43, latentscale['her44'], options.sf_dim*16, name='fs_er44')
        # er44 is (32 x 32 x self.sf_dim*16)

        # PATH 1: ConTranspose.
        p1d31 = conv3d(er44, options.sf_dim*8, ks=1, s=1, name='fs_p1d31_conv')
        # d31 is (32 x 32 x self.sf_dim*16)
        p1d32 = deconv3d(p1d31, options.sf_dim*8, ks=2, s=2, name='fs_p1d32_dc')
        # d32 is (64 x 64 x self.sf_dim*8)
        p1merge3 = tf.add(er32, p1d32)
        # merge3 is (64 x 64 x self.gf_dim*8*2)
        p1dr3 = residule_modblock(p1merge3, latentscale['hp1dr3'], options.sf_dim*8, name='fs_p1dr3')
        # dr3 is (64 x 64 x self.sf_dim*8)
        
        p1d21 = conv3d(p1dr3, options.sf_dim*4, ks=1, s=1, name='fs_p1d21_conv')
        # d21 is (64 x 64 x self.sf_dim*8)
        p1d22 = deconv3d(p1d21, options.sf_dim*4, ks=2, s=2, name='fs_p1d22_dc')
        # d22 is (128 x 128 x self.sf_dim*4)
        p1merge2 = tf.add(er22, p1d22)
        # merge2 is (128 x 128 x self.gf_dim*4*2)
        p1dr2 = residule_modblock(p1merge2, latentscale['hp1dr2'], options.sf_dim*4, name='fs_p1dr2')
        # dr2 is (128 x 128 x self.sf_dim*4)
        
        p1d11 = conv3d(p1dr2, options.sf_dim*2, ks=1, s=1, name='fs_p1d11_conv')
        # d11 is (128 x 128 x self.sf_dim*4)
        p1d12 = deconv3d(p1d11, options.sf_dim*2, ks=2, s=2, name='fs_p1d12_dc')
        # d12 is (256 x 256 x self.sf_dim*2)
        p1merge1 = tf.add(er1, p1d12)
        # merge1 is (256 x 256 x self.gf_dim*2*2)
        p1dr1 = residule_modblock(p1merge1, latentscale['hp1dr1'], options.sf_dim*2, name='fs_p1dr1')
        # dr1 is (256 x 256 x self.sf_dim*2)
        
        p1output = conv3d(p1dr1, options.output_c_dim, ks=1, s=1, name='fs_p1fn_conv')

        # PATH 2: Upsampling.
        p2d31 = conv3d(er44, options.sf_dim*8, ks=1, s=1, name='fs_p2d31_conv')
        # d31 is (32 x 32 x self.sf_dim*8)
        p2d32 = tf.keras.layers.UpSampling3D(size=(2, 2, 2), data_format='channels_last')(p2d31)
        # d32 is (64 x 64 x self.sf_dim*8)
        p2merge3 = tf.add(er32, p2d32)
        # merge3 is (64 x 64 x self.gf_dim*8)
        p2dr3 = residule_modblock(p2merge3, latentscale['hp2dr3'], options.sf_dim*8, name='fs_p2dr3')
        # dr3 is (64 x 64 x self.sf_dim*8)
        
        p2d21 = conv3d(p2dr3, options.sf_dim*4, ks=1, s=1, name='fs_p2d21_conv')
        # d21 is (64 x 64 x self.sf_dim*4)
        p2d22 = tf.keras.layers.UpSampling3D(size=(2, 2, 2), data_format='channels_last')(p2d21)
        # d22 is (128 x 128 x self.sf_dim*4)
        p2merge2 = tf.add(er22, p2d22)
        # merge2 is (128 x 128 x self.gf_dim*4)
        p2dr2 = residule_modblock(p2merge2, latentscale['hp2dr2'], options.sf_dim*4, name='fs_p2dr2')
        # dr2 is (128 x 128 x self.sf_dim*4)
        
        p2d11 = conv3d(p2dr2, options.sf_dim*2, ks=1, s=1, name='fs_p2d11_conv')
        # d11 is (128 x 128 x self.sf_dim*2)
        p2d12 = tf.keras.layers.UpSampling3D(size=(2, 2, 2), data_format='channels_last')(p2d11)
        # d12 is (256 x 256 x self.sf_dim*2)
        p2merge1 = tf.add(er1, p2d12)
        # merge1 is (256 x 256 x self.gf_dim*2)
        p2dr1 = residule_modblock(p2merge1, latentscale['hp2dr1'], options.sf_dim*2, name='fs_p2dr1')
        # dr1 is (256 x 256 x self.sf_dim*2)

        p2output = conv3d(p2dr1, options.output_c_dim, ks=1, s=1, name='fs_p2fn_conv')

        return tf.nn.sigmoid(p1output), tf.nn.sigmoid(p2output)

# =============================================================================================================



def pabs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target), axis=[1,2,3])


def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))


def dice_criterion(logits, labels):
    
    eps = tf.cast(1e-5, tf.float32)
    
    logits = tf.cast(logits, tf.float32)
    labels = tf.cast(labels, tf.float32)
    
    inter_multi = tf.reduce_sum(tf.multiply(logits, labels))
    
    sum_class_1 = tf.reduce_sum(tf.multiply(logits, logits))
    sum_class_2 = tf.reduce_sum(tf.multiply(labels, labels))
    
    dice_coe = (tf.cast(2, tf.float32) * inter_multi) / (sum_class_1 + sum_class_2 + eps)
    
    return dice_coe


def prod_input_code(n_domains, DA):
    input_code = np.zeros(n_domains)
    input_code[DA] = 1.
    return input_code


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def renormalize(input_, mask, options):
    
    input_flat = tf.reshape(input_, [options.batch_size,-1])
    mask_flat = tf.reshape(mask, [options.batch_size,-1])
    
    output_flat = []

    for i in range(options.batch_size):
        
        input_fg = tf.gather(input_flat[i, :], tf.where(mask_flat[i, :]))
        mean_fg, var_fg = tf.nn.moments(input_fg, axes=[0], keep_dims=True)
        
        epsilon = 1e-5
        inv_fg = tf.rsqrt(var_fg + epsilon)
        normalized = (input_fg - mean_fg) * inv_fg
        
        if options.is_training:
            mean, var = tf.nn.moments(input_flat[i, :], axes=[0], keep_dims=False)
            inv = tf.rsqrt(var + epsilon)
            
            intensity_scale = tf.constant(random.random() * 0.4 + 0.8)
            intensity_shift = tf.constant((random.random() * 0.4 - 0.2)) / inv
        else:
            intensity_scale = tf.constant(1.0)
            intensity_shift = tf.constant(0.0)
        
        normalized = (normalized + intensity_shift) * intensity_scale
        
        # if fore-ground is not empty.
        output1 = tf.scatter_nd(tf.where(mask_flat[i, :]), normalized, [options.crop_size*options.crop_size*options.crop_size,1])
        
        # if fore-ground is empty.
        output2 = tf.scatter_nd(tf.where(mask_flat[i, :]), input_fg, [options.crop_size*options.crop_size*options.crop_size,1])
        
        # if the summation of foreground intensities is larger than 0.1:
        #     the fore-ground is not empty
        # else:
        #     the fore-ground is empty
        output_flat.append( tf.cond(tf.reduce_sum(tf.abs(input_fg)) > 0.1, lambda: output1, lambda: output2) )

    return tf.reshape(tf.stack(output_flat, axis=0), tf.shape(input_))