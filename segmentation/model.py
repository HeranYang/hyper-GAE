from __future__ import division
import os
import time
import tensorflow as tf
from collections import namedtuple
from module import *
from utils import load_test_data,load_train_data
import imageio
import nibabel as nib


class hypergae(object):
    def __init__(self, sess, args):
        self.sess = sess
        
        self.batch_size = args.batch_size
        
        self.crop_size = args.crop_size
        
        self.code_size0 = 1
        self.code_size1 = 1
        self.code_size2 = 1
        self.code_size3 = args.n_domains
        
        self.image_dim = args.image_dim
        self.output_c_dim = args.m_labels
        
        self.n_domains = args.n_domains
        self.m_labels = args.m_labels
        
        self.L1_lambda = args.L1_lambda
        self.L3_lambda = args.L3_lambda
        self.dataset_dir = args.dataset_dir

        self.encoder_resnet = encoder_resnet
        self.decoder_resnet = decoder_resnet
        self.latentEncodeNet = latentEncodeNet
        self.latentDecodeNet = latentDecodeNet
        self.encode_fusenet = fusenet

        self.coarse_segment_net = coarseSegmentor
        self.fine_segment_net = fineSegmentor
        self.latentCoarseSegScaleNet = latentCoarseSegScaleNet
        self.latentFineSegScaleNet = latentFineSegScaleNet
        
        self.classifer = classifer
        
        self.max_update_num = args.max_update_num

        OPTIONS = namedtuple('OPTIONS', 'batch_size crop_size n_domains \
                                         gf_dim df_dim sf_dim \
                                         image_dim output_c_dim \
                                         is_training')
        self.options = OPTIONS._make((args.batch_size, args.crop_size, args.n_domains,
                                      args.ngf, args.ndf, args.nsf, 
                                      args.image_dim, args.m_labels,
                                      args.phase == 'train'))

        self._build_model()
        self.saver = tf.train.Saver(max_to_keep = 1)


    def _build_model(self):

        # =========================================================================================================================
        # -----VARIBLE DEFINIATION-----
        self.real_syndata = tf.placeholder(tf.float32,
                                        [self.batch_size, 
                                         self.crop_size, 
                                         self.crop_size, 
                                         self.crop_size,
                                         self.image_dim * self.n_domains], name='real_syn_images')
        
        self.real_label = tf.placeholder(tf.float32,
                                        [self.batch_size, 
                                         self.crop_size, 
                                         self.crop_size, 
                                         self.crop_size,
                                         self.m_labels], name='real_labels')
        
        self.mask = tf.placeholder(tf.bool,
                                   [self.batch_size, 
                                    self.crop_size, 
                                    self.crop_size, 
                                    self.crop_size,
                                    self.image_dim], name='real_mask')
        
        self.input_code = tf.placeholder(tf.float32,
                                        [1, 
                                         self.code_size0, 
                                         self.code_size1, 
                                         self.code_size2,
                                         self.code_size3 * self.n_domains], name='input_codes')
          
        self.indicate_code = tf.placeholder(tf.float32,
                                            [self.batch_size,
                                             self.image_dim * self.n_domains], name='indicate_code')
        # =========================================================================================================================

        # =========================================================================================================================
        # -----DATA SEPERATION-----
        # input image in four domains.
        self.real_synA = self.real_syndata[:, :, :, :, :self.image_dim*1]
        self.real_synB = self.real_syndata[:, :, :, :, self.image_dim*1:self.image_dim*2]
        self.real_synC = self.real_syndata[:, :, :, :, self.image_dim*2:self.image_dim*3]
        self.real_synD = self.real_syndata[:, :, :, :, self.image_dim*3:self.image_dim*4]
        
        # one hot code for distinguishing input domain, e.g., [1,0,0], [0,1,0], etc.
        self.real_codeA = self.input_code[:, :, :, :, :self.code_size3*1]
        self.real_codeB = self.input_code[:, :, :, :, self.code_size3*1:self.code_size3*2]
        self.real_codeC = self.input_code[:, :, :, :, self.code_size3*2:self.code_size3*3]
        self.real_codeD = self.input_code[:, :, :, :, self.code_size3*3:self.code_size3*4]
        
        # a number for indicating the existing modalities.
        self.indA = self.indicate_code[:, :self.image_dim*1]
        self.indB = self.indicate_code[:, self.image_dim*1:self.image_dim*2]
        self.indC = self.indicate_code[:, self.image_dim*2:self.image_dim*3]
        self.indD = self.indicate_code[:, self.image_dim*3:self.image_dim*4]
        
        # repeat vector for produce segmentation input.
        self.indicate_code_exp = tf.expand_dims(self.indicate_code, 1)
        self.indicate_code_exp = tf.expand_dims(self.indicate_code_exp, 1)
        self.indicate_code_exp = tf.expand_dims(self.indicate_code_exp, 1)
        
        self.indArep = self.indicate_code_exp[:, :, :, :, :self.image_dim*1]
        self.indBrep = self.indicate_code_exp[:, :, :, :, self.image_dim*1:self.image_dim*2]
        self.indCrep = self.indicate_code_exp[:, :, :, :, self.image_dim*2:self.image_dim*3]
        self.indDrep = self.indicate_code_exp[:, :, :, :, self.image_dim*3:self.image_dim*4]
        # =========================================================================================================================

        # =========================================================================================================================
        # -----SYNTHESIS PROCESSING-----
        # a fully connected network for modifying the filters in encoder based on input code.
        self.latEnScl_A = self.latentEncodeNet(self.real_codeA, self.options, False, name="generator_latEnSclNet")
        self.latEnScl_B = self.latentEncodeNet(self.real_codeB, self.options, True, name="generator_latEnSclNet")
        self.latEnScl_C = self.latentEncodeNet(self.real_codeC, self.options, True, name="generator_latEnSclNet")
        self.latEnScl_D = self.latentEncodeNet(self.real_codeD, self.options, True, name="generator_latEnSclNet")
        
        # a fully connected network for modifying the filters in encoder based on input code.
        self.latEnOff_A = self.latentEncodeNet(self.real_codeA, self.options, False, name="generator_latEnOffNet")
        self.latEnOff_B = self.latentEncodeNet(self.real_codeB, self.options, True, name="generator_latEnOffNet")
        self.latEnOff_C = self.latentEncodeNet(self.real_codeC, self.options, True, name="generator_latEnOffNet")
        self.latEnOff_D = self.latentEncodeNet(self.real_codeD, self.options, True, name="generator_latEnOffNet")

        # a fully connected network for modifying the filters in decoder based on input code.
        self.latDeScl_A = self.latentDecodeNet(self.real_codeA, self.options, False, name="generator_latDeSclNet")
        self.latDeScl_B = self.latentDecodeNet(self.real_codeB, self.options, True, name="generator_latDeSclNet")
        self.latDeScl_C = self.latentDecodeNet(self.real_codeC, self.options, True, name="generator_latDeSclNet")
        self.latDeScl_D = self.latentDecodeNet(self.real_codeD, self.options, True, name="generator_latDeSclNet")
        
        # a fully connected network for modifying the filters in decoder based on input code.
        self.latDeOff_A = self.latentDecodeNet(self.real_codeA, self.options, False, name="generator_latDeOffNet")
        self.latDeOff_B = self.latentDecodeNet(self.real_codeB, self.options, True, name="generator_latDeOffNet")
        self.latDeOff_C = self.latentDecodeNet(self.real_codeC, self.options, True, name="generator_latDeOffNet")
        self.latDeOff_D = self.latentDecodeNet(self.real_codeD, self.options, True, name="generator_latDeOffNet")

        # extract deep features (tissue parameters) of each modality.
        self.encode_A = self.encoder_resnet(self.real_synA, 
                                            self.latEnScl_A, 
                                            self.latEnOff_A, self.options, reuse=False, name="generator_encoder")
        self.encode_B = self.encoder_resnet(self.real_synB, 
                                            self.latEnScl_B, 
                                            self.latEnOff_B, self.options, reuse=True, name="generator_encoder")
        self.encode_C = self.encoder_resnet(self.real_synC, 
                                            self.latEnScl_C, 
                                            self.latEnOff_C, self.options, reuse=True, name="generator_encoder")
        self.encode_D = self.encoder_resnet(self.real_synD, 
                                            self.latEnScl_D, 
                                            self.latEnOff_D, self.options, reuse=True, name="generator_encoder")

        # feature fusion.
        self.encode_fused = self.encode_fusenet(self.encode_A, 
                                                self.encode_B, 
                                                self.encode_C, 
                                                self.encode_D, 
                                                self.indicate_code_exp,
                                                self.input_code, self.options, reuse=False, name="generator_fusenet")

        # reconstruct each modality from extracted deep features.
        self.fake_synA = self.decoder_resnet(self.encode_fused, 
                                             self.latDeScl_A, 
                                             self.latDeOff_A, self.options, reuse=False, name="generator_decoder")
        self.fake_synB = self.decoder_resnet(self.encode_fused, 
                                             self.latDeScl_B, 
                                             self.latDeOff_B, self.options, reuse=True, name="generator_decoder")
        self.fake_synC = self.decoder_resnet(self.encode_fused, 
                                             self.latDeScl_C, 
                                             self.latDeOff_C, self.options, reuse=True, name="generator_decoder")
        self.fake_synD = self.decoder_resnet(self.encode_fused, 
                                             self.latDeScl_D, 
                                             self.latDeOff_D, self.options, reuse=True, name="generator_decoder")
        # =========================================================================================================================

        # =========================================================================================================================
        # -----SEGMENTATION PROCESSING-----
        # prepare segmentation input.
        # renormalize the synthesized image for segmentation network input.
        self.fake_segA = renormalize(self.fake_synA, self.mask, self.options)
        self.fake_segB = renormalize(self.fake_synB, self.mask, self.options)
        self.fake_segC = renormalize(self.fake_synC, self.mask, self.options)
        self.fake_segD = renormalize(self.fake_synD, self.mask, self.options)
        
        # renormalize the real image for segmentation network input.
        self.real_segA = renormalize(self.real_synA, self.mask, self.options)
        self.real_segB = renormalize(self.real_synB, self.mask, self.options)
        self.real_segC = renormalize(self.real_synC, self.mask, self.options)
        self.real_segD = renormalize(self.real_synD, self.mask, self.options)
        
        # reform the inputs of semgentation network.
        self.seginput_A = tf.multiply(self.indArep, self.real_segA) + tf.multiply(tf.ones_like(self.indArep) - self.indArep, self.fake_segA)
        self.seginput_B = tf.multiply(self.indBrep, self.real_segB) + tf.multiply(tf.ones_like(self.indBrep) - self.indBrep, self.fake_segB)
        self.seginput_C = tf.multiply(self.indCrep, self.real_segC) + tf.multiply(tf.ones_like(self.indCrep) - self.indCrep, self.fake_segC)
        self.seginput_D = tf.multiply(self.indDrep, self.real_segD) + tf.multiply(tf.ones_like(self.indDrep) - self.indDrep, self.fake_segD)

        # input of corase segmentation.
        self.cseginput = tf.concat([self.seginput_A, 
                                    self.seginput_B, 
                                    self.seginput_C,
                                    self.seginput_D], 4)

        # CORASE SEGMENTATION.
        # a fully connected network for modifying the filters in segmentor based on indicate code.
        self.latCoaSegScl = self.latentCoarseSegScaleNet(self.indicate_code_exp, self.options, reuse=False, name="generator_latCoaSegSclNet")
        
        # coarsely segment the label from the multi-modal inputs.
        self.seg_coarselabel = self.coarse_segment_net(self.cseginput, 
                                                       self.latCoaSegScl, 
                                                       self.encode_fused, self.options, reuse=False, name="generator_coarsesegmentor")

        # FINE SEGMENTATION.
        # a fully connected network for modifying the filters in segmentor based on indicate code.
        self.latFineSegScl = self.latentFineSegScaleNet(self.indicate_code_exp, self.options, reuse=False, name="generator_latFineSegSclNet")
        
        # coarsely segment the label from the multi-modal inputs.
        self.fseginput = tf.concat([self.cseginput, self.seg_coarselabel], 4)
        
        self.seg_finelabel1, self.seg_finelabel2 = self.fine_segment_net(self.fseginput, 
                                                                         self.latFineSegScl, 
                                                                         self.encode_fused, self.options, reuse=False, name="generator_finesegmentor")
        # =========================================================================================================================

        # =========================================================================================================================
        # -----LOSS FUNCTION-----
        # --------------------------------------------------------------------
        # reconstruction loss.
        self.loss_recon = (abs_criterion(self.real_synA, self.fake_synA) \
                         + abs_criterion(self.real_synB, self.fake_synB) \
                         + abs_criterion(self.real_synC, self.fake_synC) \
                         + abs_criterion(self.real_synD, self.fake_synD)) / 4.
        # --------------------------------------------------------------------

        # --------------------------------------------------------------------
        # segmentation loss.
        self.loss_cseg_dice = (dice_criterion(self.seg_coarselabel[:,:,:,:,0], self.real_label[:,:,:,:,0]) \
                             + dice_criterion(self.seg_coarselabel[:,:,:,:,1], self.real_label[:,:,:,:,1]) \
                             + dice_criterion(self.seg_coarselabel[:,:,:,:,2], self.real_label[:,:,:,:,2])) / 3.
                 
        self.loss_fseg1_dice = (dice_criterion(self.seg_finelabel1[:,:,:,:,0], self.real_label[:,:,:,:,0]) \
                              + dice_criterion(self.seg_finelabel1[:,:,:,:,1], self.real_label[:,:,:,:,1]) \
                              + dice_criterion(self.seg_finelabel1[:,:,:,:,2], self.real_label[:,:,:,:,2])) / 3.
        
        self.loss_fseg2_dice = (dice_criterion(self.seg_finelabel2[:,:,:,:,0], self.real_label[:,:,:,:,0]) \
                              + dice_criterion(self.seg_finelabel2[:,:,:,:,1], self.real_label[:,:,:,:,1]) \
                              + dice_criterion(self.seg_finelabel2[:,:,:,:,2], self.real_label[:,:,:,:,2])) / 3.
        
        # TOTAL SEGMENTATION LOSS.
        self.loss_seg_dice = (self.loss_cseg_dice + self.loss_fseg1_dice + self.loss_fseg2_dice) / 3.
        # --------------------------------------------------------------------

        # --------------------------------------------------------------------
        # adversarial common feature loss.
        self.encode_fused_ex =  self.encode_fusenet(self.encode_A, 
                                                    self.encode_B, 
                                                    self.encode_C, 
                                                    self.encode_D, 
                                                    tf.ones_like(self.indicate_code_exp),
                                                    self.input_code, self.options, reuse=True, name="generator_fusenet")
        
        # classifier.
        self.Lab_fused = self.classifer(self.encode_fused, self.options, reuse=False, name="generator_classifier")
        self.Lab_fusedex = self.classifer(self.encode_fused_ex, self.options, reuse=True, name="generator_classifier")
        
        self.loss_acf = sce_criterion(self.Lab_fused, 
                                     tf.multiply(tf.ones_like(self.Lab_fused), 
                                                 self.indicate_code_exp)) \
                     + sce_criterion(self.Lab_fusedex, 
                                     tf.multiply(tf.ones_like(self.Lab_fusedex), 
                                                 tf.ones_like(self.indicate_code_exp)))
        # --------------------------------------------------------------------

        # total loss.
        self.loss_total = - self.loss_seg_dice \
                          + self.L1_lambda * self.loss_recon \
                          + self.L3_lambda * self.loss_acf
        # =========================================================================================================================

        # define loss for tensorboard.
        self.loss_recon_sum = tf.summary.scalar("loss_recon", self.loss_recon)
        self.loss_seg_dice_sum = tf.summary.scalar("loss_seg_dice", self.loss_seg_dice)
        self.loss_acf_sum = tf.summary.scalar("loss_acf", self.loss_acf)
        self.loss_total_sum = tf.summary.scalar("loss_total", self.loss_total)
        
        self.loss_sum = tf.summary.merge([self.loss_recon_sum, 
                                          self.loss_seg_dice_sum, 
                                          self.loss_acf_sum,
                                          self.loss_total_sum])
        
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        for var in t_vars: print(var.name)


    def train(self, args):
        """Train hypergae"""
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.loss_total, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()
        
        # # ======================================================
        # # code for continue training.
        # if args.continue_train:
        #     self.load_valid(r'./r1/checkpoint/', 900)
        #     print(" [*] Load SUCCESS")
        # # ======================================================

        saver_all = tf.train.Saver(max_to_keep = None)

        for epoch in range(args.epoch):

            # # ======================================================
            # # code for continue training.
            # # without using lr warmup stragegy.
            # # only used for continue learning.
            # lr = args.lr
            # # ======================================================
            
            # ======================================================
            # code for learning from scartch.
            # learning rate warmup strategy.
            if (epoch + 1) < args.warmup_epoch:
                lr = args.lr * (epoch + 1) / args.warmup_epoch
            else:
                lr = args.lr
            # ======================================================

            print(("Epoch: [%2d] learning rate: %1.6f" % (epoch, lr)))

            # path to four modalities.
            domA_synpath = '../data/BraTS-Dataset-pro/SliceData/3DTrain/TrainA/'
            domB_synpath = '../data/BraTS-Dataset-pro/SliceData/3DTrain/TrainB/'
            domC_synpath = '../data/BraTS-Dataset-pro/SliceData/3DTrain/TrainC/'
            domD_synpath = '../data/BraTS-Dataset-pro/SliceData/3DTrain/TrainD/'
            domSynImagePathList = [domA_synpath, domB_synpath, domC_synpath, domD_synpath]
            
            # path to label.
            domL_path = '../data/BraTS-Dataset-pro/SliceData/3DTrain/TrainL/'

            # load all slice name. 
            # since all modalities have the same name, we take the name list from label.
            dataNameList = os.listdir(domL_path) * 10
            # randomize.
            np.random.shuffle(dataNameList)

            batch_idxs = self.max_update_num // self.batch_size

            for idx in range(0, batch_idxs):
                
                indicate_code = np.zeros((self.batch_size, self.n_domains)).astype(np.float32)
                input_code = np.zeros((self.batch_size, self.code_size3 * self.n_domains)).astype(np.float32)
                
                for ibs in range(self.batch_size):
                    ## indicate_code
                    # 1 for existing modality, and 0 for missing one.
                    indicate_code_tmp = np.random.randint(0, 2, self.n_domains)
                    if not (indicate_code_tmp == 1).any():
                        indicate_code_tmp[np.random.randint(0, self.n_domains, 1)] = 1
                    
                    indicate_code_tmp = np.array(indicate_code_tmp).astype(np.float32)
                    indicate_code[ibs, :] = indicate_code_tmp
                    
                ## input_code
                code_dom0 = prod_input_code(self.n_domains, 0)
                code_dom1 = prod_input_code(self.n_domains, 1)
                code_dom2 = prod_input_code(self.n_domains, 2)
                code_dom3 = prod_input_code(self.n_domains, 3)
                
                input_code_tmp = np.array(np.hstack((code_dom0, 
                                                     code_dom1, 
                                                     code_dom2, 
                                                     code_dom3))).astype(np.float32)
                
                input_code = input_code_tmp.reshape([1, 
                                                     self.code_size0,
                                                     self.code_size1, 
                                                     self.code_size2,
                                                     self.code_size3 * self.n_domains])

                dataName_idxs = dataNameList[idx*self.batch_size : (idx+1)*self.batch_size]
                
                data0_syn_idxs = []
                data1_syn_idxs = []
                data2_syn_idxs = []
                data3_syn_idxs = []
                
                label_idxs = []
                
                for ibs in range(self.batch_size):
                    
                    data0_syn_idxs.append(domSynImagePathList[0] + dataName_idxs[ibs])
                    data1_syn_idxs.append(domSynImagePathList[1] + dataName_idxs[ibs])
                    data2_syn_idxs.append(domSynImagePathList[2] + dataName_idxs[ibs])
                    data3_syn_idxs.append(domSynImagePathList[3] + dataName_idxs[ibs])
                    
                    label_idxs.append(domL_path + dataName_idxs[ibs])

                batch_files = list(zip(data0_syn_idxs, 
                                       data1_syn_idxs, 
                                       data2_syn_idxs, 
                                       data3_syn_idxs, 
                                       label_idxs))
                
                # real_image.
                starttime0=time.time()
                batch_alls = [load_train_data(batch_file, crop_size=self.crop_size) for batch_file in batch_files]
                endtime0=time.time()
                print (endtime0 - starttime0)
                
                batch_alls = np.array(batch_alls).astype(np.float32)
                
                batch_synimages = batch_alls[:, :, :, :, :self.n_domains]
                batch_labels = batch_alls[:, :, :, :, self.n_domains : self.n_domains+self.m_labels]
                batch_masks = batch_alls[:, :, :, :, self.n_domains+self.m_labels:]

                # Update G network and record fake outputs
                starttime=time.time()
                _, summary_str, diceloss, esLabel = self.sess.run(
                    [self.g_optim, self.loss_sum, 
                     self.loss_seg_dice, self.seg_finelabel1],
                    feed_dict={self.real_syndata: batch_synimages,
                               self.real_label: batch_labels,
                               self.mask: batch_masks,
                               self.input_code: input_code, 
                               self.indicate_code: indicate_code, 
                               self.lr: lr})
                self.writer.add_summary(summary_str, counter)
                endtime=time.time()
                print (endtime-starttime)

                counter += 1
                print(("Epoch: [%2d] [%4d/%4d] time: %4.4f diceloss: %2.4f" % (
                    epoch, idx, batch_idxs, time.time() - start_time, diceloss)))

                if np.mod(counter, args.print_freq) == 1:
                    self.sample_model(args.sample_dir, epoch, idx, batch_files, input_code)

            save_epoch_file = os.path.join(args.checkpoint_dir, 'MulModal_epoch')
            if not os.path.exists(save_epoch_file):
                os.makedirs(save_epoch_file)
            if np.mod(epoch, args.save_freq) == 0:
                saver_all.save(self.sess, os.path.join(save_epoch_file, 'hypergae.epoch'), global_step=epoch)


    def load_valid(self, checkpoint_dir, epoch):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_epoch" % (self.dataset_dir)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt_name = os.path.basename('hypergae.epoch-{}'.format(epoch))
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))


    def sample_model(self, sample_dir, epoch, idx, batch_files, input_code):

        indicate_code = np.zeros((self.batch_size, self.n_domains)).astype(np.float32)
        
        for ibs in range(self.batch_size):
            ## indicate_code
            # 1 for existing modality, and 0 for missing one.
            indicate_code_tmp = np.random.randint(0, 2, self.n_domains)
            if not (indicate_code_tmp == 1).any():
                indicate_code_tmp[np.random.randint(0, self.n_domains, 1)] = 1
            
            indicate_code_tmp = np.array(indicate_code_tmp).astype(np.float32)
            indicate_code[ibs, :] = indicate_code_tmp

        ## load image and label in test mode.
        # real_image.
        batch_alls = [load_train_data(batch_file, crop_size=self.crop_size, is_testing=True) for batch_file in batch_files]
        batch_alls = np.array(batch_alls).astype(np.float32)
                
        batch_synimages = batch_alls[:, :, :, :, :self.n_domains]
        batch_labels = batch_alls[:, :, :, :, self.n_domains : self.n_domains+self.m_labels]
        batch_masks = batch_alls[:, :, :, :, self.n_domains+self.m_labels:]

        ## feed into the network.
        fake_A, fake_B, fake_C, fake_D, seg_label = self.sess.run(
            [self.fake_synA, self.fake_synB, self.fake_synC, self.fake_synD, self.seg_finelabel1],
            feed_dict={self.real_syndata: batch_synimages,
                       self.real_label: batch_labels,
                       self.mask: batch_masks,
                       self.input_code: input_code,
                       self.indicate_code: indicate_code}
        )

        images_show_all = []
        
        ibs_num = 2
        show_num = 3
        
        for ibs in range(ibs_num):
            for show_index in range(show_num):
                
                show_id = show_index + (self.crop_size // 2) - (show_num // 2)
                
                inputA = np.array(batch_synimages[ibs,show_id,:,:,0]).reshape([self.crop_size, self.crop_size])
                inputA = (inputA + 1.) * 127.5
                inputA[:5,:5] = indicate_code[ibs, 0] * 255
                reconA = np.array(fake_A[ibs,show_id,:,:]).reshape([self.crop_size, self.crop_size])
                reconA = (reconA + 1.) * 127.5
                
                inputB = np.array(batch_synimages[ibs,show_id,:,:,1]).reshape([self.crop_size, self.crop_size])
                inputB = (inputB + 1.) * 127.5
                inputB[:5,:5] = indicate_code[ibs, 1] * 255
                reconB = np.array(fake_B[ibs,show_id,:,:]).reshape([self.crop_size, self.crop_size])
                reconB = (reconB + 1.) * 127.5
                
                inputC = np.array(batch_synimages[ibs,show_id,:,:,2]).reshape([self.crop_size, self.crop_size])
                inputC = (inputC + 1.) * 127.5
                inputC[:5,:5] = indicate_code[ibs, 2] * 255
                reconC = np.array(fake_C[ibs,show_id,:,:]).reshape([self.crop_size, self.crop_size])
                reconC = (reconC + 1.) * 127.5
                
                inputD = np.array(batch_synimages[ibs,show_id,:,:,3]).reshape([self.crop_size, self.crop_size])
                inputD = (inputD + 1.) * 127.5
                inputD[:5,:5] = indicate_code[ibs, 3] * 255
                reconD = np.array(fake_D[ibs,show_id,:,:]).reshape([self.crop_size, self.crop_size])
                reconD = (reconD + 1.) * 127.5

                realWT = np.array(batch_labels[ibs,show_id,:,:,0]).reshape([self.crop_size, self.crop_size])
                realWT = realWT * 255
                resWT = np.array(seg_label[ibs,show_id,:,:,0]).reshape([self.crop_size, self.crop_size])
                resWT = resWT * 255
                
                realTC = np.array(batch_labels[ibs,show_id,:,:,1]).reshape([self.crop_size, self.crop_size])
                realTC = realTC * 255
                resTC = np.array(seg_label[ibs,show_id,:,:,1]).reshape([self.crop_size, self.crop_size])
                resTC = resTC * 255
                
                realET = np.array(batch_labels[ibs,show_id,:,:,2]).reshape([self.crop_size, self.crop_size])
                realET = realET * 255
                resET = np.array(seg_label[ibs,show_id,:,:,2]).reshape([self.crop_size, self.crop_size])
                resET = resET * 255

                images_real = np.concatenate([inputA, inputB, inputC, inputD, realWT, realTC, realET], axis = 1)
                images_res = np.concatenate([reconA, reconB, reconC, reconD, resWT, resTC, resET], axis = 1)
                images_show = np.concatenate([images_real, images_res], axis = 0)
                
                if ibs == 0 and show_index == 0:
                    images_show_all = images_show
                else:
                    images_show_all = np.concatenate([images_show_all, images_show], axis = 0)

        imageio.imwrite('./{}/result_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx), images_show_all)


    def valid(self, args):
        """valid hypergae"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        domA_path = r'../data/BraTS-Dataset-pro/VolumeData/Valid/ValidA/'
        domB_path = r'../data/BraTS-Dataset-pro/VolumeData/Valid/ValidB/'
        domC_path = r'../data/BraTS-Dataset-pro/VolumeData/Valid/ValidC/'
        domD_path = r'../data/BraTS-Dataset-pro/VolumeData/Valid/ValidD/'
        domImagePathList = [domA_path, domB_path, domC_path, domD_path]
        
        domL_path = r'../data/BraTS-Dataset-pro/VolumeData/Valid/ValidL/'

        input_image_name = 'BraTS19-id{:0>3d}.nii.gz'

        # ======================================================
        # modified according to dataset split.
        valNum = 6
        valVec = np.arange(valNum) + 1
        
        # modified according to trainging process.
        epochNum = 41
        epochVec = np.arange(epochNum) * 10 + 800
        # ======================================================

        indicate_matrix = [[1,0,0,0],
                           [0,1,0,0],
                           [0,0,1,0],
                           [0,0,0,1],
                           [1,1,0,0],
                           [1,0,1,0],
                           [1,0,0,1],
                           [0,1,1,0],
                           [0,1,0,1],
                           [0,0,1,1],
                           [1,1,1,0],
                           [1,1,0,1],
                           [1,0,1,1],
                           [0,1,1,1],
                           [1,1,1,1]]
        indicate_matrix = np.array(indicate_matrix)
        
        inputIndVec = np.arange(15)
        
        for epoch in epochVec:
        
            valid_dir = './valid'
            if not os.path.exists(valid_dir):
                os.makedirs(valid_dir)

            valid_checkpoint = './checkpoint'

            self.load_valid(valid_checkpoint, epoch)

            ## domain_code
            domain_code = np.arange(self.n_domains)
            D0, D1, D2, D3 = domain_code
            
            ## input_code
            code_dom0 = prod_input_code(self.n_domains, D0)
            code_dom1 = prod_input_code(self.n_domains, D1)
            code_dom2 = prod_input_code(self.n_domains, D2)
            code_dom3 = prod_input_code(self.n_domains, D3)
            
            input_code_tmp = np.array(np.hstack((code_dom0, 
                                                 code_dom1, 
                                                 code_dom2, 
                                                 code_dom3))).astype(np.float32)
            
            input_code = input_code_tmp.reshape([1, 
                                                 self.code_size0, 
                                                 self.code_size1, 
                                                 self.code_size2,
                                                 self.code_size3 * self.n_domains])
            
            input_code = np.repeat(input_code, self.batch_size, axis=0)

            for val_id in valVec:
                    
                dataName_idxs = input_image_name.format(val_id)
                
                data0_idxs = domImagePathList[D0] + dataName_idxs
                data1_idxs = domImagePathList[D1] + dataName_idxs
                data2_idxs = domImagePathList[D2] + dataName_idxs
                data3_idxs = domImagePathList[D3] + dataName_idxs
                
                label_idxs = domL_path + dataName_idxs
                
                batch_files = [data0_idxs, data1_idxs, data2_idxs, data3_idxs, label_idxs]

                inputImageAll = nib.load(data0_idxs)
                inputLabelAll = nib.load(label_idxs)

                volA, volB, volC, volD, volLabel, volMask = load_test_data(batch_files)
                
                volA = np.array(volA).astype(np.float32)
                volB = np.array(volB).astype(np.float32)
                volC = np.array(volC).astype(np.float32)
                volD = np.array(volD).astype(np.float32)
                volLabel = np.array(volLabel).astype(np.float32)
                volMask = np.array(volMask).astype(np.bool)
                
                batch_volumes = np.stack((volA, volB, volC, volD, volMask), axis=3)
                
                # ====================================================================================================
                # cropping volume into crop_size.
                HEIGHT = args.crop_size
                WIDTH = args.crop_size
                DEPTH = args.crop_size
                NUM_CLS = args.m_labels
                
                h, w, d = np.shape(volA)
                overlap_perc = 0.5

                h_cnt = np.int(np.ceil((h - HEIGHT) / (HEIGHT * (1 - overlap_perc))))
                h_idx_list = range(0, h_cnt)
                h_idx_list = [h_idx * np.int(HEIGHT * (1 - overlap_perc)) for h_idx in h_idx_list]
                h_idx_list.append(h - HEIGHT)

                w_cnt = np.int(np.ceil((w - WIDTH) / (WIDTH * (1 - overlap_perc))))
                w_idx_list = range(0, w_cnt)
                w_idx_list = [w_idx * np.int(WIDTH * (1 - overlap_perc)) for w_idx in w_idx_list]
                w_idx_list.append(w - WIDTH)

                d_cnt = np.int(np.ceil((d - DEPTH) / (DEPTH * (1 - overlap_perc))))
                d_idx_list = range(0, d_cnt)
                d_idx_list = [d_idx * np.int(DEPTH * (1 - overlap_perc)) for d_idx in d_idx_list]
                d_idx_list.append(d - DEPTH)
                # ====================================================================================================

                for inputInd in inputIndVec:

                    print('Processing image: id ' + str(val_id) + '  input ' + str(inputInd))

                    indicate_code = np.expand_dims(indicate_matrix[inputInd], axis=0).astype(np.int32)
                    indicate_code = np.repeat(indicate_code, self.batch_size, axis=0)

                    pred_whole = np.zeros((h, w, d, NUM_CLS))
                    avg_whole = np.zeros((h, w, d, NUM_CLS))
                    avg_block = np.ones((HEIGHT, WIDTH, DEPTH, NUM_CLS))
                    
                    for d_idx in d_idx_list:
                        for w_idx in w_idx_list:
                            for h_idx in h_idx_list:
                                
                                batch_alls = batch_volumes[h_idx:h_idx + HEIGHT,
                                                             w_idx:w_idx + WIDTH, 
                                                             d_idx:d_idx + DEPTH, :]
                                
                                batch_alls = np.expand_dims(batch_alls, axis=0)
                                
                                
                                batch_images = batch_alls[:, :, :, :, :self.n_domains]
                                batch_masks = batch_alls[:, :, :, :, self.n_domains:]

                                seg_label = self.sess.run(
                                    self.seg_finelabel1,
                                    feed_dict={self.real_syndata: batch_images,
                                               self.mask: batch_masks,
                                               self.input_code: input_code,
                                               self.indicate_code: indicate_code}
                                )
                                
                                seg_label = np.array(seg_label).astype('float16').reshape([HEIGHT, WIDTH, DEPTH, NUM_CLS])
                                
                                
                                pred_whole[h_idx:h_idx + HEIGHT, 
                                           w_idx:w_idx + WIDTH, 
                                           d_idx:d_idx + DEPTH, :] = pred_whole[h_idx:h_idx + HEIGHT, 
                                                                                w_idx:w_idx + WIDTH, 
                                                                                d_idx:d_idx + DEPTH, :] + seg_label
                                avg_whole[h_idx:h_idx + HEIGHT, 
                                          w_idx:w_idx + WIDTH, 
                                          d_idx:d_idx + DEPTH, :] = avg_whole[h_idx:h_idx + HEIGHT, 
                                                                              w_idx:w_idx + WIDTH, 
                                                                              d_idx:d_idx + DEPTH, :] + avg_block
                    
                    pred_whole = pred_whole / avg_whole

                    label_head_output = inputLabelAll.header
                    label_affine_output = inputLabelAll.affine

                    saveLab0 = nib.Nifti1Image(pred_whole[:,:,:,0], label_affine_output, label_head_output)
                    nib.save(saveLab0, '{}/Epoch{:0>3d}-id{:0>3d}_input{:0>2d}_esLabelWT.nii.gz'.format(valid_dir, epoch, val_id, inputInd))
                    
                    saveLab1 = nib.Nifti1Image(pred_whole[:,:,:,1], label_affine_output, label_head_output)
                    nib.save(saveLab1, '{}/Epoch{:0>3d}-id{:0>3d}_input{:0>2d}_esLabelTC.nii.gz'.format(valid_dir, epoch, val_id, inputInd))
                    
                    saveLab2 = nib.Nifti1Image(pred_whole[:,:,:,2], label_affine_output, label_head_output)
                    nib.save(saveLab2, '{}/Epoch{:0>3d}-id{:0>3d}_input{:0>2d}_esLabelET.nii.gz'.format(valid_dir, epoch, val_id, inputInd))

                    # save the ground-truth images and labels.
                    if epoch == epochVec[0] and inputInd == 0:
                        
                        gtlabtmp0 = np.zeros(inputLabelAll.shape, dtype=np.int16)
                        gtlabtmp0[volLabel>0] = 1
                        gtLab0 = nib.Nifti1Image(gtlabtmp0, label_affine_output, label_head_output)
                        nib.save(gtLab0, '{}/Aid{:0>3d}_gtLabelWT.nii.gz'.format(valid_dir, val_id))
                        
                        gtlabtmp1 = np.zeros(inputLabelAll.shape, dtype=np.int16)
                        gtlabtmp1[volLabel==4] = 1
                        gtlabtmp1[volLabel==1] = 1
                        gtLab1 = nib.Nifti1Image(gtlabtmp1, label_affine_output, label_head_output)
                        nib.save(gtLab1, '{}/Aid{:0>3d}_gtLabelTC.nii.gz'.format(valid_dir, val_id))
                        
                        gtlabtmp2 = np.zeros(inputLabelAll.shape, dtype=np.int16)
                        gtlabtmp2[volLabel==4] = 1
                        gtLab2 = nib.Nifti1Image(gtlabtmp2, label_affine_output, label_head_output)
                        nib.save(gtLab2, '{}/Aid{:0>3d}_gtLabelET.nii.gz'.format(valid_dir, val_id))


    def test(self, args):
        """Test hypergae"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        domA_path = r'../data/BraTS-Dataset-pro/VolumeData/Test/TestA/'
        domB_path = r'../data/BraTS-Dataset-pro/VolumeData/Test/TestB/'
        domC_path = r'../data/BraTS-Dataset-pro/VolumeData/Test/TestC/'
        domD_path = r'../data/BraTS-Dataset-pro/VolumeData/Test/TestD/'
        domImagePathList = [domA_path, domB_path, domC_path, domD_path]
        
        domL_path = r'../data/BraTS-Dataset-pro/VolumeData/Test/TestL/'

        input_image_name = 'BraTS19-id{:0>3d}.nii.gz'

        # ======================================================
        # modified according to dataset split.
        tedataSize = 111
        teIdVec = np.arange(tedataSize) + 1
        
        # selected epoch in validation process.
        epoch = 1150
        # ======================================================

        test_checkpoint = './checkpoint'
        self.load_valid(test_checkpoint, epoch)

        test_base = './test'
        if not os.path.exists(test_base):
            os.makedirs(test_base)

        test_dir = '{}/epoch_{}'.format(test_base, epoch)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        indicate_matrix = [[1,0,0,0],
                           [0,1,0,0],
                           [0,0,1,0],
                           [0,0,0,1],
                           [1,1,0,0],
                           [1,0,1,0],
                           [1,0,0,1],
                           [0,1,1,0],
                           [0,1,0,1],
                           [0,0,1,1],
                           [1,1,1,0],
                           [1,1,0,1],
                           [1,0,1,1],
                           [0,1,1,1],
                           [1,1,1,1]]
                           
        indicate_matrix = np.array(indicate_matrix)
        
        inputIndVec = np.arange(15)

        ## domain_code
        domain_code = np.arange(self.n_domains)
        D0, D1, D2, D3 = domain_code

        ## input_code
        code_dom0 = prod_input_code(self.n_domains, D0)
        code_dom1 = prod_input_code(self.n_domains, D1)
        code_dom2 = prod_input_code(self.n_domains, D2)
        code_dom3 = prod_input_code(self.n_domains, D3)
        
        input_code_tmp = np.array(np.hstack((code_dom0, 
                                             code_dom1, 
                                             code_dom2, 
                                             code_dom3))).astype(np.float32)
        
        input_code = input_code_tmp.reshape([1, 
                                             self.code_size0, 
                                             self.code_size1, 
                                             self.code_size2,
                                             self.code_size3 * self.n_domains])
        
        input_code = np.repeat(input_code, self.batch_size, axis=0)

        for teId in teIdVec:
            
            dataName_idxs = input_image_name.format(teId)
                
            data0_idxs = domImagePathList[D0] + dataName_idxs
            data1_idxs = domImagePathList[D1] + dataName_idxs
            data2_idxs = domImagePathList[D2] + dataName_idxs
            data3_idxs = domImagePathList[D3] + dataName_idxs
            
            label_idxs = domL_path + dataName_idxs
            
            batch_files = [data0_idxs, data1_idxs, data2_idxs, data3_idxs, label_idxs]

            inputImageAll = nib.load(data0_idxs)
            inputLabelAll = nib.load(label_idxs)

            volA, volB, volC, volD, volLabel, volMask = load_test_data(batch_files)
            
            volA = np.array(volA).astype(np.float32)
            volB = np.array(volB).astype(np.float32)
            volC = np.array(volC).astype(np.float32)
            volD = np.array(volD).astype(np.float32)
            volLabel = np.array(volLabel).astype(np.float32)
            volMask = np.array(volMask).astype(np.bool)
            
            batch_volumes = np.stack((volA, volB, volC, volD, volMask), axis=3)
                
            # ====================================================================================================
            # cropping volume into crop_size.
            HEIGHT = args.crop_size
            WIDTH = args.crop_size
            DEPTH = args.crop_size
            NUM_CLS = args.m_labels
            
            h, w, d = np.shape(volA)
            overlap_perc = 0.5

            h_cnt = np.int(np.ceil((h - HEIGHT) / (HEIGHT * (1 - overlap_perc))))
            h_idx_list = range(0, h_cnt)
            h_idx_list = [h_idx * np.int(HEIGHT * (1 - overlap_perc)) for h_idx in h_idx_list]
            h_idx_list.append(h - HEIGHT)

            w_cnt = np.int(np.ceil((w - WIDTH) / (WIDTH * (1 - overlap_perc))))
            w_idx_list = range(0, w_cnt)
            w_idx_list = [w_idx * np.int(WIDTH * (1 - overlap_perc)) for w_idx in w_idx_list]
            w_idx_list.append(w - WIDTH)

            d_cnt = np.int(np.ceil((d - DEPTH) / (DEPTH * (1 - overlap_perc))))
            d_idx_list = range(0, d_cnt)
            d_idx_list = [d_idx * np.int(DEPTH * (1 - overlap_perc)) for d_idx in d_idx_list]
            d_idx_list.append(d - DEPTH)
            # ====================================================================================================

            for inputInd in inputIndVec:
                
                starttime_test=time.time()
                
                print('Processing image: id ' + str(teId) + '  input ' + str(inputInd))
                    
                indicate_code = np.expand_dims(indicate_matrix[inputInd], axis=0).astype(np.int32)
                indicate_code = np.repeat(indicate_code, self.batch_size, axis=0)
                
                pred_whole = np.zeros((h, w, d, NUM_CLS))
                avg_whole = np.zeros((h, w, d, NUM_CLS))
                avg_block = np.ones((HEIGHT, WIDTH, DEPTH, NUM_CLS))
                
                for d_idx in d_idx_list:
                    for w_idx in w_idx_list:
                        for h_idx in h_idx_list:
                            
                            batch_alls = batch_volumes[h_idx:h_idx + HEIGHT, 
                                                          w_idx:w_idx + WIDTH, 
                                                          d_idx:d_idx + DEPTH, :]
                            
                            batch_alls = np.expand_dims(batch_alls, axis=0)
                            
                            batch_images0 = batch_alls[:, :, :, :, :self.n_domains]
                            batch_masks0 = batch_alls[:, :, :, :, self.n_domains:]

                            # =========================================================================================
                            # flip-1.
                            batch_images = batch_images0[:,:,:,:,:]
                            batch_masks = batch_masks0[:,:,:,:,:]
                            
                            seg_label = self.sess.run(
                                self.seg_finelabel1,
                                feed_dict={self.real_syndata: batch_images,
                                           self.mask: batch_masks,
                                           self.input_code: input_code,
                                           self.indicate_code: indicate_code}
                            )
                            
                            seg_label = np.array(seg_label).astype('float16').reshape([HEIGHT, WIDTH, DEPTH, NUM_CLS])

                            pred_whole[h_idx:h_idx + HEIGHT, 
                                       w_idx:w_idx + WIDTH, 
                                       d_idx:d_idx + DEPTH, :] = pred_whole[h_idx:h_idx + HEIGHT, 
                                                                            w_idx:w_idx + WIDTH, 
                                                                            d_idx:d_idx + DEPTH, :] + seg_label
                            avg_whole[h_idx:h_idx + HEIGHT, 
                                      w_idx:w_idx + WIDTH, 
                                      d_idx:d_idx + DEPTH, :] = avg_whole[h_idx:h_idx + HEIGHT, 
                                                                          w_idx:w_idx + WIDTH, 
                                                                          d_idx:d_idx + DEPTH, :] + avg_block

                              # flip-2.
                            batch_images = batch_images0[:, ::-1, :, :, :]
                            batch_masks = batch_masks0[:, ::-1, :, :, :]
                            
                            seg_label = self.sess.run(
                                  self.seg_finelabel1,
                                  feed_dict={self.real_syndata: batch_images,
                                              self.mask: batch_masks,
                                              self.input_code: input_code,
                                              self.indicate_code: indicate_code}
                              )
                            
                            seg_label = np.array(seg_label[:, ::-1, :, :, :]).astype('float16').reshape([HEIGHT, WIDTH, DEPTH, NUM_CLS])
                            
                            pred_whole[h_idx:h_idx + HEIGHT, 
                                          w_idx:w_idx + WIDTH, 
                                          d_idx:d_idx + DEPTH, :] = pred_whole[h_idx:h_idx + HEIGHT, 
                                                                              w_idx:w_idx + WIDTH, 
                                                                              d_idx:d_idx + DEPTH, :] + seg_label
                            avg_whole[h_idx:h_idx + HEIGHT, 
                                        w_idx:w_idx + WIDTH, 
                                        d_idx:d_idx + DEPTH, :] = avg_whole[h_idx:h_idx + HEIGHT, 
                                                                            w_idx:w_idx + WIDTH, 
                                                                            d_idx:d_idx + DEPTH, :] + avg_block

                              # flip-3.
                            batch_images = batch_images0[:, :, ::-1, :, :]
                            batch_masks = batch_masks0[:, :, ::-1, :, :]
                            
                            seg_label = self.sess.run(
                                  self.seg_finelabel1,
                                  feed_dict={self.real_syndata: batch_images,
                                              self.mask: batch_masks,
                                              self.input_code: input_code,
                                              self.indicate_code: indicate_code}
                              )
                            
                            seg_label = np.array(seg_label[:, :, ::-1, :, :]).astype('float16').reshape([HEIGHT, WIDTH, DEPTH, NUM_CLS])
                            
                            pred_whole[h_idx:h_idx + HEIGHT, 
                                          w_idx:w_idx + WIDTH, 
                                          d_idx:d_idx + DEPTH, :] = pred_whole[h_idx:h_idx + HEIGHT, 
                                                                              w_idx:w_idx + WIDTH, 
                                                                              d_idx:d_idx + DEPTH, :] + seg_label
                            avg_whole[h_idx:h_idx + HEIGHT, 
                                        w_idx:w_idx + WIDTH, 
                                        d_idx:d_idx + DEPTH, :] = avg_whole[h_idx:h_idx + HEIGHT, 
                                                                            w_idx:w_idx + WIDTH, 
                                                                            d_idx:d_idx + DEPTH, :] + avg_block

                              # flip-4.
                            batch_images = batch_images0[:, :, :, ::-1, :]
                            batch_masks = batch_masks0[:, :, :, ::-1, :]
                            
                            seg_label = self.sess.run(
                                  self.seg_finelabel1,
                                  feed_dict={self.real_syndata: batch_images,
                                              self.mask: batch_masks,
                                              self.input_code: input_code,
                                              self.indicate_code: indicate_code}
                              )
                            
                            seg_label = np.array(seg_label[:, :, :, ::-1, :]).astype('float16').reshape([HEIGHT, WIDTH, DEPTH, NUM_CLS])
                            
                            pred_whole[h_idx:h_idx + HEIGHT, 
                                          w_idx:w_idx + WIDTH, 
                                          d_idx:d_idx + DEPTH, :] = pred_whole[h_idx:h_idx + HEIGHT, 
                                                                              w_idx:w_idx + WIDTH, 
                                                                              d_idx:d_idx + DEPTH, :] + seg_label
                            avg_whole[h_idx:h_idx + HEIGHT, 
                                        w_idx:w_idx + WIDTH, 
                                        d_idx:d_idx + DEPTH, :] = avg_whole[h_idx:h_idx + HEIGHT, 
                                                                            w_idx:w_idx + WIDTH, 
                                                                            d_idx:d_idx + DEPTH, :] + avg_block

                              # flip-5.
                            batch_images = batch_images0[:, ::-1, ::-1, :, :]
                            batch_masks = batch_masks0[:, ::-1, ::-1, :, :]
                            
                            seg_label = self.sess.run(
                                  self.seg_finelabel1,
                                  feed_dict={self.real_syndata: batch_images,
                                              self.mask: batch_masks,
                                              self.input_code: input_code,
                                              self.indicate_code: indicate_code}
                              )
                            
                            seg_label = np.array(seg_label[:, ::-1, ::-1, :, :]).astype('float16').reshape([HEIGHT, WIDTH, DEPTH, NUM_CLS])
                            
                            pred_whole[h_idx:h_idx + HEIGHT, 
                                          w_idx:w_idx + WIDTH, 
                                          d_idx:d_idx + DEPTH, :] = pred_whole[h_idx:h_idx + HEIGHT, 
                                                                              w_idx:w_idx + WIDTH, 
                                                                              d_idx:d_idx + DEPTH, :] + seg_label
                            avg_whole[h_idx:h_idx + HEIGHT, 
                                        w_idx:w_idx + WIDTH, 
                                        d_idx:d_idx + DEPTH, :] = avg_whole[h_idx:h_idx + HEIGHT, 
                                                                            w_idx:w_idx + WIDTH, 
                                                                            d_idx:d_idx + DEPTH, :] + avg_block

                              # flip-6.
                            batch_images = batch_images0[:, ::-1, :, ::-1, :]
                            batch_masks = batch_masks0[:, ::-1, :, ::-1, :]
                            
                            seg_label = self.sess.run(
                                  self.seg_finelabel1,
                                  feed_dict={self.real_syndata: batch_images,
                                              self.mask: batch_masks,
                                              self.input_code: input_code,
                                              self.indicate_code: indicate_code}
                              )
                            
                            seg_label = np.array(seg_label[:, ::-1, :, ::-1, :]).astype('float16').reshape([HEIGHT, WIDTH, DEPTH, NUM_CLS])
                            
                            pred_whole[h_idx:h_idx + HEIGHT, 
                                          w_idx:w_idx + WIDTH, 
                                          d_idx:d_idx + DEPTH, :] = pred_whole[h_idx:h_idx + HEIGHT, 
                                                                              w_idx:w_idx + WIDTH, 
                                                                              d_idx:d_idx + DEPTH, :] + seg_label
                            avg_whole[h_idx:h_idx + HEIGHT, 
                                        w_idx:w_idx + WIDTH, 
                                        d_idx:d_idx + DEPTH, :] = avg_whole[h_idx:h_idx + HEIGHT, 
                                                                            w_idx:w_idx + WIDTH, 
                                                                            d_idx:d_idx + DEPTH, :] + avg_block

                              # flip-7.
                            batch_images = batch_images0[:, :, ::-1, ::-1, :]
                            batch_masks = batch_masks0[:, :, ::-1, ::-1, :]
                            
                            seg_label = self.sess.run(
                                  self.seg_finelabel1,
                                  feed_dict={self.real_syndata: batch_images,
                                              self.mask: batch_masks,
                                              self.input_code: input_code,
                                              self.indicate_code: indicate_code}
                              )
                            
                            seg_label = np.array(seg_label[:, :, ::-1, ::-1, :]).astype('float16').reshape([HEIGHT, WIDTH, DEPTH, NUM_CLS])
                            
                            pred_whole[h_idx:h_idx + HEIGHT, 
                                          w_idx:w_idx + WIDTH, 
                                          d_idx:d_idx + DEPTH, :] = pred_whole[h_idx:h_idx + HEIGHT, 
                                                                              w_idx:w_idx + WIDTH, 
                                                                              d_idx:d_idx + DEPTH, :] + seg_label
                            avg_whole[h_idx:h_idx + HEIGHT, 
                                        w_idx:w_idx + WIDTH, 
                                        d_idx:d_idx + DEPTH, :] = avg_whole[h_idx:h_idx + HEIGHT, 
                                                                            w_idx:w_idx + WIDTH, 
                                                                            d_idx:d_idx + DEPTH, :] + avg_block

                              # flip-8.
                            batch_images = batch_images0[:, ::-1, ::-1, ::-1, :]
                            batch_masks = batch_masks0[:, ::-1, ::-1, ::-1, :]
                            
                            seg_label = self.sess.run(
                                  self.seg_finelabel1,
                                  feed_dict={self.real_syndata: batch_images,
                                              self.mask: batch_masks,
                                              self.input_code: input_code,
                                              self.indicate_code: indicate_code}
                              )
                            
                            seg_label = np.array(seg_label[:,::-1,::-1,::-1,:]).astype('float16').reshape([HEIGHT, WIDTH, DEPTH, NUM_CLS])
                            
                            pred_whole[h_idx:h_idx + HEIGHT, 
                                          w_idx:w_idx + WIDTH, 
                                          d_idx:d_idx + DEPTH, :] = pred_whole[h_idx:h_idx + HEIGHT, 
                                                                              w_idx:w_idx + WIDTH, 
                                                                              d_idx:d_idx + DEPTH, :] + seg_label
                            avg_whole[h_idx:h_idx + HEIGHT, 
                                        w_idx:w_idx + WIDTH, 
                                        d_idx:d_idx + DEPTH, :] = avg_whole[h_idx:h_idx + HEIGHT, 
                                                                            w_idx:w_idx + WIDTH, 
                                                                            d_idx:d_idx + DEPTH, :] + avg_block

                pred_whole = pred_whole / avg_whole

                label_head_output = inputLabelAll.header
                label_affine_output = inputLabelAll.affine

                saveLab0 = nib.Nifti1Image(pred_whole[:,:,:,0], label_affine_output, label_head_output)
                nib.save(saveLab0, '{}/id{:0>3d}_input{:0>2d}_esLabelWT.nii.gz'.format(test_dir, teId, inputInd))
                
                saveLab1 = nib.Nifti1Image(pred_whole[:,:,:,1], label_affine_output, label_head_output)
                nib.save(saveLab1, '{}/id{:0>3d}_input{:0>2d}_esLabelTC.nii.gz'.format(test_dir, teId, inputInd))
                
                saveLab2 = nib.Nifti1Image(pred_whole[:,:,:,2], label_affine_output, label_head_output)
                nib.save(saveLab2, '{}/id{:0>3d}_input{:0>2d}_esLabelET.nii.gz'.format(test_dir, teId, inputInd))

                endtime_test=time.time()
                print(endtime_test - starttime_test)
                
                gtlabtmp0 = np.zeros(inputLabelAll.shape, dtype=np.int16)
                gtlabtmp0[volLabel>0] = 1
                gtLab0 = nib.Nifti1Image(gtlabtmp0, label_affine_output, label_head_output)
                nib.save(gtLab0, '{}/id{:0>3d}_gtLabelWT.nii.gz'.format(test_dir, teId))
                
                gtlabtmp1 = np.zeros(inputLabelAll.shape, dtype=np.int16)
                gtlabtmp1[volLabel==4] = 1
                gtlabtmp1[volLabel==1] = 1
                gtLab1 = nib.Nifti1Image(gtlabtmp1, label_affine_output, label_head_output)
                nib.save(gtLab1, '{}/id{:0>3d}_gtLabelTC.nii.gz'.format(test_dir, teId))
                
                gtlabtmp2 = np.zeros(inputLabelAll.shape, dtype=np.int16)
                gtlabtmp2[volLabel==4] = 1
                gtLab2 = nib.Nifti1Image(gtlabtmp2, label_affine_output, label_head_output)
                nib.save(gtLab2, '{}/id{:0>3d}_gtLabelET.nii.gz'.format(test_dir, teId))
