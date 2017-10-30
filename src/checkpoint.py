import numpy as np
import tensorflow as tf
from datetime import datetime
import pickle

class Checkpoint(object):
    """ save and restore checkpoints """
    def __init__(self, checkpoints_dir, init_time):
        self.checkpoints_dir = checkpoints_dir
        self.init_time = init_time

    def _savepath(self, episode):
        return self.checkpoints_dir+"/"+self.init_time+"-"+str(episode)

    def dump_vars(graph):
        for var in sorted(graph.get_collection('trainable_variables'),key=lambda x: x.name):
            print(var.name, var)
        for var in sorted(graph.get_collection('global_variables'),key=lambda x: x.name):
            print(var.name, var)
        for var in sorted(graph.get_collection('local_variables'),key=lambda x: x.name):
            print(var.name, var)
    
    def save(self, policy, val_func, scaler, episode):
        mypath = self._savepath(episode)
        print("saving checkpoint to:", mypath)
        Checkpoint.dump_vars(policy.g)

        # vars = policy.g.get_collection('trainable_variables')
        # signature_def_map={x.name:x for x in vars}

        with policy.g.as_default():
            builder = tf.saved_model.builder.SavedModelBuilder(mypath+".policy")
            builder.add_meta_graph_and_variables(policy.sess,
                                                 [tf.saved_model.tag_constants.TRAINING],
                                                 #signature_def_map = {
                                                 #    "policy": tf.saved_model.signature_def_utils.predict_signature_def(
                                                 #        inputs= {"obs_ph": policy.obs_ph},
                                                 #        outputs= {"act_ph": policy.act_ph})
                                                 #},
                                                 signature_def_map=None,
                                                 assets_collection=None
            )
            builder.save()
            
        with val_func.g.as_default():
            builder = tf.saved_model.builder.SavedModelBuilder(mypath+".val_func")
            builder.add_meta_graph_and_variables(val_func.sess,
                                                 [tf.saved_model.tag_constants.TRAINING],
                                                 #signature_def_map = {
                                                 #    "val_func": tf.saved_model.signature_def_utils.predict_signature_def(
                                                 #        inputs= {"obs_ph": val_func.obs_ph},
                                                 #        outputs= {"val_ph": val_func.val_ph})
                                                 #},
                                                 signature_def_map=None,
                                                 assets_collection=None
            )
            builder.save()

        # pickle and save scaler
        with open(mypath+".scaler", 'wb') as f:
            pickle.dump((scaler, episode), f)

    def restore(self, policy, val_func, scaler, restore_path):
        #mypath = self.checkpoints_dir+"/"+restore_path
        mypath = restore_path

        print("restoring checkpoint from:", mypath)

        from policy import Policy
        from value_function import NNValueFunction
        
        policy.sess.close()
        policy = Policy(policy.obs_dim, policy.act_dim, policy.kl_targ)
        with policy.g.as_default():
            print("0000000")
            Checkpoint.dump_vars(tf.get_default_graph())
            tf.saved_model.loader.load(policy.sess, [tf.saved_model.tag_constants.TRAINING], mypath+".policy")
            print("11111111")
            Checkpoint.dump_vars(tf.get_default_graph())

        val_func.sess.close()
        val_func = NNValueFunction(val_func.obs_dim)
        with val_func.g.as_default():
            print("2222222")
            Checkpoint.dump_vars(tf.get_default_graph())
            tf.saved_model.loader.load(val_func.sess, [tf.saved_model.tag_constants.TRAINING], mypath+".val_func")
            print("3333333")
            Checkpoint.dump_vars(tf.get_default_graph())

        # unpickle and restore scaler
        with open(mypath+".scaler", 'rb') as f:
            (scaler, episode) = pickle.load(f)

        return(policy, val_func, scaler, episode)
        
class CheckpointOld(object):
    """ save and restore checkpoints """
    def __init__(self, policy, scaler, val_func, export_dir):
        self.policy = policy
        self.scaler = scaler
        self.val_func = val_func
        self.export_dir = export_dir

        now = datetime.utcnow().strftime("%b-%d_%H:%M:%S")
        self.path = export_dir+"/"+now
        
    def save(self, global_step=None):
        # Save model weights to disk
        save_path = self.policy.saver.save(self.policy.sess, self.path, global_step=global_step)
        print("Models saved in file: %s" % save_path)

    def restore(self, global_step=None):
        latest_path=tf.train.latest_checkpoint(self.path)
        print("latest_path: %s" % latest_path)
        # Restore model weights from previously saved model
        self.policy.saver.restore(self.policy.sess, latest_path)
        print("Models restored from file: %s" % latest_path)

    
    
