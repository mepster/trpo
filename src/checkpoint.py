import numpy as np
import tensorflow as tf
from datetime import datetime
import pickle

class Checkpoint(object):
    """ save and restore checkpoints """
    def __init__(self, checkpoints_dir):
        now = datetime.utcnow().strftime("%b-%d_%H:%M:%S")
        self.dir = checkpoints_dir+"/timestamp"
        #self.dir = checkpoints_dir+"/"+now

    def _mypath(self, global_step=None):
        if global_step:
            return self.dir+"-"+global_step
        else:
            return self.dir
        
    def save(self, policy, val_func, scaler, global_step=None):
        mypath = self._mypath(global_step)
        print("saving checkpoint to:", mypath)

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
            pickle.dump(scaler, f)


    def restore(self, policy, val_func, scaler, global_step=None):
        mypath = self._mypath(global_step)
        print("restoring checkpoint from:", mypath)

        with policy.g.as_default():
            policy.sess.close()
            policy.sess=tf.Session()
            tf.saved_model.loader.load(policy.sess, [tf.saved_model.tag_constants.TRAINING], mypath+".policy")
            policy.g = policy.sess.graph
            policy.sess.run(tf.global_variables_initializer())

        with val_func.g.as_default():
            val_func.sess.close()
            val_func.sess=tf.Session()
            tf.saved_model.loader.load(val_func.sess, [tf.saved_model.tag_constants.TRAINING], mypath+".val_func")
            val_func.g = val_func.sess.graph
            val_func.sess.run(tf.global_variables_initializer())

        # unpickle and restore scaler
        with open(mypath+".scaler", 'rb') as f:
            scaler = pickle.load(f)

        return(policy, val_func, scaler)
        
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

    
    
