import numpy as np
import tensorflow as tf
from datetime import datetime
import pickle, os

from policy import Policy
from value_function import NNValueFunction

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
        # pickle and save scaler - NOTE: this file includes some other variables too
        mypath = self.checkpoints_dir+"/scaler/"+self.init_time
        makedirs("/".join(mypath.split("/")[:-1]))
        print("saving scaler checkpoint to:", mypath, "episode:", episode)
        with open(mypath+"-"+str(episode)+".scaler", 'wb') as f:
            pickle.dump((scaler, episode, policy.obs_dim, policy.act_dim, policy.kl_targ, self.init_time), f)

        mypath = self.checkpoints_dir+"/policy/"+self.init_time
        makedirs("/".join(mypath.split("/")[:-1]))
        print("saving policy checkpoint to:", mypath, "episode:", episode)
        with policy.g.as_default():
            saver = tf.train.Saver()
            saver.save(policy.sess, mypath, global_step=episode)

        mypath = self.checkpoints_dir+"/val_func/"+self.init_time
        makedirs("/".join(mypath.split("/")[:-1]))
        print("saving val_func checkpoint to:", mypath, "episode:", episode)
        with val_func.g.as_default():
            saver = tf.train.Saver()
            saver.save(val_func.sess, mypath, global_step=episode)

    def restore(self, restore_path):
        name=restore_path.split("/")[-1] # remove preceding/path/components/to/name

        # unpickle and restore scaler - NOTE: this file includes some other variables too
        mypath = "saves/scaler/"+name
        print("restoring scaler checkpoint from:", mypath)
        with open(mypath+".scaler", 'rb') as f:
            (scaler, episode, obs_dim, act_dim, kl_targ, self.init_time) = pickle.load(f)

        # policy
        mypath = "saves/policy/"+name
        print("restoring policy checkpoint from:", mypath)
        policy = Policy(obs_dim, act_dim, kl_targ, restore_path=mypath)
        print("restored policy:")
        Checkpoint.dump_vars(policy.g)

        # val_func
        mypath = "saves/val_func/"+name
        print("restoring val_func checkpoint from:", mypath)
        val_func = NNValueFunction(obs_dim, restore_path=mypath)
        print("restored val_func:")
        Checkpoint.dump_vars(val_func.g)

        print("finished restore.")
        return(policy, val_func, scaler, episode, obs_dim, act_dim, kl_targ)

def makedirs(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass



#########################################################

if 0:
    cache = {}
    def set_wt(sess, graph, var, val):
        k = str(var)
        kc = str(sess)+str(graph)+str(var)
        if not kc in cache:
            with graph.as_default():
                ph = tf.placeholder(val.dtype, shape=val.shape)
                op = var.assign(ph)
                cache[kc] = [ var, ph, op]
        _, ph, op = cache[kc]
        sess.run(op, feed_dict={ph: val})

    def get_variable_by_name(name):
        list = [v for v in tf.local_variables() if v.name == name]
        print("NAME:",name)
        print("vars:",tf.local_variables())
        print("LIST:",list)
        return(list[0])

    def save_old3(self, policy, val_func, scaler, episode):
        graph = policy.g
        sess = policy.sess
        #print ("XXX")
        #print(g.get_operations())
        #print ("XXX")
        for var in sorted(graph.get_collection('trainable_variables'),key=lambda x: x.name):
            w = sess.run(var)
            print(var.name, var, type(w), w)
            var2 = Checkpoint.get_variable_by_name(var.name)
            print(var is var2)
            w2 = sess.run(var2)
            print(var2.name, var2, type(w2), w2)
            print(var is var2)
            #set_wt(sess, graph, 
        exit(0)


    def save_old(self, policy, val_func, scaler, episode):
        mypath = self._savepath(episode)
        print("saving checkpoint to:", mypath)
        Checkpoint.dump_vars(policy.g)

        # vars = policy.g.get_collection('trainable_variables')
        # signature_def_map={x.name:x for x in vars}

        #policy_signature = (tf.saved_model.signature_def_utils.build_signature_def(
        #    inputs={'obs_ph': tf.saved_model.utils.build_tensor_info(policy.obs_ph)},
        #    outputs={'act_ph': tf.saved_model.utils.build_tensor_info(policy.act_ph)}
        #))
        
        #signature_def_map = {
        #    "policy_signature": policy_signature
        #}

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

    def restore_old(self, policy, val_func, scaler, restore_path):
        #mypath = self.checkpoints_dir+"/"+restore_path
        mypath = restore_path

        print("restoring checkpoint from:", mypath)

        from policy import Policy
        from value_function import NNValueFunction
        
        policy = Policy(policy.obs_dim, policy.act_dim, policy.kl_targ, restore_flag=True)
        with policy.g.as_default():
            print("0000000A")
            Checkpoint.dump_vars(policy.g)
            tf.saved_model.loader.load(policy.sess, [tf.saved_model.tag_constants.TRAINING], mypath+".policy")
            print("1111111A")
            Checkpoint.dump_vars(policy.g)
        policy._placeholders()
        print("YYYY:",policy.obs_ph)
            
        val_func = NNValueFunction(val_func.obs_dim, restore_flag=True)
        with val_func.g.as_default():
            print("2222222A")
            Checkpoint.dump_vars(val_func.g)
            tf.saved_model.loader.load(val_func.sess, [tf.saved_model.tag_constants.TRAINING], mypath+".val_func")
            print("3333333A")
            Checkpoint.dump_vars(val_func.g)
        val_func._placeholders()
        print("YYYY:",val_func.obs_ph)

        # unpickle and restore scaler
        with open(mypath+".scaler", 'rb') as f:
            (scaler, episode) = pickle.load(f)

        print("FINISHED RESTORE")
        return(policy, val_func, scaler, episode)
        
